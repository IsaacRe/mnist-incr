# Implementation details follow Li et. al. 2016 'Convergent: Do Different Neural Networks Learn The Same
# Representations?'. https://arxiv.org/abs/1511.07543 .
import torch
import numpy as np


class CorrelationTracker:
    """
    Class that keeps track of feature activations in a particular layer of a CNN and computes correlation both
    between features in a layer of a single trained CNN (within-net correlation) and between features in two separately
    trained CNN's (between-net correlation).
    Option to attach forward hook to the passed network and save feature activations during forward pass for later
    computation (TODO)
    IMPORTANT: Class uses 'feature' attribute of the passed net objects. Nets must have 'feature' implemented for
    computation of correlation matrix to work
    """

    def __init__(self, net=None, feature_idx=None, store_on_gpu=False):
        """
        Initialize a CorrelationTracker object. If net and feature_idx are passed, will attempt to setup on-line
        activation tracking.
        :param net: the CNN for which to track activations. Must have attribute, features, for access to intermediate
                    feature output
        :param feature_idx: the index of the feature output to track
        :param store_on_gpu: whether to store activation values on gpu (memory-intensive)
        """
        self.net = net
        self.feature_idx = feature_idx
        self.n_feat = None

        if net is not None:
            self.n_feat = net.features[feature_idx].out_features

        # TODO implement on-line correlation computation
        self.store_on_gpu = store_on_gpu
        self.activations = None
        # For controlling what happens the next time our forward hook gets called
        # 'pass': do nothing
        # 'save': save (overwrite) per-feature, per-sample activation values to self.activations
        self.hook_mode = 'pass'

    def _get_feature_means(self, dataloader, feature_idx, *nets):
        """
        Returns mean of each feature over the passed data
        :param dataloader: the dataloader
        :return: vector of means for each feature over the data
        """
        assert len(nets) > 0, "No networks passed!"
        mu = [None] * len(nets)
        total = 0
        for i, (idx, images, labels) in enumerate(dataloader):
            total += len(labels)
            for j, net in enumerate(nets):
                for k, layer in enumerate(net.features):
                    out = layer(images)
                    if feature_idx == k:
                        break
                out = torch.mean(out, dim=(0, 1, 2))  # Average over batch and spatial dimensions
                if mu[j] is None:
                    mu[j] = out
                else:
                    mu[j] += out
        mu = [m / total for m in mu]
        return mu

    def set_mode(self, mode):
        """
        Configure the action to be taken during the next forward pass of self.net
        :param mode: the action to be taken
        :return:
        """
        self.hook_mode = mode

    def within_net_corr(self, dataloader, net, feature_idx=None):
        """
        Compute the within-net correlation on the provided data.
        :param dataloader: the dataloader
        :param net: the trained CNN
        :param feature_idx: index of the layer to use. If None, will use self.feature_idx
        :return: the correlation matrix
        """
        feature_idx = self.feature_idx if feature_idx is None else feature_idx
        assert feature_idx is not None, "Attribute, 'feature_idx' is None and no feature_idx was explicitly provided."

        # get mean activation over the data for each feature
        [mu] = self._get_feature_means(dataloader, feature_idx, net)

        corr_matr = None
        sig = None
        total = None
        for i, (idx, images, labels) in enumerate(dataloader):
            total += len(labels)
            out = images
            for j, layer in enumerate(net.features):
                out = layer(out)
                if feature_idx == j:
                    break
            # TODO make sure dimension 3 is what I think it is
            out = out.flatten(dim=(0, 1, 2))  # [(batch * width * height) X filters]
            assert self.n_feat == out.shape[1], "Output of layer %d number of features does not match self.n_feat"
            n_feat = self.n_feat

            # get x_i - mu_i, the deviation of each feature output from its mean
            deviation = out - mu  # [(B * W * H) X F]
            # corr_matr_temp_ij = deviation_i * deviation_j
            deviation_expanded = deviation.unsqueeze(2).repeat(1, 1, n_feat)  # [(B * W * H) X F X F]
            corr_matr_temp = deviation_expanded * deviation_expanded.view(0, 2, 1)  # [(B * W * H) X F X F]
            # sum over batch and spatial dimensions
            corr_matr_temp = torch.sum(corr_matr_temp, dim=0)  # [F X F]
            if corr_matr is None:
                corr_matr = corr_matr_temp
            else:
                corr_matr += corr_matr_temp

            # accumulate feature-wise variances for this match
            if sig is None:
                sig = torch.mean(deviation**2, dim=0)  # [F]
            else:
                sig += torch.mean(deviation**2, dim=0)  # [F]

        sig = (sig / total)**0.5  # feature-wise std deviation
        corr_matr /= total  # un-normalized correlation matrix

        # sig_matr_ij = sig_i * sig_j
        sig_expanded = sig.unsqueeze(1).repeat(1, n_feat)  # [F X F]
        sig_matr = sig_expanded * sig_expanded.view(1, 0)
        corr_matr /= sig_matr  # normalized correlation matrix

        return corr_matr

    def between_net_corr(self, dataloader, net_1, net_2, feature_idx=None):
        """
        Compute the between-net correlation for net_1, net_2 on the provided data
        :param dataloader: the dataloader
        :param net_1: the first trained CNN
        :param net_2: the second trained CNN
        :return: the correlation matrix
        """
        feature_idx = self.feature_idx if feature_idx is None else feature_idx
        assert feature_idx is not None, "Attribute, 'feature_idx' is None and no feature_idx was explicitly provided."

        # get mean activations over the data for each feature for nets 1 and 2
        mu = self._get_feature_means(dataloader, feature_idx, net_1, net_2)

        corr_matr = None
        sigs = [None, None]
        total = None
        for i, (idx, images, labels) in enumerate(dataloader):
            total += len(labels)
            deviations = [None, None]
            for j, net in enumerate([net_1, net_2]):
                out = images
                for k, layer in enumerate(net.features):
                    out = layer(out)
                    if feature_idx == k:
                        break

                out = out.flatten(dim=(0, 1, 2))  # [(batch * width * height) X filters]
                assert self.n_feat == out.shape[1], "Output of layer %d number of features does not match self.n_feat"
                n_feat = self.n_feat

                # get x_i - mu_i, the deviation of each feature output from its mean
                deviations[j] = out - mu[j]  # [(B * W * H) X F]

            # corr_matr_temp_ij = deviations[0]_i * deviations[1]_j
            dev_1_expanded, dev_2_expanded = [dev.unsqueeze(2).repeat(1, 1, n_feat) for dev in deviations]  # [(B * W * H) X F X F] (each)
            corr_matr_temp = dev_1_expanded * dev_2_expanded.view(0, 2, 1)  # [(B * W * H) X F X F]
            # sum over batch and spatial dimensions
            corr_matr_temp = torch.sum(corr_matr_temp, dim=0)  # [F X F]
            if corr_matr is None:
                corr_matr = corr_matr_temp
            else:
                corr_matr += corr_matr_temp

            # accumulate feature-wise variances for this match
            if sigs == [None, None]:
                sigs = [torch.mean(dev**2, dim=0) for dev in deviations] # [F] (each)
            else:
                sigs += [torch.mean(dev**2, dim=0) for dev in deviations] # [F] (each)

        sig = [(sig / total)**0.5 for sig in sigs]  # feature-wise std deviation
        corr_matr /= total  # un-normalized correlation matrix

        # sig_matr_ij = sigs[0]_i * sigs[1]_j
        # TODO potential bug point
        sig_matr = sigs[0].unsqueeze(1).repeat(1, n_feat) * sigs[1].unsqueeze(0).repeat(n_feat, 1)
        corr_matr /= sig_matr  # normalized correlation matrix

        return corr_matr


class FeatureMatcher:
    """
    Class that conducts feature matching between features of the same CNN or of two separate CNN's, given a correlation
    matrix of the feature activations.
    Once within/between-net correlation is obtained, the class can conduct:
        one-to-one feature matching (TODO implement)
        one-to-many feature matching (TODO implement)
        many-to-many feature matching (TODO implement)
    """

    def one2one(self, corr_matr, replace=True):
        """
        Uses bipartite semi-matching to find maximally correlated feature for each feature
        :param corr_matr: the correlation matrix
        :param replace: if True, when finding match for each feature, will consider all features, otherwise will
                        consider only features that have not yet been selected
        :return: list, [(f_1, f_2) for every f_1], of maximally correlated feature pairs
        """
        # TODO implement bipartite matching/semi-matching
        pass

    def one2many(self, corr_matr):
        # TODO
        pass

    def many2many(self, corr_matr):
        # TODO
        pass
