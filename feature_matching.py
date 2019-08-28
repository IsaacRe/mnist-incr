# Implementation details follow 'Convergent: Do Different Neural Networks Learn The Same Representations?', Li et. al.
# (2016).
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
        :param net: the trained CNN to use
        :param feature_idx: index of the layer to use. If None, will use self.feature_idx
        :return: the correlation matrix
        """
        feature_idx = self.feature_idx if feature_idx is None else feature_idx
        assert feature_idx is not None, "Attribute, 'feature_idx' is None and no feature_idx was explicitly provided."

        # get mean activation over the data of each feature
        [mu] = self._get_feature_means(dataloader, feature_idx, net)

        corr_matr = None
        sig = None
        total = None
        for i, (idx, images, labels) in enumerate(dataloader):
            total += len(labels)
            for j, layer in enumerate(net.features):
                out = layer(images)
                if feature_idx == j:
                    break
            out = torch.mean(out, dim=(0, 1, 2))  # average across batch and spatial dimensions
            if sig is None:
                sig = (out - mu)**2
            else:
                sig += (out - mu)**2
            # TODO Compute A where A_ij = (out_i - mu_i) * (out_j - mu_j)
            corr_matr += A
        sig /= total
        corr_matr /= total

        # TODO compute S where S_ij = sig_i * sig_j
        # TODO compute corr_matr /= S (elem-wise division)

        return corr_matr

    def between_net_corr(self, dataloader, net_1=None, net_2=None, feature_idx=None):
        """
        Return the correlation matrix of features. If
        :param net_1:
        :param dataloader:
        :param net_2:
        :return: the correlation matrix
        """
        feature_idx = self.feature_idx if feature_idx is None else feature_idx
        assert feature_idx is not None, "Attribute, 'feature_idx' is None and no feature_idx was explicitly provided."

        # TODO
        pass


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
