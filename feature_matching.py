# Implementation details follow Li et. al. 2016 'Convergent Learning: Do Different Neural Networks Learn The Same
# Representations?'. https://arxiv.org/abs/1511.07543 .
import argparse
import torch
import numpy as np
from test import get_batch_suite
from net import Net


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
            self.device = net.features[feature_idx].weight.device

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
                device = net.features[feature_idx].weight.device
                out = images.to(device)
                for k, layer in enumerate(net.features):
                    out = layer(out)
                    if feature_idx == k:
                        break
                out = out.data
                if len(out.shape) > 2:
                    out = torch.mean(out, dim=(2, 3))  # Average over spatial dims sum over batch dims
                out = torch.sum(out, dim=0)
                if mu[j] is None:
                    mu[j] = out
                else:
                    mu[j] += out
        mu = [m / total for m in mu]  # now average over batch dimension
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
        device = net.features[feature_idx].weight.device

        # get mean activation over the data for each feature
        [mu] = self._get_feature_means(dataloader, feature_idx, net)

        corr_matr = None
        sig = None
        total = 0
        feat_w = 1
        for i, (idx, images, labels) in enumerate(dataloader):
            total += len(labels)
            out = images.to(device)
            for j, layer in enumerate(net.features):
                out = layer(out)
                if feature_idx == j:
                    break
            out = out.data
            feat_w = 1
            if len(out.shape) > 2:
                feat_w = out.shape[2]
                out = out.transpose(1, 3).flatten(start_dim=0, end_dim=2)  # [(batch * width * height) X filters]
            if self.n_feat is None:
                self.n_feat = out.shape[1]
            assert self.n_feat == out.shape[1], "Output of layer %d number of features does not match self.n_feat"
            n_feat = self.n_feat

            # get x_i - mu_i, the deviation of each feature output from its mean
            deviation = out - mu  # [(B * W * H) X F]
            # corr_matr_temp_ij = deviation_i * deviation_j
            deviation_expanded = deviation.unsqueeze(2).repeat(1, 1, n_feat)  # [(B * W * H) X F X F]
            corr_matr_temp = deviation_expanded * deviation_expanded.transpose(1, 2)  # [(B * W * H) X F X F]
            # sum over batch and spatial dimensions
            corr_matr_temp = torch.sum(corr_matr_temp, dim=0)  # [F X F]
            if corr_matr is None:
                corr_matr = corr_matr_temp
            else:
                corr_matr += corr_matr_temp

            # accumulate feature-wise variances for this match
            if sig is None:
                sig = torch.sum(deviation**2, dim=0)  # [F]
            else:
                sig += torch.sum(deviation**2, dim=0)  # [F]

        total_feats = feat_w**2  # total number of outputs for each filter across both spatial dimensions
        sig = (sig / total / total_feats)**0.5  # feature-wise std deviation
        corr_matr /= (total * total_feats)  # un-normalized correlation matrix

        # sig_matr_ij = sig_i * sig_j
        sig_expanded = sig.unsqueeze(1).repeat(1, n_feat)  # [F X F]
        sig_matr = sig_expanded * sig_expanded.transpose(0, 1)
        corr_matr /= sig_matr  # normalized correlation matrix

        return corr_matr.cpu().numpy()

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
        total = 0
        feat_w = 1
        for i, (idx, images, labels) in enumerate(dataloader):
            total += len(labels)
            deviations = [None, None]
            for j, net in enumerate([net_1, net_2]):
                device = net.features[feature_idx].weight.device
                out = images.to(device)
                for k, layer in enumerate(net.features):
                    out = layer(out)
                    if feature_idx == k:
                        break
                out = out.data
                feat_w = 1
                if len(out.shape) > 2:
                    feat_w = out.shape[2]
                    out = out.transpose(1, 3).flatten(start_dim=0, end_dim=2)  # [(batch * width * height) X filters]
                if self.n_feat is None:
                    self.n_feat = out.shape[1]
                assert self.n_feat == out.shape[1], "Output of layer %d number of features does not match self.n_feat"
                n_feat = self.n_feat

                # get x_i - mu_i, the deviation of each feature output from its mean
                deviations[j] = out - mu[j]  # [(B * W * H) X F]

            # corr_matr_temp_ij = deviations[0]_i * deviations[1]_j
            dev_1_expanded, dev_2_expanded = [dev.unsqueeze(2).repeat(1, 1, n_feat) for dev in deviations]  # [(B * W * H) X F X F] (each)
            corr_matr_temp = dev_1_expanded * dev_2_expanded.transpose(1, 2)  # [(B * W * H) X F X F]
            # sum over batch and spatial dimensions
            corr_matr_temp = torch.sum(corr_matr_temp, dim=0)  # [F X F]
            if corr_matr is None:
                corr_matr = corr_matr_temp
            else:
                corr_matr += corr_matr_temp

            # accumulate feature-wise variances for this match
            if sigs[0] is None:
                sigs = [torch.sum(dev**2, dim=0) for dev in deviations]  # [F] (each)
            else:
                sigs = [sig + torch.sum(dev**2, dim=0) for sig, dev in zip(sigs, deviations)]  # [F] (each)

        total_feats = feat_w**2  # total number of outputs for each filter across both spatial dimensions
        sigs = [(sig / total / total_feats)**0.5 for sig in sigs]  # feature-wise std deviation
        corr_matr /= (total * total_feats)  # un-normalized correlation matrix

        # sig_matr_ij = sigs[0]_i * sigs[1]_j
        sig_1_expanded, sig_2_expanded = [sig.unsqueeze(1).repeat(1, n_feat) for sig in sigs]
        sig_matr = sig_1_expanded * sig_2_expanded.transpose(0, 1)
        corr_matr /= sig_matr  # normalized correlation matrix

        return corr_matr.cpu().numpy()


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
        matches = np.ndarray(corr_matr.shape[0])
        correlations = np.ndarray(corr_matr.shape[0])
        for i in range(corr_matr.shape[0]):
            j = np.argmax(corr_matr[i])
            if not replace:
                corr_matr[:, j] = -np.ones((corr_matr.shape[1]))
            matches[i] = j
            correlations[i] = corr_matr[i, j]
        return matches, correlations

    def one2many(self, corr_matr):
        # TODO
        pass

    def many2many(self, corr_matr):
        # TODO
        pass


def threshold_correlations(corr_matr, threshold):
    return len(np.where(np.abs(corr_matr) > threshold)[0]) / corr_matr.size


def within_net_correlation(dataloader, net, feature_idx, threshold=0.7):
    corr_tracker = CorrelationTracker()
    corr_matr = corr_tracker.within_net_corr(dataloader, net, feature_idx=feature_idx)
    return threshold_correlations(corr_matr, threshold), corr_matr


def between_net_correlation(dataloader, net_1, net_2, feature_idx, threshold=0.7):
    corr_tracker = CorrelationTracker()
    corr_matr = corr_tracker.between_net_corr(dataloader, net_1, net_2, feature_idx=feature_idx)
    return threshold_correlations(corr_matr, threshold), corr_matr


def match(*args):
    _, corr_matr = between_net_correlation(*args)
    feat_match = FeatureMatcher()
    matches, corr = feat_match.one2one(corr_matr)
    return matches, corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nets', type=str, nargs='+', help='paths to model files to use')
    parser.add_argument('--mode', type=str, nargs='+', choices=['within', 'between'], default=['within'],
                        help='which type of correlation matrix to generate')
    parser.add_argument('--feature-idx', type=int, default=3,
                        help="index of the layer in net.features whose output to use. 3 is output of the 2nd"
                             "conv layer")
    parser.add_argument('--no-save', action='store_false', dest='save', help='dont save the matrices generated')
    args = parser.parse_args()

    _, dataloader = get_batch_suite(100, train=False)

    net_1 = Net()
    if args.nets[0] != 'random':
        net_1.load_state_dict(torch.load(args.nets[0]))

    save = {}
    if 'within' in args.mode:
        corr_w = within_net_correlation(dataloader, net_1, args.feature_idx)
        save['within'] = corr_w
    if 'between' in args.mode:
        assert len(args.nets) > 1, 'Must specify two networks for between-net correlation'
        net_2 = Net()
        if args.nets[1] != 'random':
            net_2.load_state_dict(torch.load(args.nets[1]))
        corr_b = between_net_correlation(dataloader, net_1, net_2, args.feature_idx)
        save['between'] = corr_b
    if args.save:
        np.savez('-'.join([net.split('.')[0] for net in args.nets]) + '-correlations.npz', **save)
