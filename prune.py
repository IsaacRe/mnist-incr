import numpy as np
import torch.nn as nn
import torch
from stats_tracking import _to_device


def _get_device(module):
    param = next(module.parameters())
    if param.is_cuda:
        return param.get_device()
    else:
        return 'cpu'


class DynamicPrune:
    """
    Abstract class to prune activations of a network during forward pass
    Inheriting classes must implement:
        _f_hook(module, input, output): the method that will me registered with each module undergoing pruning and
                                        called during that module's forward pass to alter its output
        _prune(*args, **kwargs): method returning the mask to be used in computation of the pruned network output during _f_hook
    """

    def __init__(self, model, *prune_args, prune_method='by_value', **prune_kwargs):
        assert type(self) is not DynamicPrune, "Tried to instantiate object of type DynamicPrune, which is abstract"
        if not hasattr(self, '_prune_%s' % prune_method):
            raise NotImplementedError('Passed pruning method has not been implemented: %s' % prune_method)
        self.prune_method = prune_method
        self.device = _get_device(model)
        self.model = model
        self.module_names = []
        self.modules = {}
        self.name_from_module = {}
        self._f_hooks = {}
        self.masks = {}
        self._setup_pruning(prune_args, prune_kwargs)

    def _setup_pruning(self, prune_args, prune_kwargs):
        assert hasattr(self.model, 'modules'), "Model does not have modules() method"
        for name, m in self.model.named_modules():
            if not hasattr(m, 'weight'):
                continue
            # if a final fc layer then skip
            if type(m) is torch.nn.Linear and m.out_features == 10:
                continue
            # create stats object and attach to module
            self.module_names += [name]
            self.modules[name] = m
            self.name_from_module[m] = name
            self._f_hooks[name] = m.register_forward_hook(self._f_hook)
            self.masks[m] = self._prune(m, *prune_args, **prune_kwargs)

    def stop_masking(self):
        for hook in self._f_hooks.values():
            hook.remove()

    def resume_masking(self):
        pass  # TODO


class ActivationPrune(DynamicPrune):
    """
    Class to prune activations of output neurons in a neural network
    """

    def _f_hook(self, module, input, output):
        m_name = self.name_from_module[module]
        mask = self.masks[m_name]
        output[mask] = 0.0
        return output

    def _prune(self, module, *args, **kwargs):
        return self._prune_by_value(module, *args, **kwargs)

    """
    All pruning methods must return a mask that is the same shape of the passed module's output
    """

    def _prune_by_value(self, module, stat_name, alpha=0.1, prune_rate=None, prune_highest=False):
        assert hasattr(module, 'running_stats'), "Passed module has no running stats attached"
        assert hasattr(module.running_stats, stat_name), "Module's running stats has no entry for stat: %s" % stat_name
        values = module.running_stats[stat_name]
        if prune_rate is None:
            return np.where(values > alpha) if prune_highest else np.where(values < alpha)
        else:
            raise NotImplementedError("Prune_rate not currently implemented")
