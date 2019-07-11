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

    def __init__(self, model, *prune_args, **prune_kwargs):
        assert type(self) is not DynamicPrune, "Tried to instantiate object of type DynamicPrune, which is abstract"
        self.device = _get_device(model)
        self.model = model
        self.module_names = []
        self.modules = {}
        self.name_from_module = {}
        self._f_hooks = {}
        self.masks = {}
        self.prune_ratio = {}
        self._setup_pruning(prune_args, prune_kwargs)
        # bool whether currently masking pruned outputs
        self.masking = True

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
            self.masks[name] = self.prune(m, *prune_args, **prune_kwargs)

    def stop_masking(self):
        for hook in self._f_hooks.values():
            hook.remove()
        self.masking = False

    def resume_masking(self):
        for name, m in self.modules.items():
            self._f_hooks[name] = m.register_forward_hook(self._f_hook)
        self.masking = True


class ActivationPrune(DynamicPrune):
    """
    Class to prune activations of output neurons in a neural network
    """

    def _f_hook(self, module, input, output):
        m_name = self.name_from_module[module]
        mask = self.masks[m_name]
        output[(slice(None), *mask)] = 0.0
        self.prune_ratio[m_name] = int(mask[0].shape[0]), int(output[0].numel())
        return None

    def prune(self, module, *args, **kwargs):
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
