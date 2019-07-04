import torch


def _to_device(x, device):
    if device == 'cpu':
        return x.cpu()
    elif device == 'cuda':
        return x.cuda()
    assert type(device) is int, "Invalid device specification"
    return x.cuda(device)


class Tracker:
    """
    Abstract wrapper class for an nn.Module that keeps running statistics during training
    Inheriting classes must implement:
        _f_hook(module, input, output): the method to be called during the forward pass, must register backward hook
                                        using _make_b_hook if collecting gradient data during backward pass
        _b_hook(module, grad): the method to be called during the backward pass, if collecting gradient data
        STATS: list containing names of statistics being measured by trackers of that type
    """

    def __init__(self, model, device='cpu'):
        assert type(self) is not Tracker, "Tried creating object of class Tracker, which is an abstract class"
        self.device = device
        self.model = model
        self.module_names = []
        # the following are referenced by module name
        self.modules = {}
        self._f_hooks = {}
        self._b_hooks = {}
        self.all_module_stats = {}
        self._setup_tracking()

    """
    Returns the backward hook implemented by the inheriting class with access to the passed module object.
    Necessary since Tensor.register_hook (used for backward pass hooks) doesn't support access to the current module
    """
    def _make_b_hook(self, module):
        def b_hook(grad):
            return self._b_hook(module, grad)
        return b_hook

    def _setup_tracking(self):
        assert hasattr(self.model, 'modules'), "Model does not have modules() method"
        for name, m in self.model.named_modules():
            # setup tracking for each individual layer with network weights
            if hasattr(m, 'weight'):
                # create stats object and attach to module
                m.running_stats = Stats(*self.STATS)
                self.module_names += [name]
                self.modules[name] = m
                self._f_hooks[name] = m.register_forward_hook(self._f_hook)
                self.all_module_stats[name] = m.running_stats

    def stop_tracking(self):
        for hook in self._f_hooks + self._b_hooks:
            hook.remove()

    def resume_tracking(self):
        pass  # TODO

    def reset_all(self):
        for stats, _ in self.all_module_stats.items():
            stats.reset()

    def export_stats(self, *out_stats):
        return {name: {stat: val.numpy() for stat, val in stats.items() if stat in out_stats}
                for name, stats in self.all_module_stats.item()}


class Stats:
    """
    Running stats class allowing dynamic addition of statistics
    """

    def __init__(self, *stats):
        self.stats = stats
        self.counts = {k: 0 for k in stats}
        for stat in stats:
            self.__dict__[stat] = None
            self.__dict__['update_%s' % stat] = self._make_update_func(stat)

    def __repr__(self):
        return str(self.export())

    def export(self):
        return {stat: self.__dict__[stat] for stat in self.stats}

    """
    Creates a member method to update the passed property
    """
    def _make_update_func(self, stat_name):
        def update_func(x):
            return self._update(x, stat_name)
        return update_func

    """
    Base method defining update logic. Called by dynamically created update methods
    """
    def _update(self, x, stat_name):
        if self.counts[stat_name] == 0:
            self.__dict__[stat_name] = x
        else:
            cnt = self.counts[stat_name]
            self.__dict__[stat_name] = (self.__dict__[stat_name] * cnt + x) / (cnt + 1)
        self.counts[stat_name] += 1

    """
    Reset running statistics
    """
    def reset(self):
        for k in self.counts:
            self.counts[k] = 0


class ActivationTracker(Tracker):
    """
    Keeps track of a layer's activation statistics:
        output: running average of the layers outputs
        grad: running output of the gradient backpropagated to this layer
        output * grad
        abs_out: |output|
        abs_grad: |grad|
        abs_grad_x_out: |grad_x_out|
    """
    STATS = ['out', 'grad', 'grad_x_out', 'abs_out', 'abs_grad', 'abs_grad_x_out', 'sigma_out']

    def _f_hook(self, module, input, output):
        # get module name
        [m_name] = [name for name in self.modules if self.modules[name] is module]
        # setup the backward hook by registering it to the output tensor
        self._b_hooks[m_name] = output.register_hook(self._make_b_hook(module))

        output = _to_device(output.data, self.device)
        module.last_output = output

        # update stats
        module.running_stats.update_out(output)
        module.running_stats.update_abs_out(output.abs())

    def _b_hook(self, module, grad):
        # update stats
        module.running_stats.update_grad(grad)
        module.running_stats.update_abs_grad(grad.abs())
        grad_x_out = grad * module.last_output
        module.running_stats.update_grad_x_out(grad_x_out)
        module.running_stats.update_abs_grad_x_out(grad_x_out.abs())

        # set last_output to None now that it's been used
        module.last_output = None


class WeightTracker(Tracker):
    """
    Keeps track of layer's weight statistics
    """
    # TODO
    pass
