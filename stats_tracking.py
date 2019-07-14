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
    Abstract class that keeps running statistics during training
    Inheriting classes must implement:
        _f_hook(module, input, output): the method to be called during the forward pass, must register backward hook
                                        using _make_b_hook if collecting gradient data during backward pass
        _b_hook(module, grad): the method to be called during the backward pass, if collecting gradient data
        STATS: list containing names of statistics being measured by trackers of that type
    """

    def __init__(self, model, track_logits=False, device='cpu'):
        assert type(self) is not Tracker, "Tried creating object of class Tracker, which is an abstract class"
        self.device = device
        self.model = model
        self.module_names = []
        # the following are referenced by module name
        self.modules = {}
        self.name_from_module = {}
        self._f_hooks = {}
        self._b_hooks = {}
        self.all_module_stats = {}
        self._setup_tracking(track_logits)

    """
    Returns the backward hook implemented by the inheriting class with access to the passed module object.
    Necessary since Tensor.register_hook (used for backward pass hooks) doesn't support access to the current module
    """
    def _make_b_hook(self, module):
        def b_hook(grad):
            return self._b_hook(module, grad)
        return b_hook

    def _setup_tracking(self, track_logits):
        assert hasattr(self.model, 'modules'), "Model does not have modules() method"
        for name, m in self.model.named_modules():
            # setup tracking for each individual layer with network weights
            if hasattr(m, 'weight'):
                # if a final fc layer and track_logits is false then skip
                if type(m) is torch.nn.Linear and m.out_features == 10 and not track_logits:
                    continue
                # create stats object and attach to module
                m.running_stats = Stats(*self.STATS)
                self.module_names += [name]
                self.modules[name] = m
                self.name_from_module[m] = name
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
        return [(name, stats.export(*out_stats)) for name, stats in self.all_module_stats.items()]


class Statistic(torch.Tensor):
    """
    A single statistic. Supports updating and accessing from a single reference
    """

    def __init__(self):
        super(Statistic, self).__init__()
        self.count = 0

    def __call__(self, val):
        self._update(val)

    def _update(self, val):
        assert type(val) is torch.Tensor, "Running stats should be saved as torch.Tensor's"
        if self.count == 0:
            self.data = val
        else:
            self.data = (self.data * self.count + val) / (self.count + 1)
        self.count += 1

    def reset(self):
        self.count = 0


class Stats:
    """
    Running stats class allowing dynamic creation of statistics to track. Allows access of individual stats
        through dereferencing the Stats object ( stats[<stat_name>] ) and updating individual stats by calling
        after dereferencing ( stats[<stats_name>]() )
    """

    def __init__(self, *stats):
        self.stats = {stat: Statistic() for stat in stats}
        for stat in stats:
            self.__dict__[stat] = self.stats[stat]

    def __getitem__(self, item):
        return self.stats[item]

    def __repr__(self):
        return str(self.export())

    def __iter__(self):
        return self.stats.__iter__()

    def values(self):
        return list(self.stats.values())

    def keys(self):
        return list(self.stats)

    def items(self):
        return list(self.stats.items())

    def export(self, *out_stats):
        return [(stat, torch.flatten(self[stat]).data.numpy()) for stat in self.stats if stat in out_stats]


class ActivationTracker(Tracker):
    """
    Keeps track of a layer's activation statistics:
        output: running average of the layers outputs
        grad: running output of the gradient backpropagated to this layer
        output * grad
        abs_grad: |grad|
        abs_grad_x_out: |grad_x_out|
    """
    STATS = ['out', 'grad', 'grad_x_out', 'abs_grad', 'abs_grad_x_out', 'sigma_out']

    def __init__(self, *args, **kwargs):
        super(ActivationTracker, self).__init__(*args, **kwargs)
        self.last_out = {}

    def _f_hook(self, module, input, output):
        # get module name
        m_name = self.name_from_module[module]
        # setup the backward hook by registering it to the output tensor
        self._b_hooks[m_name] = output.register_hook(self._make_b_hook(module))

        output = _to_device(torch.mean(output.data, dim=0), self.device)
        # if not last layer, apply relu
        if type(module) is not torch.nn.Linear or module.out_features > 10:
            output = torch.nn.functional.relu(output)
        self.last_out[m_name] = output

        # update stats
        module.running_stats.out(output)

    def _b_hook(self, module, grad):
        # get module name
        m_name = self.name_from_module[module]
        grad = _to_device(torch.mean(grad, dim=0), self.device)

        # update stats
        module.running_stats.grad(grad)
        module.running_stats.abs_grad(grad.abs())
        grad_x_out = grad * self.last_out[m_name]
        module.running_stats.grad_x_out(grad_x_out)
        module.running_stats.abs_grad_x_out(grad_x_out.abs())

        # set last_output to None now that it's been used
        self.last_out[m_name] = None


class WeightTracker(Tracker):
    """
    Keeps track of layer's weight statistics
    """
    # TODO
    pass
