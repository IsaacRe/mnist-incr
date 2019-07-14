from __future__ import print_function
from os.path import exists
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from stats_tracking import ActivationTracker
from prune import ActivationPrune


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


running_loss = 0


def join_backward(loss, optimizer, update=False):
    global running_loss
    running_loss += loss
    if update:
        running_loss.backward()
        optimizer.step()
        running_loss = 0


def train(args, model, device, train_loader, optimizer, exp=0):
    model.train()
    for epoch in range(args.num_epoch):
        if args.num_updates is not None:
            update_idxs = list(np.random.choice(len(train_loader), args.num_updates - 1, replace=False)) + [len(train_loader) - 1]
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            if args.num_updates is None:
                loss.backward()
                optimizer.step()
            else:
                assert not args.track_stats, "Incompatible arguments: --track-stats; --num-updates"
                join_backward(loss, optimizer, batch_idx in update_idxs)
            if batch_idx % args.log_interval == 0:
                if args.incremental:
                    print('Learning Exposure: {}, {}/{}th epoch [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        exp, epoch + 1, args.num_epoch, batch_idx * train_loader.batch_size, len(train_loader.sampler.indices),
                               100. * train_loader.batch_size * batch_idx / len(train_loader.sampler.indices), loss.item()))
                else:
                    print('{}/{}th epoch [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, args.num_epoch, batch_idx * train_loader.batch_size, len(train_loader) * train_loader.batch_size,
                        100. * batch_idx / len(train_loader), loss.item()))


accs, loss = [], []


def test(args, model, device, test_loaders, stats_tracker=None, pruner=None):
    global accs, loss
    model.eval()
    test_loss = 0
    correct = 0
    for i, test_loader in enumerate(test_loaders):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    if stats_tracker is not None:
        # currently tracking abs_out, grad, abs_grad, abs_grad_x_out
        all_stats = {}
        for name, stats in stats_tracker.export_stats(*args.stats):
            all_stats[name] = stats
        np.savez('network_stats/activations.npz', **all_stats)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if pruner is not None and pruner.masking:
        num_out, num_prune = 0, 0
        for masked, total in pruner.prune_ratio.values():
            num_out += total
            num_prune += masked

        print('\nNumber of pruned outputs: {}/{}'.format(num_prune, num_out))

    if not args.track_stats:
        accs = np.concatenate([accs, np.array([100. * correct / len(test_loader.dataset)])])
        loss = np.concatenate([loss, np.array([test_loss])])

        if args.num_updates is None:
            np.savez('accs_%d-train_%d-explr_%d-epoch_%d-updates_%d-lexp_%s.npz' %
                     (args.lexp_len, args.num_explr,args.num_epoch, args.num_updates, args.num_lexp, args.id), accs=accs,
                     loss=loss)
        else:
            np.savez('accs_%d-train_%d-explr_%d-epoch_%d-lexp_%s.npz' % (args.lexp_len, args.num_explr, args.num_epoch,
                                                                         args.num_lexp, args.id), accs=accs, loss=loss)

    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--no-save', action='store_false', dest='save_model',
                        help='For Saving the current Model')

    parser.add_argument('--id', type=str, help="Identify multiple runs with same params")
    parser.add_argument('--pt', type=str, default=None, help="Specify the model to pretrain from")

    # stat tracking args
    parser.add_argument('--track-stats', action='store_true', help='Enable stat tracking')
    parser.add_argument('--track-logits', action='store_true', help='Whether to track the final fc layer activations')
    parser.add_argument('--stats', type=str, nargs='+', choices=ActivationTracker.STATS, default=['out'],
                        help='Specify the stat(s) to be tracked')

    # pruning args
    parser.add_argument('--test-prune', action='store_true',
                        help='Whether to test effect of network pruning on accuracy - only implemented for batch learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='Threshold for pruning by the given statistic')
    parser.add_argument('--prune-rate', type=float,
                        help='Number of output neurons to prune relative to total number of output neurons')
    parser.add_argument('--prune-by', type=str, default=['out'], nargs='+', choices=ActivationTracker.STATS+['random'],
                        help='The statistic by which the network outputs will be pruned')
    parser.add_argument('--plot-prune-rate', action='store_true',
                        help='Whether to plot differences in accuracy for different pruning methods as prune rate increases')

    # incremental training args
    parser.add_argument('--incremental', action='store_true', help='Whether to conduct incremental training')
    parser.add_argument('--num-exemplars', type=int, dest='num_explr')
    parser.add_argument('--lexp-len', type=int, default=100) # Full dataset has > 5000 samples per class
    parser.add_argument('--num-epoch', type=int, default=10, metavar='N',
                        help='number of epochs to train during each learning exposure')
    parser.add_argument('--num-lexp', default=10, type=int,
                        help='Number of learning exposure')
    parser.add_argument('--num-updates', type=int, default=None,
                        help='If set, fixes the number of model updates per epoch of each learning exposure')
    parser.add_argument('--new-perm', action='store_true', help='Whether to choose a new random learning exposure perm')
    parser.add_argument('--fix-class-lexp', action='store_true', help='Whether to iterate through all classes before'
                                                                      'beginning a new permutation')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    exemplars = {}

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_set = datasets.MNIST('../data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    test_set = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    def make_sampler(i):
        select = np.where(test_set.train_labels == i)[0]
        indices = np.random.choice(select, len(select), replace=False)
        return torch.utils.data.SubsetRandomSampler(indices.astype('int64'))
    test_loaders = [torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                                                sampler=make_sampler(i), **kwargs) for i in range(10)]

    model = Net().to(device)
    if args.pt is not None:
        model.load_state_dict(torch.load(args.pt))

    stats_tracker = None
    if args.track_stats:
        stats_tracker = ActivationTracker(model, args.track_logits)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.incremental:
        assert args.num_lexp % 10 == 0
        lexp_path = '%d-lexps.npy' % args.num_lexp
        if exists(lexp_path) and not args.new_perm:
            lexps = np.load(lexp_path)
        elif args.fix_class_lexp:
            lexps = np.concatenate([np.random.choice(10, 10, replace=False) for _ in range(args.num_lexp // 10)])
        else:
            lexps = np.random.choice(list(range(10)) * (args.num_lexp // 10), args.num_lexp, replace=False)

        # save lexp order
        if not exists(lexp_path):
            np.save(lexp_path, lexps)

        for itr, i in enumerate(lexps):
            # select data for learning exposure
            indices = np.random.choice(np.where(train_set.train_labels == i)[0], args.lexp_len, replace=False)
            # add exemplars
            explr_indices = np.array([])
            for c in exemplars:
                if c == i:
                    continue
                explr_indices = np.concatenate([explr_indices, exemplars[c],
                                                np.random.choice(exemplars[c], args.lexp_len - args.num_explr)])
            sampler = torch.utils.data.SubsetRandomSampler(np.concatenate([indices, explr_indices]).astype('int64'))
            train_loader = torch.utils.data.DataLoader(train_set,
                                                       batch_size=args.batch_size, shuffle=False, sampler=sampler,
                                                       **kwargs)
            train(args, model, device, train_loader, optimizer, itr)
            test(args, model, device, test_loaders, stats_tracker)

            # save abs_out

            if args.num_explr > 0:
                exemplars[i] = np.random.choice(indices, args.num_explr, replace=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   **kwargs)
        train(args, model, device, train_loader, optimizer)
        test(args, model, device, test_loaders, stats_tracker)
        if args.test_prune:
            prune_rates = np.arange(0.0, 1.0, 0.1) if args.plot_prune_rate else [args.prune_rate]
            accs = {crit: [] for crit in args.prune_by}
            for prune_rate in prune_rates:
                for criteria in args.prune_by:
                    print('Testing prune by %s...' % criteria)
                    pruner = ActivationPrune(model, criteria, alpha=args.alpha, prune_rate=prune_rate)
                    accuracy = test(args, model, device, test_loaders, pruner=pruner)
                    accs[criteria] += [accuracy]
            if args.plot_prune_rate:
                for criteria in args.prune_by:
                    plt.plot(prune_rates, accs[criteria], label=criteria)
                plt.xlabel('Prune Rate')
                plt.ylabel('% Accuracy')
                plt.title('MNIST 1-Epoch Accuracy after Pruning')
                plt.legend()
                plt.savefig('prune_rate_plot.png')

    if (args.save_model):
        if args.num_updates is None:
            torch.save(model.state_dict(), 'mnist_%d-train_%d-explr_%d-epoch_%d-updates_%d-lexp_%s.pt' %
                       (args.lexp_len, args.num_explr, args.num_epoch, args.num_updates, args.num_lexp, args.id))
        else:
            torch.save(model.state_dict(), 'mnist_%d-train_%d-explr_%d-epoch_%d-lexp_%s.pt' %
                       (args.lexp_len, args.num_explr, args.num_epoch, args.num_lexp, args.id))


if __name__ == '__main__':
    main()