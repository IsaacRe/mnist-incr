from os.path import exists
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from test import add_return_index
add_return_index(datasets.MNIST)
from net import Net


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
        for batch_idx, (idx, data, target) in enumerate(train_loader):
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


def test(args, model, device, test_loaders):
    global accs, loss
    model.eval()
    test_loss = 0
    correct = 0
    for i, test_loader in enumerate(test_loaders):
        for idx, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accs = np.concatenate([accs, np.array([100. * correct / len(test_loader.dataset)])])
    loss = np.concatenate([loss, np.array([test_loss])])

    if args.save_acc:
        if args.num_updates is None:
            np.savez('accs_%d-train_%d-explr_%d-epoch_%d-updates_%d-lexp_%s.npz' %
                     (args.lexp_len, args.num_explr,args.num_epoch, args.num_updates, args.num_lexp, args.id), accs=accs,
                     loss=loss)
        else:
            np.savez('accs_%d-train_%d-explr_%d-epoch_%d-lexp_%s.npz' % (args.lexp_len, args.num_explr, args.num_epoch,
                                                                         args.num_lexp, args.id), accs=accs, loss=loss)


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
    parser.add_argument('--save-acc', action='store_true', help='whether to record model performance')

    parser.add_argument('--id', type=str, help="Identify multiple runs with same params")
    parser.add_argument('--pt', type=str, default=None, help="Specify the model to pretrain from")

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
            test(args, model, device, test_loaders)

            # save abs_out

            if args.num_explr > 0:
                exemplars[i] = np.random.choice(indices, args.num_explr, replace=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   **kwargs)
        train(args, model, device, train_loader, optimizer)
        test(args, model, device, test_loaders)

    if (args.save_model):
        if args.num_updates is None:
            torch.save(model.state_dict(), 'mnist_%d-train_%d-explr_%d-epoch_%d-updates_%d-lexp_%s.pt' %
                       (args.lexp_len, args.num_explr, args.num_epoch, args.num_updates, args.num_lexp, args.id))
        else:
            torch.save(model.state_dict(), 'mnist_%d-train_%d-explr_%d-epoch_%d-lexp_%s.pt' %
                       (args.lexp_len, args.num_explr, args.num_epoch, args.num_lexp, args.id))


if __name__ == '__main__':
    main()