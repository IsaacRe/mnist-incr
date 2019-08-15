import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from test import get_incr_suite, get_batch_suite


parser = argparse.ArgumentParser()
parser.add_argument('--num-trials', type=int, default=500)
parser.add_argument('--max-lexps', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=20)
args = parser.parse_args()


# In[2]:
number_of_trials = args.num_trials
max_lexps = args.max_lexps  # max number of learning exposures to run each time
lr = 0.01
momentum = 0.5
num_epoch = 1
print_freq = args.print_freq

break_at = []
for trial in range(number_of_trials):
    # currently uses concatenated random permutations of 10
    lexps = np.concatenate([np.random.choice(10, 10, replace=False) for i in range(max_lexps // 10)])

    # In[9]:

    net, x_ent, loaders = get_incr_suite(100)
    #net, x_ent, loader = get_batch_suite(100)
    _, _, test_loader = get_batch_suite(100, train=False)
    optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)


    # In[10]:


    max_time_learned = []
    avg_time_learned = []
    integral_time_learned = []
    accs = []

    time_since_learned = {}

    for lexp, c in enumerate(lexps):
        print('Beginning Learning Exposure for Class {}'.format(c))
        net.train()
        loader = loaders[c]
        for batch_idx, (_, x, y) in enumerate(loader):
            optim.zero_grad()
            out = net(x)

            loss = x_ent(out, y)
            loss.backward()
            optim.step()

            if batch_idx % print_freq == 0:
                print('Learning Exposure: {}/{}, Batch: {}/{}, Loss: {}'.format(lexp + 1, max_lexps,
                                                                                batch_idx, len(loader),
                                                                                loss.item()))
        net.eval()

        # testing + learn/forget tracking
        max_time, total_time, count_correct, count_total = 0, 0, 0, 0
        for ids, x, y in test_loader:
            out = net(x)
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            scores = pred.eq(y.view_as(pred))

            for i, score in zip(ids, scores):
                count_total += 1
                idx = i.item()
                if idx not in time_since_learned:
                    time_since_learned[idx] = -1
                if score == 1:
                    time_since_learned[idx] += 1
                    total_time += time_since_learned[idx]
                    count_correct += 1
                else:
                    time_since_learned[idx] = -1
                if time_since_learned[idx] > max_time:
                    max_time = time_since_learned[idx]
        max_time_learned += [max_time]
        avg_time_learned += [total_time / count_correct if count_correct > 0 else 0.0]
        integral_time_learned += [total_time]

        # break when we see that the model has begun to retain info
        if avg_time_learned[-1] > 0.0:
            break

        acc = count_correct / count_total * 100.
        accs += [acc]

        print('Testing after Learning Exposure {}/{} on class {}... Accuracy: %{}'.format(lexp + 1, max_lexps,
                                                                                          c,
                                                                                          acc))
    break_at += [lexp]
    print('Finished training model {}/{}. Began learning at learning exposure {}'.format(trial + 1, number_of_trials,
                                                                                         lexp + 1))
    plt.hist(break_at)
    plt.title('Num lexps before accuracy increases are maintained')
    plt.xlabel('Learning Exposure')
    plt.savefig('no-explr-time-hist.png')
