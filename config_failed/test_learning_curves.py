import pickle

import matplotlib
import numpy as np

import utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def process_config_data(config_name, n_points):
    save_dir = utils.find_model_metadata('metadata/', config_name)
    with open(save_dir + '/meta.pkl', 'rb') as f:
        d1 = pickle.load(f)

    l_train_iter1 = np.squeeze(d1['losses_train_iter'])
    print(len(l_train_iter1))
    l_train_iter1 = moving_average(l_train_iter1, n=n_points)
    return l_train_iter1


target_path = 'metadata/'
utils.autodir(target_path)
plt.figure()
n_points = 10

# config_names = ['f_cn2_fashion_gp', 'f_cn2_fashion_tp']
config_names = ['f_cn2_fashion_gp_wn_lr_high', 'f_cn2_fashion_tp_wn_lr_high']
# config_names = ['f_cn2_fashion_gp_wn_corr_high', 'f_cn2_fashion_tp_wn_corr_high']
labels = ['GP', 'TP']
colors = ['black', 'red']
for config, label, color in zip(config_names, labels, colors):
    print(config)
    losses = process_config_data(config, n_points)
    print(np.min(losses), np.max(losses), len(losses))
    x_range = np.arange(0, len(losses))
    plt.plot(x_range, losses, color, linewidth=1.5, label=label)

plt.ylim(1500, 10000)
plt.xlabel('iteration', fontsize=20)
plt.ylabel('NLL', fontsize=20)

plt.legend(loc='upper right', fontsize=18)
plt.savefig(target_path + '/train_loss_%s_%s.png' % (config_names[0], config_names[1]),
            bbox_inches='tight', dpi=1000)
