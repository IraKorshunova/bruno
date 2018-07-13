"""
Implementation of Real-NVP by Laurent Dinh (https://arxiv.org/abs/1605.08803)
Code was started from the PixelCNN++ code (https://github.com/openai/pixel-cnn)
"""

import os
import json
import argparse
import numpy as np
import utils
import importlib
import matplotlib
import pickle
import tensorflow as tf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('-ds', '--data_set', type=str, default='omniglot', help='Dataset name')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# config
configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))

save_dir = utils.find_model_metadata('metadata/', args.config_name)
experiment_id = os.path.dirname(save_dir)
print('exp_id', experiment_id)

with open(save_dir + '/meta.pkl', 'rb') as f:
    d = pickle.load(f)

l_valid = np.squeeze(d['losses_eval_valid'])
l_train = np.squeeze(d['losses_eval_train'])
n_points = len(l_valid)
ndims = len(l_valid[0])
n_iter = d['iteration']
x_range = range(config.validate_every, n_iter + config.validate_every + 1, config.validate_every)

target_path = save_dir + '/learning_curves'
utils.autodir(target_path)

for d in range(ndims):
    print(d)
    l_valid_d = [l_valid[i][d] for i in range(n_points)]
    l_train_d = [l_train[i][d] for i in range(n_points)]

    fig = plt.figure(figsize=(4, 3))
    plt.grid(True, which="both", ls="-", linewidth='0.2')
    plt.plot(x_range, l_valid_d, 'green', linewidth=1.)
    plt.scatter(x_range, l_valid_d, s=1.5, c='green')

    plt.plot(x_range, l_train_d, 'red', linewidth=1.)
    plt.scatter(x_range, l_train_d, s=1.5, c='red')

    plt.xlabel(r'$dim %s$' % d)
    plt.ylabel('NLL')
    plt.savefig(target_path + '/p%s.png' % d,
                bbox_inches='tight', dpi=600)

l_valid_mean = [np.mean(l_valid[i]) for i in range(n_points)]
l_train_mean = [np.mean(l_train[i]) for i in range(n_points)]

fig = plt.figure(figsize=(4, 3))
plt.grid(True, which="both", ls="-", linewidth='0.2')
plt.plot(x_range, l_valid_mean, 'green', linewidth=1.)
plt.scatter(x_range, l_valid_mean, s=1.5, c='green')

plt.plot(x_range, l_train_mean, 'red', linewidth=1.)
plt.scatter(x_range, l_train_mean, s=1.5, c='red')

plt.xlabel(r'$dim %s$' % d)
plt.ylabel('NLL')
plt.savefig(target_path + '/p_mean.png',
            bbox_inches='tight', dpi=600)

fig = plt.figure(figsize=(4, 3))
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

for d in range(ndims):
    print(d)
    l_valid_d = [l_valid[i][d] for i in range(n_points)]
    l_train_d = [l_train[i][d] for i in range(n_points)]

    cc = colors.popitem()[1]
    plt.grid(True, which="both", ls="-", linewidth='0.2')
    plt.plot(x_range, l_valid_d, color=cc, linewidth=1.)
    plt.scatter(x_range, l_valid_d, s=1.5, c=cc)

plt.savefig(target_path + '/p_all.png',
            bbox_inches='tight', dpi=600)

l_valid_d = l_valid[-1]
l_train_d = l_train[-1]

fig = plt.figure(figsize=(4, 3))
plt.grid(True, which="both", ls="-", linewidth='0.2')

plt.plot(range(ndims), l_valid_d, color='green', linewidth=1.)
plt.scatter(range(ndims), l_valid_d, s=1.5, c='green')

plt.plot(range(ndims), l_train_d, color='red', linewidth=1.)
plt.scatter(range(ndims), l_train_d, s=1.5, c='red')

plt.xlabel(r'$dim$')
plt.ylabel('NLL')
plt.savefig(target_path + '/p_dims.png',
            bbox_inches='tight', dpi=600)
