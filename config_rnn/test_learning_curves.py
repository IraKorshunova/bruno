import argparse
import importlib
import json
import os
import pickle

import matplotlib
import numpy as np

import utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
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
l_avg_train = np.squeeze(d['losses_avg_train'])

target_path = save_dir + '/learning_curves'
utils.autodir(target_path)

plt.figure()
x_range = np.arange(len(l_avg_train))
plt.plot(x_range, l_avg_train, 'green', linewidth=1.)
plt.scatter(x_range, l_avg_train, s=1.5, c='green')
plt.xlabel(r'$steps$')
plt.ylabel('NLL')
plt.savefig(target_path + '/train_avg_loss.png',
            bbox_inches='tight', dpi=1000)

plt.figure()
n_points = len(l_valid)
l_valid = l_valid[:, None] if len(l_valid.shape) == 1 else l_valid
l_train = l_train[:, None] if len(l_train.shape) == 1 else l_train
ndims = len(l_valid[0])
n_iter = d['iteration']
x_range = range(config.validate_every, n_iter + config.validate_every, config.validate_every)
plt.plot(x_range, np.mean(l_valid, axis=1), 'green', linewidth=1.)
plt.plot(x_range, np.mean(l_train, axis=1), 'red', linewidth=1.)
plt.xlabel(r'$steps$')
plt.ylabel('NLL')
plt.savefig(target_path + '/train_valid_loss.png',
            bbox_inches='tight', dpi=1000)
