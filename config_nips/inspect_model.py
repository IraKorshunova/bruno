"""
Implementation of Real-NVP by Laurent Dinh (https://arxiv.org/abs/1605.08803)
Code was started from the PixelCNN++ code (https://github.com/openai/pixel-cnn)
"""

import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
import utils
import importlib
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('-ds', '--data_set', type=str, default='omniglot', help='Dataset name')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(42)
tf.set_random_seed(42)

# config
configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))

save_dir = utils.find_model_metadata('metadata/', args.config_name)
experiment_id = os.path.dirname(save_dir)
print('exp_id', experiment_id)

# create the model
model = tf.make_template('model', config.build_model, sampling_mode=True)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(config.sample_batch_size,) + config.obs_shape)
training_phase = tf.placeholder(tf.bool, name='phase')
samples = model(x_in, training_phase)

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    nu = config.student_layer.nu.eval().flatten()

    # print('mu\n', config.student_layer.mu.eval())
    print('--------------------')
    print('nu\n', nu)
    print('--------------------')
    print('var\n', config.student_layer.var.eval())
    print('--------------------')
    corr = config.student_layer.corr.eval()
    print('corr\n', corr)
    corr = corr.flatten()

    for c, n in zip(corr[np.where(corr > 0.05)[0]], nu[np.where(corr > 0.05)[0]]):
        print(c, n)

    print(np.min(nu), corr[np.where(nu == np.min(nu))[0]])
    n_remained = []
    eps_range = []
    for t in np.arange(0, 0.9, 0.01):
        nr = np.sum(corr > t)
        if not n_remained:
            n_remained.append(nr)
            eps_range.append(t)
        else:
            if nr != n_remained[-1]:
                n_remained.append(nr)
                eps_range.append(t)
        print(t, n_remained[-1])

        # print np.where(corr > 0.2)
        # print corr[np.where(corr > 0.2)[0]]
        # print len(corr[np.where(corr > 0.2)[0]])

    # samples
    target_path = save_dir

    fig = plt.figure(figsize=(4, 3))
    plt.grid(True, which="both", ls="-", linewidth='0.2')
    plt.plot(eps_range, n_remained, 'black', linewidth=1.)
    plt.scatter(eps_range, n_remained, s=1.5, c='black')
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_yscale("log", nonposy='clip')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('number of dimensions')
    plt.savefig(target_path + '/eps_plot.png',
                bbox_inches='tight', dpi=600)
