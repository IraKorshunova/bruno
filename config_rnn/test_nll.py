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

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
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
model = tf.make_template('model', config.build_model)
all_params = tf.trainable_variables()

# phase: trainig/testing
training_phase = tf.placeholder(tf.bool, name='phase')
x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
log_probs = model(x_in, training_phase)

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    n_iter = 10000
    batch_idxs = range(0, n_iter)

    # test
    data_iter = config.test_data_iter
    data_iter.batch_size = 1

    losses = []
    for _, x_batch in zip(batch_idxs, data_iter.generate()):
        l = sess.run(log_probs, feed_dict={x_in: x_batch, training_phase: 0})
        losses.append(l)
    avg_loss = -1. * np.mean(losses)
    bits_per_dim = avg_loss / np.log(2.) / 784
    print('Test Loss', avg_loss)
    print('Bits per dim', bits_per_dim)

    # train
    data_iter = config.train_data_iter
    data_iter.batch_size = 1

    losses = []
    for _, x_batch in zip(batch_idxs, data_iter.generate()):
        l = sess.run(log_probs, feed_dict={x_in: x_batch, training_phase: 0})
        losses.append(l)
    avg_loss = -1. * np.mean(losses)
    bits_per_dim = avg_loss / np.log(2.) / 784
    print('Train Loss', avg_loss)
    print('Bits per dim', bits_per_dim)
