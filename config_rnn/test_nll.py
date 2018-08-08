import argparse
import importlib
import json
import os
import time

import numpy as np
import tensorflow as tf

import utils
from config_rnn import defaults

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--n_batches', type=int, default=10000, help='number of batches')
args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
# -----------------------------------------------------------------------------
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

x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
log_probs = model(x_in)[0]

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    print('Sequence length:', args.seq_len)

    batch_idxs = range(0, args.n_batches)

    # test
    data_iter = config.test_data_iter
    data_iter.batch_size = 1

    probs = []
    probs_prior = []
    for _, x_batch in zip(batch_idxs, data_iter.generate(noise_rng=rng)):
        l = sess.run(log_probs, feed_dict={x_in: x_batch})
        probs.append(l)
        probs_prior.append(l[:, 0])
    avg_loss = -1. * np.mean(probs)
    bits_per_dim = avg_loss / np.log(2.) / config.ndim
    print('Test Loss %.3f' % avg_loss)
    print('Bits per dim %.3f' % bits_per_dim)
    print('Test loss under prior %.3f' % -np.mean(probs_prior))

    # train
    data_iter = config.train_data_iter
    data_iter.batch_size = 1

    probs = []
    probs_prior = []
    for _, x_batch in zip(batch_idxs, data_iter.generate(noise_rng=rng)):
        l = sess.run(log_probs, feed_dict={x_in: x_batch})
        probs.append(l)
        probs_prior.append(l[:, 0])
    avg_loss = -1. * np.mean(probs)
    bits_per_dim = avg_loss / np.log(2.) / config.ndim
    print('Train Loss %.3f' % avg_loss)
    print('Bits per dim %.3f' % bits_per_dim)
    print('Train loss under prior %.3f' % -np.mean(probs_prior))
