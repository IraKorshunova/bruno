import argparse
import importlib
import json
import os
import time

import matplotlib
import numpy as np
import tensorflow as tf

import utils
from config_rnn import defaults

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')

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

target_path = save_dir + '/misc_plots'
utils.autodir(target_path)

# create the model
model = tf.make_template('model', config.build_model)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
model_output = model(x_in)
log_probs, variances = model_output[0], model_output[4]

saver = tf.train.Saver()
n_batches = 100

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    print('Sequence length:', args.seq_len)

    noise_rng = np.random.RandomState(42)
    rng = np.random.RandomState(42)
    batch_idxs = range(0, n_batches)
    data_iter = config.test_data_iter
    data_iter.batch_size = 1

    enropy_all_batches = []
    log_probs_all_batches = []
    for k, (x_batch, y_batch) in enumerate(data_iter.generate_each_digit(noise_rng=rng, rng=rng, same_image=True)):
        lp, vars = sess.run([log_probs, variances], feed_dict={x_in: x_batch})

        es, lps = [], []
        for i in range(config.seq_len):
            e = np.sum(np.log(vars[:, i]), axis=-1)
            es.append(e)
            lps.append(lp[:, i])
        es = np.asarray(es)
        lps = np.asarray(lps)
        enropy_all_batches.append(es)
        log_probs_all_batches.append(lps)

    print(np.asarray(enropy_all_batches).shape)
    enropy_same_img = np.mean(np.asarray(enropy_all_batches), axis=0)
    log_probs_same_img = np.mean(np.asarray(log_probs_all_batches), axis=0)

    print('------------------------------------------------')
    noise_rng = np.random.RandomState(42)
    rng = np.random.RandomState(42)
    batch_idxs = range(0, n_batches)
    data_iter = config.test_data_iter
    data_iter.batch_size = 1
    data_iter.rng = np.random.RandomState(42)

    enropy_all_batches = []
    log_probs_all_batches = []
    for k, (x_batch, y_batch) in enumerate(data_iter.generate_each_digit(noise_rng=rng, rng=rng, same_image=False)):
        lp, vars = sess.run([log_probs, variances], feed_dict={x_in: x_batch})

        es, lps = [], []
        for i in range(config.seq_len):
            e = np.sum(np.log(vars[:, i]), axis=-1)
            es.append(e)
            lps.append(lp[:, i])
        es = np.asarray(es)
        lps = np.asarray(lps)
        enropy_all_batches.append(es)
        log_probs_all_batches.append(lps)

    enropy_mix_img = np.mean(np.asarray(enropy_all_batches), axis=0)
    log_probs_mix_img = np.mean(np.asarray(log_probs_all_batches), axis=0)

    fig = plt.figure(figsize=(4, 3))
    plt.plot(range(config.seq_len), enropy_same_img, 'black', linewidth=1.)
    plt.scatter(range(config.seq_len), enropy_same_img, s=1.5, c='black')
    plt.plot(range(config.seq_len), enropy_mix_img, 'red', linewidth=1.)
    plt.scatter(range(config.seq_len), enropy_mix_img, s=1.5, c='red')
    plt.xlabel('steps')
    plt.ylabel('differential entropy + const')
    plt.savefig(target_path + '/entropy_plot.png',
                bbox_inches='tight', dpi=1000)

    fig = plt.figure(figsize=(4, 3))
    plt.plot(range(config.seq_len), log_probs_same_img, 'black', linewidth=1.)
    plt.scatter(range(config.seq_len), log_probs_same_img, s=1.5, c='black')
    plt.plot(range(config.seq_len), log_probs_mix_img, 'red', linewidth=1.)
    plt.scatter(range(config.seq_len), log_probs_mix_img, s=1.5, c='red')
    plt.xlabel('steps')
    plt.ylabel('log_probs')
    plt.savefig(target_path + '/log_probs_plot.png',
                bbox_inches='tight', dpi=1000)
