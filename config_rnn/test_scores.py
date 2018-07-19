import argparse
import importlib
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
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('--set', type=str, default='test', help='train or test set?')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr?')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')
args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print(args)
if args.mask_dims == 0:
    assert args.eps_corr == 0.
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

x_in = tf.placeholder(tf.float32, shape=(config.seq_len,) + config.obs_shape)
model_output = model(x_in)
latent_log_probs, latent_log_probs_prior = model_output[1], model_output[2]

saver = tf.train.Saver()

data_iter = config.test_data_iter if args.set == 'test' else config.train_data_iter

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    n_iter = 1000
    batch_idxs = range(0, n_iter)

    scores = []
    prior_ll = []
    for _, x_batch in zip(batch_idxs, data_iter.generate_diagonal_roll()):
        lp, lp_prior = sess.run([latent_log_probs, latent_log_probs_prior], feed_dict={x_in: x_batch})
        score = np.diag(lp - lp_prior)
        scores.append(score)
        prior_ll.append(lp_prior[0])

    scores = np.stack(scores, axis=0)
    print(scores.shape)
    scores_mean = np.mean(scores, axis=0)
    print(scores_mean)
    print('log likelihood under the prior:')
    prior_ll_mean = np.mean(prior_ll)
    print('LL:', prior_ll_mean)
    print('bits per dim:', -1. * prior_ll_mean / config.ndim / np.log(2.))

target_path = save_dir
fig = plt.figure(figsize=(4, 3))
plt.grid(True, which="both", ls="-", linewidth='0.2')
plt.plot(range(len(scores_mean)), scores_mean, 'black', linewidth=1.)
plt.scatter(range(len(scores_mean)), scores_mean, s=1.5, c='black')
plt.xlabel('step')
plt.ylabel(r'score ($\epsilon$=%s)' % args.eps_corr)
plt.savefig(target_path + '/scores_plot_eps%s_len%s_%s.png' % (args.eps_corr, args.seq_len, args.set),
            bbox_inches='tight', dpi=600)
