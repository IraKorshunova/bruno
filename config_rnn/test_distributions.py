import argparse
import importlib
import math
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

x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
model_output = model(x_in)
log_probs, latent_log_probs, latent_log_probs_prior = model_output[0], model_output[1], model_output[2]
z_vec = model_output[3]
student_states = model_output[4]
betas = model_output[5]

saver = tf.train.Saver()

data_iter = config.test_data_iter if args.set == 'test' else config.train_data_iter


def student_pdf_1d(X, mu, var, nu):
    num = math.gamma((1. + nu) / 2.) * pow(
        1. + (1. / (nu - 2)) * (1. / var * (X - mu) ** 2), -(1. + nu) / 2.)
    denom = math.gamma(nu / 2.) * pow((nu - 2) * math.pi * var, 0.5)
    return num / denom


def gauss_pdf_1d(X, mu, var):
    return 1. / np.sqrt(2. * np.pi * var) * np.exp(- (X - mu) ** 2 / (2. * var))


with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    corr = config.student_layer.corr.eval().flatten()

    batch_idxs = range(0, 1)

    scores = []
    prior_ll = []
    log_probs = []
    for _, x_batch in zip(batch_idxs, data_iter.generate_diagonal_roll()):
        x_batch = x_batch[:1]
        lp_x, lp_z, lp_prior_z, states, z, bs = sess.run(
            [log_probs, latent_log_probs, latent_log_probs_prior, student_states, z_vec, betas],
            feed_dict={x_in: x_batch})

        x_plt = x_batch[0].swapaxes(0, 1)
        x_plt = x_plt.reshape((28, config.seq_len * 28))

        print(len(bs))
        bs_argmax = None
        b = np.zeros((args.seq_len, config.ndim))
        zs = np.zeros((args.seq_len, config.ndim))
        for i in range(len(bs)):
            print(bs[i])
            print(bs[i].shape)
            b[i, :] = bs[i]
            zs[i, :] = z[0, i]
            bs_argmax = np.argmax(bs[i])
            print('-----------------------')
        print(len(bs))
        print(bs_argmax)

        score = lp_z - lp_prior_z
        sigma = np.zeros((args.seq_len, config.ndim))
        mu = np.zeros((args.seq_len, config.ndim))
        nu = np.zeros((args.seq_len, config.ndim))
        probs = np.zeros((args.seq_len, config.ndim))
        for i in range(len(states)):
            if len(states[i]) == 3:
                m, s, n = states[i]
                nu[i] = n
            else:
                m, s = states[i]
                n = None
                nu[i] = None
            sigma[i] = s
            mu[i] = m
            for j in range(config.ndim):

                if corr[j] > args.eps_corr:
                    if n is None:
                        probs[i, j] = gauss_pdf_1d(z[0, i, j], m[0, j], s[0, j])
                    else:
                        probs[i, j] = student_pdf_1d(z[0, i, j], m[0, j], s[0, j], n[0, j])

        target_path = save_dir
        fig = plt.figure()
        plt.matshow(b.T)
        plt.colorbar()
        plt.xlabel('steps')
        plt.ylabel('dimensions')
        plt.savefig(target_path + '/betas_%s.png' % args.eps_corr,
                    bbox_inches='tight', dpi=1000)

        fig = plt.figure()
        plt.matshow(nu.T)
        plt.colorbar()
        plt.xlabel('steps')
        plt.ylabel('dimensions')
        plt.savefig(target_path + '/nu_%s.png' % args.eps_corr,
                    bbox_inches='tight', dpi=1000)

        fig = plt.figure()
        plt.hist(zs[:, bs_argmax])
        print('max z idx', np.argmax(zs[:, bs_argmax]))
        plt.xlabel('steps')
        plt.ylabel('dimensions')
        plt.savefig(target_path + '/z_hist_%s.png' % args.eps_corr,
                    bbox_inches='tight', dpi=1000)
        my_dpi = 600
        plt.figure(figsize=(28. * config.obs_shape[0] / my_dpi, (28 * 28) / my_dpi),
                   dpi=my_dpi,
                   frameon=False)
        plt.imshow(x_plt, cmap='gray', interpolation='None')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(target_path + '/x_sequence.png',
                    bbox_inches='tight')
        plt.close('all')
