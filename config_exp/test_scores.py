import argparse
import importlib
import os
import time

import numpy as np
import tensorflow as tf

import utils
from config_rnn import defaults

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr?')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')
args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print(args)
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
latent_log_probs, latent_log_probs_prior = model_output[1], model_output[2]

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    n_iter = 1000
    batch_idxs = range(0, n_iter)

    # test
    data_iter = config.train_data_iter
    data_iter.batch_size = 1

    scores = []
    prior_ll = []
    for _, (x_batch, _) in zip(batch_idxs, data_iter.generate_each_digit(same_image=True)):
        lp, lp_prior = sess.run([latent_log_probs, latent_log_probs_prior], feed_dict={x_in: x_batch})
        scores.append(lp - lp_prior)
        prior_ll.append(lp_prior)
        print(scores[-1])
        print('--------------------------')

    scores = np.stack(scores, axis=0)
    print(scores.shape)
    scores_mean = np.mean(scores, axis=0)
    print(scores_mean)
    print('log likelihood under the prior:')
    prior_ll_mean = np.mean(prior_ll)
    print('LL:', prior_ll_mean)
    print('bits per dim:', -1. * prior_ll_mean / config.ndim / np.log(2.))
