import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf
import utils
import importlib
from config_rnn import defaults

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr?')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

args = parser.parse_args()
defaults.set_parameters(args)
print(args)

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

# phase: training/testing
training_phase = tf.placeholder(tf.bool, name='phase')
x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
log_probs, prior_log_probs = model(x_in, training_phase, return_prior=True)

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
    for _, x_batch in zip(batch_idxs, data_iter.generate()):
        l, lp = sess.run([log_probs, prior_log_probs], feed_dict={x_in: x_batch, training_phase: 0})
        scores.append(l - lp)
        prior_ll.append(lp)
        print(lp, l)
        print('--------------------------')

    scores = np.stack(scores, axis=0)
    print(scores.shape)
    scores_mean = np.mean(scores, axis=0)
    print(scores_mean)
    prior_ll_mean = np.mean(prior_ll)
    print(prior_ll_mean)
    print(-1. * prior_ll_mean / config.ndim / np.log(2.))
