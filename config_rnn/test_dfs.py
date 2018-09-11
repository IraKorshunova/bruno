import argparse
import importlib
import json
import os
import time

import matplotlib
import numpy as np
import tensorflow as tf

import utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, required=True, help='Configuration name')
args, unknown = parser.parse_known_args()
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
model = tf.make_template('model', config.build_model, sampling_mode=True)
x_in = tf.placeholder(tf.float32, shape=(config.sample_batch_size,) + config.obs_shape)
model_output = model(x_in)

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    var = config.student_layer.var.eval().flatten()
    corr = config.student_layer.corr.eval().flatten()

    if hasattr(config.student_layer, 'nu'):
        nu = config.student_layer.nu.eval().flatten()
    else:
        nu = np.zeros_like(var)

    print('nu\n', nu)
    print('--------------------')
    print('var\n', var)
    print('--------------------')
    print('corr\n', corr)

    print('******* corr - nu ********')
    idxs = np.where(corr > 0.05)[0]
    for i, c, n, v in zip(idxs, corr[idxs], nu[idxs], var[idxs]):
        print(i, c, n, v)
    print('--------------------------')

    plt.figure(figsize=(4, 3))
    plt.hist(nu, bins=100, normed=True)
    plt.savefig(save_dir + '/nu_hist.png',
                bbox_inches='tight', dpi=600)

    plt.figure(figsize=(4, 3))
    plt.scatter(nu, corr, s=1.5, c='black')
    plt.xlabel('nu')
    plt.ylabel('corr')
    plt.savefig(save_dir + '/nu_corr.png',
                bbox_inches='tight', dpi=600)
