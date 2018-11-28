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
from config_conditional import defaults

my_dpi = 1000

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--n_row', type=int, default=3, help='Number of rows')
parser.add_argument('--n_col', type=int, default=17, help='Number of cols')
args, _ = parser.parse_known_args()
args.n_context = 0  # don't change this
args.seq_len = 2  # don't change this
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

if not os.path.isdir(save_dir + '/samples'):
    os.makedirs(save_dir + '/samples')
samples_dir = save_dir + '/samples'

print('exp_id', experiment_id)
print('n_context', config.n_context)
print('seq_len', config.seq_len)

# create the model
model = tf.make_template('model', config.build_model, sampling_mode=True)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
y_label = tf.placeholder(tf.float32, shape=(1,) + config.label_shape)
samples = model(x_in, y_label)

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    prior_samples = []
    for j in range(args.n_row):
        for k in range(args.n_col):
            x_batch = np.zeros((1,) + config.obs_shape)
            y_batch = np.zeros((1,) + config.label_shape)
            y_batch[0, 0] = config.train_data_iter.process_angle(k * 20)
            y_batch[0, 1] = config.train_data_iter.process_angle(k * 20)
            feed_dict = {x_in: x_batch, y_label: y_batch}

            angle = int(round(config.train_data_iter.deprocess_angle(y_batch[0, 0, :2])))
            print(angle)
            sampled_x = sess.run(samples, feed_dict)[0, 0]
            prior_samples.append(sampled_x)

    prior_samples = np.asarray(prior_samples)
    prior_samples = np.reshape(prior_samples, (args.n_row, args.n_col) + prior_samples.shape[1:])
    img_dim = config.obs_shape[1]
    n_channels = config.obs_shape[-1]

    sample_plt = prior_samples
    sample_plt = sample_plt.swapaxes(1, 2)
    sample_plt = sample_plt.reshape((args.n_row * img_dim, args.n_col * img_dim, n_channels))
    sample_plt = sample_plt / 256. if np.max(sample_plt) >= 2. else sample_plt
    sample_plt = np.clip(sample_plt, 0., 1.)

    plt.figure(figsize=(10 * img_dim * args.n_col / my_dpi, 10 * img_dim * args.n_row / my_dpi),
               dpi=my_dpi,
               frameon=False)

    img = sample_plt
    if n_channels == 1:
        plt.imshow(img[:, :, 0], cmap='gray', interpolation='None')
    else:
        plt.imshow(img, interpolation='None')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    img_path = os.path.join(samples_dir, 'prior_sample.png')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='png', dpi=my_dpi)
    plt.close('all')
