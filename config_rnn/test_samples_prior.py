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

my_dpi = 1000

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--n_row', type=int, default=10, help='Number of rows')
args, _ = parser.parse_known_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args
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

# create the model
model = tf.make_template('model', config.build_model, sampling_mode=True)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(config.sample_batch_size,) + config.obs_shape)
samples = model(x_in, sampling_mode=True)[0]

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    generator = config.test_data_iter.generate_each_digit()
    for i, (x_batch, y_batch) in zip(range(3), generator):
        print("Generating prior samples", i)
        feed_dict = {x_in: x_batch}
        prior_samples = []
        for j in range(args.n_row * args.n_row):
            sampled_xx = sess.run(samples, feed_dict)
            sampled_xx = sampled_xx[0, 0]
            prior_samples.append(sampled_xx)

        prior_samples = np.asarray(prior_samples)
        prior_samples = np.reshape(prior_samples, (args.n_row, args.n_row) + prior_samples.shape[1:])

        img_dim = config.obs_shape[1]
        n_channels = config.obs_shape[-1]

        sample_plt = prior_samples
        sample_plt = sample_plt.swapaxes(1, 2)
        sample_plt = sample_plt.reshape((args.n_row * img_dim, args.n_row * img_dim, n_channels))
        sample_plt = sample_plt / 256. if np.max(sample_plt) >= 2. else sample_plt
        sample_plt = np.clip(sample_plt, 0., 1.)

        plt.figure(figsize=(img_dim * args.n_row / my_dpi, img_dim * args.n_row / my_dpi),
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

        img_path = os.path.join(samples_dir, 'prior_sample_%s.png' % i)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='png', dpi=my_dpi)
        plt.close('all')
