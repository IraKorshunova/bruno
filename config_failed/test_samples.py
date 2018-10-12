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
from matplotlib import gridspec

my_dpi = 1000

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
parser.add_argument('--set', type=str, default='test', help='Test or train part')
parser.add_argument('--same_image', type=int, default=0, help='Same image as input')
parser.add_argument('--plot_n', type=int, default=0, help='Plot only last x images. 0 = plot all of them')
args, _ = parser.parse_known_args()
defaults.set_parameters(args)
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

if args.set == 'test':
    data_iter = config.test_data_iter
elif args.set == 'train':
    data_iter = config.train_data_iter
else:
    raise ValueError('wrong set')

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    generator = data_iter.generate_each_digit(same_image=args.same_image, rng=rng)
    for i, (x_batch, y_batch) in enumerate(generator):
        print("Generating samples", i)
        feed_dict = {x_in: x_batch}
        sampled_xx = sess.run(samples, feed_dict)
        img_dim = config.obs_shape[1]
        n_channels = config.obs_shape[-1]
        seq_len = config.seq_len

        if args.plot_n > 0:
            sampled_xx = sampled_xx[:, -args.plot_n:]
            x_batch = x_batch[:, -args.plot_n:]
            seq_len = args.plot_n - 1

        if args.plot_n == 0:
            prior_image = np.zeros((1,) + x_batch[0, 0].shape) + 255.
            x_seq = np.concatenate((prior_image, x_batch[0]))
        else:
            x_seq = x_batch[0]

        x_plt = x_seq.swapaxes(0, 1)
        x_plt = x_plt.reshape((img_dim, (seq_len + 1) * img_dim, n_channels))
        x_plt = x_plt / 256. if np.max(x_plt) >= 2. else x_plt
        x_plt = np.clip(x_plt, 0., 1.)

        sample_plt = sampled_xx.reshape((config.n_samples, (seq_len + 1), img_dim, img_dim, n_channels))
        sample_plt = sample_plt.swapaxes(1, 2)
        sample_plt = sample_plt.reshape((config.n_samples * img_dim, (seq_len + 1) * img_dim, n_channels))
        sample_plt = sample_plt / 256. if np.max(sample_plt) >= 2. else sample_plt
        sample_plt = np.clip(sample_plt, 0., 1.)

        plt.figure(figsize=(img_dim * config.obs_shape[0] / my_dpi, (img_dim + config.n_samples * img_dim) / my_dpi),
                   dpi=my_dpi,
                   frameon=False)
        gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=0.1, hspace=0.1, height_ratios=[1, config.n_samples])

        ax0 = plt.subplot(gs[0])
        img = x_plt
        if n_channels == 1:
            plt.imshow(img[:, :, 0], cmap='gray', interpolation='None')
        else:
            plt.imshow(img, interpolation='None')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        ax1 = plt.subplot(gs[1])
        img = sample_plt
        if n_channels == 1:
            plt.imshow(img[:, :, 0], cmap='gray', interpolation='None')
        else:
            plt.imshow(img, interpolation='None')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        img_path = os.path.join(samples_dir, 'sample_%s_%s_%s' % (args.set, i, args.same_image))
        img_path += '.png' if args.plot_n == 0 else '_%s.png' % args.plot_n
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='png', dpi=my_dpi)
        plt.close('all')
