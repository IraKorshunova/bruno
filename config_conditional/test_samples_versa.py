"""
Generates samples from conditional BRUNO and plots them together with VERSA samples that have
to be generated beforehand.
"""
import argparse
import importlib
import json
import os

import matplotlib
import numpy as np
import tensorflow as tf

import utils

matplotlib.use('Agg')
from config_conditional import defaults

import matplotlib.pyplot as plt
from matplotlib import gridspec

from PIL import Image

my_dpi = 1000

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--n_samples', type=int, default=3, help='Number of sequences')
parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
parser.add_argument('--n_context', type=int, default=1, help='Context length')
parser.add_argument('--sort', type=int, default=0, help='Sort by angle')
parser.add_argument('--versa_model_path', type=str,
                    default='/mnt/storage/users/ikorshun/versa/src/checkpoint/2019-06-19-17-59-01/')
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
print('n_context', config.n_context)
print('seq_len', config.seq_len)

# create the model
model = tf.make_template('model', config.build_model, sampling_mode=True)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
y_label = tf.placeholder(tf.float32, shape=(1,) + config.label_shape)
samples = model(x_in, y_label)

saver = tf.train.Saver()

data_iter = config.test_data_iter
data_iter.batch_size = 1
generator = data_iter.generate_each_digit(rng=np.random.RandomState(317070))

with tf.Session() as sess:
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    for i, (x_batch, y_batch) in enumerate(generator):
        versa_samples_dir = args.versa_model_path + 'images_{0:02d}'.format(i)
        versa_img_paths = os.listdir(versa_samples_dir)
        versa_img_paths = [p for p in versa_img_paths if 'generated' in p]
        angle2versa_path = {}
        for p in versa_img_paths:
            angle = int(p.replace('.png', '').split('_')[-1])
            angle2versa_path[angle] = versa_samples_dir + '/' + p

        print(i, "Generating samples...")
        feed_dict = {x_in: x_batch, y_label: y_batch}
        sampled_xx = []
        for _ in range(args.n_samples):
            sampled_xx.append(sess.run(samples, feed_dict))
        sampled_xx = np.concatenate(sampled_xx, axis=0)
        img_dim = config.obs_shape[1]
        n_channels = config.obs_shape[-1]

        angles = []
        for j in range(config.seq_len):
            angle = int(round(data_iter.deprocess_angle(y_batch[0, j, :2]), -1))
            if j < config.n_context:
                angle = -1
            angles.append(angle)
        print(angles)

        if args.sort:
            angles_argsort = np.argsort(angles)
            angles = np.asarray(angles)[angles_argsort]
            sampled_xx = sampled_xx[:, angles_argsort]
            x_batch = x_batch[:, angles_argsort]
            y_batch = y_batch[:, angles_argsort]

        versa_x = []
        print('n angles', len(angles))
        for a in angles:
            print('angle', a)
            if a > -1:
                img = Image.open(angle2versa_path[int(a)])
                img = np.array(img.getdata())
                versa_x.append(img)
        versa_x = np.float32(np.asarray(versa_x))

        fs = 2
        x_seq = x_batch[0]
        x_plt = x_seq.swapaxes(0, 1)
        x_plt = x_plt.reshape((img_dim, config.seq_len * img_dim, n_channels))
        x_plt = np.clip(x_plt, 0., 255.)
        if np.max(x_plt) > 1.:
            x_plt /= 255.
        print('x', np.mean(x_plt), np.std(x_plt))

        x_context = x_plt[:, :config.n_context * img_dim]
        x_plt_framed = np.zeros((x_context.shape[0] + 2 * fs, x_context.shape[1] + 2 * fs, 3))
        x_plt_framed[fs:-fs, fs:-fs, :] += x_context
        x_plt_framed[:fs, :, 0] = 1
        x_plt_framed[-fs:, :, 0] = 1
        x_plt_framed[:, :fs, 0] = 1
        x_plt_framed[:, -fs:, 0] = 1
        x_context = np.copy(x_plt_framed)

        x_plt = x_plt[:, config.n_context * img_dim:]

        sample_plt = sampled_xx.reshape((args.n_samples, (config.seq_len), img_dim, img_dim, n_channels))
        sample_plt = sample_plt.swapaxes(1, 2)
        sample_plt = sample_plt.reshape((args.n_samples * img_dim, config.seq_len * img_dim, n_channels))
        sample_plt = sample_plt[:, config.n_context * img_dim:]
        sample_plt = np.clip(sample_plt, 0., 255.)
        if np.max(sample_plt) > 1.:
            sample_plt /= 255.
        print('samples', np.mean(sample_plt), np.std(sample_plt))

        n_versa_samples = versa_x.shape[0]
        versa_plt = versa_x.reshape((1, n_versa_samples, img_dim, img_dim, 1))
        versa_plt = versa_plt.swapaxes(1, 2)
        versa_plt = versa_plt.reshape((img_dim, n_versa_samples * img_dim, n_channels))
        versa_plt = np.clip(versa_plt, 0., 255.)
        if np.max(versa_plt) > 1.:
            versa_plt /= 255.
        print('versa', np.mean(versa_plt), np.std(versa_plt))

        plt.figure(figsize=(img_dim * config.obs_shape[0] / my_dpi * 5,
                            (img_dim + (args.n_samples + 1) * img_dim) / my_dpi * 5),
                   dpi=my_dpi,
                   frameon=False)
        gs = gridspec.GridSpec(nrows=3, ncols=2, wspace=0.025, hspace=0.025, height_ratios=[1, args.n_samples, 1],
                               width_ratios=[config.n_context, config.seq_len - config.n_context])

        ax0 = plt.subplot(gs[0])
        img = x_context
        plt.imshow(img, interpolation='None', vmin=0., vmax=1.)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        ax1 = plt.subplot(gs[1])
        img = x_plt
        plt.imshow(img[:, :, 0], cmap='gray', interpolation='None', vmin=0., vmax=1.)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        ax3 = plt.subplot(gs[3])
        img = sample_plt
        plt.imshow(img[:, :, 0], cmap='gray', interpolation='None', vmin=0., vmax=1.)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.text(-17, 120, "A \n\nB \n\nC \n\nD", size=4)

        ax4 = plt.subplot(gs[5])
        img = versa_plt
        plt.imshow(img[:, :, 0], cmap='gray', interpolation='None', vmin=0., vmax=1.)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        img_path = os.path.join(samples_dir,
                                'sample_%s_%s.png' % (args.set, i))
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='png', dpi=1000)

        plt.close('all')
