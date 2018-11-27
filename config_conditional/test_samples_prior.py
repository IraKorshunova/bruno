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
from config_norb import defaults

my_dpi = 1000

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--set', type=str, default='test', help='Test or train part')
parser.add_argument('--same_image', type=int, default=0, help='Same image as input')
parser.add_argument('--n_samples', type=int, default=4, help='Number of sequences')
parser.add_argument('--seq_len', type=int, default=2, help='Sequence length')
parser.add_argument('--n_context', type=int, default=0, help='Context length')
parser.add_argument('--rotation', type=int, default=0, help='Rotation to various angles')
parser.add_argument('--random_classes', type=int, default=1, help='Random class order')
parser.add_argument('--n_row', type=int, default=3, help='Number of rows')
parser.add_argument('--n_col', type=int, default=17, help='Number of cols')
parser.add_argument('--sort', type=int, default=0, help='Sort by angle')
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

x_in = tf.placeholder(tf.float32, shape=(config.n_samples,) + config.obs_shape)
y_label = tf.placeholder(tf.float32, shape=(config.n_samples,) + config.label_shape)
samples = model(x_in, y_label)

saver = tf.train.Saver()

if args.set == 'test':
    data_iter = config.test_data_iter
elif args.set == 'train':
    data_iter = config.train_data_iter
elif args.set == 'train_chunk':
    data_iter = config.train_data_iter
else:
    raise ValueError('wrong set')

data_iter.batch_size = config.n_samples
if args.rotation:
    generator = data_iter.generate_each_digit_rotation(n_context=args.n_context, random_classes=args.random_classes)
else:
    generator = data_iter.generate_each_digit(random_classes=args.random_classes,
                                              rng=np.random.RandomState(317070))

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    prior_samples = []
    for j in range(args.n_row):
        for k in range(args.n_col):
            x_batch = np.zeros((config.n_samples,) + config.obs_shape)[:config.n_samples]
            y_batch = np.zeros((config.n_samples,) + config.label_shape)[:config.n_samples]
            y_batch[0, 0] = data_iter.process_angle(k * 20)
            y_batch[0, 1] = data_iter.process_angle(k * 20)
            feed_dict = {x_in: x_batch, y_label: y_batch}

            angle = int(round(data_iter.deprocess_angle(y_batch[0, 0, :2])))
            print(angle)
            sampled_x = sess.run(samples, feed_dict)[0, 0]
            prior_samples.append(sampled_x)

    prior_samples = np.asarray(prior_samples)
    print(prior_samples.shape)
    prior_samples = np.reshape(prior_samples, (args.n_row, args.n_col) + prior_samples.shape[1:])
    print(prior_samples.shape)
    img_dim = config.obs_shape[1]
    n_channels = config.obs_shape[-1]

    sample_plt = prior_samples
    sample_plt = sample_plt.swapaxes(1, 2)
    sample_plt = sample_plt.reshape((args.n_row * img_dim, args.n_col * img_dim, n_channels))
    sample_plt = sample_plt / 256. if np.max(sample_plt) >= 2. else sample_plt
    sample_plt = np.clip(sample_plt, 0., 1.)

    plt.figure(figsize=(10* img_dim * args.n_col / my_dpi, 10* img_dim * args.n_row / my_dpi),
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
