import argparse
import importlib
import matplotlib
import numpy as np
import tensorflow as tf
import multivariate_student
import utils
from config_rnn import defaults
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='name of the configuration')
parser.add_argument('--seq_len', type=int, default=2, help='sequence length')

args = parser.parse_args()
defaults.set_parameters(args)
print(args)
gp_model = True if 'gp' in args.config_name else False
# -----------------------------------------------------------------------------

np.random.seed(seed=42)

configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))

# metadata
save_dir = utils.find_model_metadata('metadata/', args.config_name)
experiment_id = os.path.dirname(save_dir)
print('exp_id', experiment_id)

# samples
target_path = save_dir + '/hists'
utils.autodir(target_path)

# create the model
model = tf.make_template('model', config.build_model, sampling_mode=False)
all_params = tf.trainable_variables()

training_phase = tf.placeholder(tf.bool, name='phase')
batch_size = 100
x_in = tf.placeholder(tf.float32, shape=(batch_size,) + config.obs_shape)
z_codes = model(x_in, training_phase, return_z=True)

saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    nu = config.student_layer.nu.eval().flatten()
    mu = config.student_layer.mu.eval().flatten()
    var = config.student_layer.var.eval().flatten()

    batch_idxs = range(0, 1)

    # test
    data_iter = config.test_data_iter
    data_iter.batch_size = batch_size

    all_codes = None
    for _, x_batch in zip(batch_idxs, data_iter.generate()):
        print('here!')
        codes = sess.run(z_codes, feed_dict={x_in: x_batch, training_phase: 0})
        print(codes.shape)
        all_codes = codes if all_codes is None else np.concatenate((codes, all_codes), axis=0)
        all_codes = all_codes[:, 0, :]  # take only fist element of each sequence
        print(all_codes.shape)

    for i in xrange(all_codes.shape[-1]):
        plt.figure()
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        ax = plt.gca()
        ax.margins(x=0)

        if gp_model:
            x_lim = (np.min(all_codes[:, i]), np.max(all_codes[:, i]))
            x_lim = (min(mu[i] - 5 * np.sqrt(var[i]), x_lim[0]), max(mu[i] + 5 * np.sqrt(var[i]), x_lim[0]))
        else:
            x_lim = (mu[i] - 5 * np.sqrt(var[i]), mu[i] + 5 * np.sqrt(var[i]))

        x_range = np.linspace(x_lim[0], x_lim[1], 1000)
        y = multivariate_student.multivariate_student.pdf_1d(x_range, mu[i], var[i], nu[i])
        plt.plot(x_range, y, 'black', label='theor', linewidth=2.5)

        if gp_model:
            plt.hist(all_codes[:, i], bins=100, normed=True, alpha=0.5, label='actual')
        else:
            plt.hist(all_codes[:, i], bins=100, normed=True,
                     alpha=0.5, range=(x_lim[0], x_lim[1]), label='actual')

        print(i, np.min(all_codes[:, i]), np.max(all_codes[:, i]), np.argmin(all_codes[:, i]), np.argmax(
            all_codes[:, i]))

        plt.legend(loc='upper right', fontsize=18)
        plt.xlabel('z', fontsize=20)
        plt.ylabel('p(z)', fontsize=20)
        plt.savefig(target_path + '/hist_latent_%s.png' % i, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.clf()
