import argparse
import importlib
import json
import os

import matplotlib
import numpy as np
import tensorflow as tf

import utils
from config_rnn import defaults

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_anomaly(config_name, n_sequences):
    configs_dir = __file__.split('/')[-2]
    config = importlib.import_module('%s.%s' % (configs_dir, config_name))

    # metadata
    save_dir = utils.find_model_metadata('metadata/', args.config_name)
    expid = os.path.dirname(save_dir).split('/')[-1]

    # samples
    target_path = save_dir + "/anomaly/"
    utils.autodir(target_path)

    print('Building the model', expid)
    model = tf.make_template('model', config.build_model)

    data_iter = config.test_data_iter
    data_iter.batch_size = 1

    x_in = tf.placeholder(tf.float32, shape=(data_iter.batch_size,) + config.obs_shape)
    model_output = model(x_in)
    latent_log_probs, latent_log_probs_prior = model_output[1], model_output[2]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt_file = save_dir + 'params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        for iteration, (x_batch, y_batch) in zip(range(n_sequences), data_iter.generate_anomaly()):

            assert x_batch.shape[0] == data_iter.batch_size
            assert data_iter.batch_size == 1

            lp, lpp = sess.run([latent_log_probs, latent_log_probs_prior], feed_dict={x_in: x_batch})
            lp = np.squeeze(lp)
            lpp = np.squeeze(lpp)
            scores = []
            for i in range(y_batch.shape[-1]):
                print(i, lp[i], lpp[i], lp[i] - lpp[i], y_batch[0, i])
                scores.append(lp[i] - lpp[i])
            print('-------------')

            quartile_1, quartile_3 = np.percentile(scores, [25, 75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)
            anomaly_idxs = np.where(np.asarray(scores) < lower_bound)[0]
            # don't count the first image as an outlier
            anomaly_idxs = np.delete(anomaly_idxs, np.argwhere(anomaly_idxs == 0))
            print(anomaly_idxs)

            x_batch = np.squeeze(x_batch)
            x_batch = x_batch[anomaly_idxs]

            if len(anomaly_idxs) != 0:
                plt.figure(figsize=(4, 1.5))

                gs = gridspec.GridSpec(nrows=len(anomaly_idxs), ncols=6, hspace=0.1, wspace=0.1)

                ax0 = plt.subplot(gs[:, :-1])
                for i in anomaly_idxs:
                    plt.plot((i + 1, i + 1), (min(scores) - 2, max(scores) + 2), 'r--', linewidth=0.5, dashes=(1, 0.5))
                plt.plot(range(1, args.seq_len + 1), scores, 'black', linewidth=1.)
                plt.gca().axes.set_ylim([min(scores) - 2, max(scores) + 2])
                plt.gca().axes.set_xlim([-0.2, args.seq_len + 0.2])
                plt.scatter(range(1, args.seq_len + 1), scores, c='black', s=1.5)
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                plt.tick_params(axis='both', which='major', labelsize=6)
                plt.xlabel('n', fontsize=8, labelpad=0)
                plt.ylabel('score', fontsize=8, labelpad=0)

                for i in range(len(anomaly_idxs)):
                    ax1 = plt.subplot(gs[i, -1])
                    plt.imshow(x_batch[i], cmap='gray', interpolation='None')
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')

                plt.savefig(target_path + '/anomaly_%s_%s.png' % (iteration, args.mask_dims),
                            bbox_inches='tight', dpi=600, pad_inches=0)


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='name of the configuration')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--n_sequences', type=int, default=1, help='number of sequences')
parser.add_argument('--eval_only_last', type=int, default=0, help='evaluate only p(last|all prev) for speed')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')

args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
if args.mask_dims == 0:
    assert args.eps_corr == 0.
# -----------------------------------------------------------------------------

plot_anomaly(args.config_name, args.n_sequences)
