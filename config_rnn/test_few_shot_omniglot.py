import argparse
import importlib
import os
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf

import logger
import utils
from config_rnn import defaults

np.set_printoptions(suppress=True)


def classify(config_name, seq_len, n_trials, batch_size):
    configs_dir = __file__.split('/')[-2]
    config = importlib.import_module('%s.%s' % (configs_dir, config_name))

    # metadata
    save_dir = utils.find_model_metadata('metadata/', args.config_name)
    expid = os.path.dirname(save_dir).split('/')[-1]

    assert seq_len == config.seq_len

    utils.autodir('logs')
    sys.stdout = logger.Logger(
        'logs/%s_test_class_%s_%s_%s.log' % (expid, n_trials, config.seq_len, batch_size))
    sys.stderr = sys.stdout

    print('Building the model', expid)
    model = tf.make_template('model', config.build_model)

    data_iter = config.test_data_iter2
    data_iter.batch_size = batch_size

    x_in = tf.placeholder(tf.float32, shape=(data_iter.batch_size,) + config.obs_shape)
    log_probs = model(x_in)[0]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt_file = save_dir + 'params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        trial_accuracies = []

        for trial in range(n_trials):

            generator = data_iter.generate(trial=trial)

            n_correct = 0
            n_total = 0

            x_number2scores = defaultdict(list)
            x_number2true_y = {}
            x_number2ys = {}
            for iteration, (x_batch, y_batch, x_number) in enumerate(generator):
                y_true = int(y_batch[0, -1])

                log_p = sess.run(log_probs, feed_dict={x_in: x_batch})
                if np.isnan(np.min(log_p)):
                    print(log_p, 'nans!')
                    sys.exit(0)

                log_p = log_p.reshape((data_iter.batch_size, config.seq_len))[:, -1]

                x_number2scores[x_number].append(log_p)
                x_number2true_y[x_number] = y_true
                x_number2ys[x_number] = y_batch[:, 0]
                if (1. * iteration + 1) % 1000 == 0 or n_trials == 1:
                    print(x_number + 1)

            # average scores
            for k, v in x_number2scores.items():
                y_true = x_number2true_y[k]
                avg_score = np.mean(np.asarray(v), axis=0)
                max_idx = np.argmax(avg_score)
                if x_number2ys[k][max_idx] == y_true:
                    n_correct += 1
                n_total += 1

            acc = n_correct / n_total
            print(trial, 'accuracy', acc)
            print('n test examples', n_total)
            trial_accuracies.append(acc)
            print(trial_accuracies)

        print('---------------------------------------------')
        print(n_trials, config.seq_len)
        print(trial_accuracies)
        print('average accuracy over trials', np.mean(trial_accuracies))
        print('std accuracy over trials', np.std(trial_accuracies))


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='name of the configuration')
parser.add_argument('--seq_len', type=int, default=2, help='sequence length = number of shots + 1')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size = K-way')
parser.add_argument('--n_trials', type=int, default=20, help='number of trials')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')

args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print(args)
# -----------------------------------------------------------------------------

classify(config_name=args.config_name,
         seq_len=args.seq_len,
         n_trials=args.n_trials,
         batch_size=args.batch_size)
