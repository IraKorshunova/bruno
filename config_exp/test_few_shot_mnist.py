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


def classify(config_name, seq_len, n_trials, digit_classes, condition_on_train):
    configs_dir = __file__.split('/')[-2]
    config = importlib.import_module('%s.%s' % (configs_dir, config_name))

    # metadata
    save_dir = utils.find_model_metadata('metadata/', args.config_name)
    expid = os.path.dirname(save_dir).split('/')[-1]

    assert seq_len == config.seq_len

    utils.autodir('logs')
    sys.stdout = logger.Logger(
        'logs/%s_test_class_%s_%s%s.log' % (expid, n_trials, config.seq_len,
                                            '_ontrain' if condition_on_train else ''))
    sys.stderr = sys.stdout

    print('Building the model', expid)
    model = tf.make_template('model', config.build_model)

    data_iter = config.test_data_iter2
    data_iter.digits = digit_classes if digit_classes is not None else data_iter.digits
    print(data_iter.batch_size)
    print(data_iter.digits)

    x_in = tf.placeholder(tf.float32, shape=(data_iter.batch_size,) + config.obs_shape)
    log_probs = model(x_in)[0]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt_file = save_dir + 'params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        trial_accuracies = []

        for trial in range(n_trials):

            generator = data_iter.generate(n_random_samples=1, trial=trial,
                                           condition_on_train=condition_on_train)

            cm = np.zeros((10, 10))

            x_number2scores = defaultdict(list)
            x_number2y = {}
            for iteration, (x_batch, y_batch, x_number) in enumerate(generator):
                y_true = int(y_batch[0, -1])

                log_p = sess.run(log_probs, feed_dict={x_in: x_batch})

                log_p = log_p.reshape((data_iter.batch_size, config.seq_len))[:, -1]

                x_number2scores[x_number].append(log_p)
                x_number2y[x_number] = y_true
                if (1. * iteration + 1) % 1000 == 0 or n_trials == 1:
                    print(x_number + 1)

            # average scores
            for k, v in x_number2scores.items():
                y_true = x_number2y[k]
                avg_score = np.mean(np.asarray(v), axis=0)
                class_idx = data_iter.digits[np.argmax(avg_score)]
                cm[y_true, class_idx] += 1
            print(cm)
            acc = np.trace(cm) / np.sum(cm)
            print(trial, 'accuracy', acc)
            print('n test examples:', np.sum(cm))
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
parser.add_argument('--n_trials', type=int, default=100, help='number of trials')
parser.add_argument('--digit_classes', type=int, default=[1, 3, 5, 7, 9], nargs='+', help='digits to test')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')
parser.add_argument('--condition_on_train', type=int, default=1,
                    help='use images from the train subset (not used for training anyway)')
args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print(args)
# -----------------------------------------------------------------------------

classify(config_name=args.config_name,
         seq_len=args.seq_len,
         n_trials=args.n_trials,
         digit_classes=args.digit_classes,
         condition_on_train=bool(args.condition_on_train))
