import argparse
import importlib
import json
import os
import time

import numpy as np
import tensorflow as tf

import utils
from config_rnn import defaults

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('-same', '--same_image', type=int, default=0, help='Same image as inputl')

args, _ = parser.parse_known_args()
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
print('exp_id', experiment_id)

# create the model
model = tf.make_template('model', config.build_model)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(1,) + config.obs_shape)
variances = model(x_in)[4]

saver = tf.train.Saver()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    print('Sequence length:', args.seq_len)

    batch_idxs = range(0, 1)

    # test
    data_iter = config.test_data_iter
    data_iter.batch_size = 1

    for _, x_batch in zip(batch_idxs, data_iter.generate_each_digit(noise_rng=rng, same_image=args.same_image)):
        vars = sess.run(variances, feed_dict={x_in: x_batch[0]})
        for v in vars:
            print(np.sum(np.log(v)))
