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
from matplotlib import gridspec

my_dpi = 1000

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('-s', '--set', type=str, default='test', help='Test or train part')
parser.add_argument('-same', '--same_image', type=int, default=0, help='Same image as inputl')
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
samples, log_probs = model(x_in, sampling_mode=True)

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

    generator = data_iter.generate_each_digit(same_image=args.same_image)
    for i, (x_batch, y_batch) in enumerate(generator):
        print(i, "Generating samples...")
        feed_dict = {x_in: x_batch}
        lps = []
        for j in range(100):
            lp = sess.run(log_probs, feed_dict)
            lps.append(lp)
        lps = np.concatenate(lps, axis=0)
        print(lps.shape)
        print(-1. * np.mean(lps, axis=0))
        img_dim = config.obs_shape[1]
        n_channels = config.obs_shape[-1]
