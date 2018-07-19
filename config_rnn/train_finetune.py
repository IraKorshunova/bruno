import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

import logger

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('-t', '--save_interval', type=int, default=2,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=int, default=0,
                    help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
# -----------------------------------------------------------------------------
assert len(''.join(filter(str.isdigit, os.environ["CUDA_VISIBLE_DEVICES"]))) == 1  # only to use on 1 gpu

np.random.seed(seed=42)
tf.set_random_seed(42)

# config
configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))
experiment_id = '%s-%s' % (args.config_name.split('.')[-1], time.strftime("%Y_%m_%d", time.localtime()))

if not os.path.isdir('metadata'):
    os.makedirs('metadata')
save_dir = 'metadata/' + experiment_id

# logs
if not os.path.isdir('logs'):
    os.makedirs('logs')
sys.stdout = logger.Logger('logs/%s.log' % experiment_id)
sys.stderr = sys.stdout

print('exp_id', experiment_id)

# create the model
model = tf.make_template('model', config.build_model)
x_init = tf.placeholder(tf.float32, shape=(config.batch_size,) + config.obs_shape)
init_pass = model(x_init, init=True)

all_params = tf.trainable_variables()
n_parameters = 0
for variable in all_params:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    n_parameters += variable_parameters
print('Number of parameters', n_parameters)

x_in = tf.placeholder(tf.float32,
                      shape=(config.batch_size * config.meta_batch_size,) + config.obs_shape)
tf_lr = tf.placeholder(tf.float32, shape=[])
tf_student_grad_scale = tf.placeholder(tf.float32, shape=[])

with tf.variable_scope('train'):
    log_probs = model(x_in)[0]
    train_loss = tf.check_numerics(config.loss(log_probs), 'loss has nans')
    grads = tf.gradients(train_loss, all_params)

student_params = ['prior_nu', 'prior_mean', 'prior_var', 'prior_corr']
for j in range(len(grads)):
    if any(name in all_params[j].name for name in student_params):
        grads[j] *= tf_student_grad_scale

grads_and_vars = zip(grads, all_params)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops_gpu0 = []
for u in update_ops:
    if u.name.startswith('gpu_0/train'):
        update_ops_gpu0.append(u)

with tf.control_dependencies(update_ops_gpu0):
    if hasattr(config, 'optimizer') and config.optimizer == 'rmsprop':
        print('using rmsprop')
        train_step = tf.train.RMSPropOptimizer(learning_rate=tf_lr).apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=None, name='rmsprop')
    else:
        train_step = tf.train.AdamOptimizer(learning_rate=tf_lr, beta1=0.95, beta2=0.9995).apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=None, name='adam')

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('starting training')
test_bpd = []
lr = config.learning_rate
student_grad_scale = config.scale_student_grad

train_data_iter = config.train_data_iter
train_iter_losses = []

batch_idxs = range(0, config.max_iter)

print_every = 100
with tf.Session() as sess:
    ckpt_file = config.base_metadata_path
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_file))

    for iteration, (x_batch, _) in zip(batch_idxs, train_data_iter.generate()):
        prev_time = time.clock()

        if hasattr(config, 'learning_rate_schedule') and iteration in config.learning_rate_schedule:
            lr = np.float32(config.learning_rate_schedule[iteration])
            print('setting learning rate to %.7f' % config.learning_rate_schedule[iteration])
        elif hasattr(config, 'lr_decay'):
            lr *= config.lr_decay

        if hasattr(config, 'student_grad_schedule') and iteration in config.student_grad_schedule:
            student_grad_scale = np.float32(config.student_grad_schedule[iteration])
            print('setting student grad scale to %.7f' % config.student_grad_schedule[iteration])

        feed_dict = {tf_lr: lr, tf_student_grad_scale: student_grad_scale, x_in: x_batch}
        l, _ = sess.run([train_loss, train_step], feed_dict)
        train_iter_losses.append(l)

        current_time = time.clock()
        if (iteration + 1) % print_every == 0:
            avg_train_loss = np.mean(train_iter_losses)
            train_iter_losses = []
            print('%d/%d train_loss=%6.8f time/batch=%.2f' % (
                iteration + 1, config.max_iter, avg_train_loss, current_time - prev_time))
            corr = config.student_layer.corr.eval().flatten()

        if (iteration + 1) % config.save_every == 0:
            print('saving model', experiment_id)
            print('learning rate', lr)
            saver.save(sess, save_dir + '/params.ckpt')
            corr = config.student_layer.corr.eval().flatten()
            print('0.01', np.sum(corr > 0.01))
            print('0.1', np.sum(corr > 0.1))
            print('0.2', np.sum(corr > 0.2))
            print('0.3', np.sum(corr > 0.3))
            print('0.5', np.sum(corr > 0.5))
            print('0.7', np.sum(corr > 0.7))
            print('min, max', np.min(corr), np.max(corr))

            if hasattr(config.student_layer, 'nu'):
                nu = config.student_layer.nu.eval().flatten()
                print('nu', np.median(nu), np.min(nu), np.max(nu))
