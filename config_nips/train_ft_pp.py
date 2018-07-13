"""
Implementation of Real-NVP by Laurent Dinh (https://arxiv.org/abs/1605.08803)
Code was started from the PixelCNN++ code (https://github.com/openai/pixel-cnn)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf
import importlib
import logger

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-c', '--config_name', type=str, default='nvp_1', help='Configuration name')
parser.add_argument('-t', '--save_interval', type=int, default=2,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=int, default=0,
                    help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
# optimization
parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------

assert args.nr_gpu == 1

# fix random seed for reproducibility
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

# phase: traing/testing
training_phase = tf.placeholder(tf.bool, name='phase')

# run once for data dependent initialization of parameters
x_init = tf.placeholder(tf.float32, shape=(config.batch_size,) + config.obs_shape)
init_pass = model(x_init, training_phase, init=True)

# get loss gradients over multiple GPUs
xs = []
grads = []
loss_gen = []
all_params = tf.trainable_variables()

n_parameters = 0
for variable in all_params:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    n_parameters += variable_parameters
print('Number of parameters', n_parameters)

for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32,
                             shape=(config.batch_size * config.meta_batch_size / args.nr_gpu,) + config.obs_shape))
    with tf.device('/gpu:%d' % i):
        # train
        with tf.variable_scope('gpu_%d' % i):
            with tf.variable_scope('train'):
                log_probs = model(xs[i], training_phase)
                loss_gen.append(tf.check_numerics(config.loss(log_probs), 'loss has nans'))
                # gradients
                grads.append(tf.gradients(loss_gen[i], all_params))

# add gradients together and get training updates
student_params = ['prior_nu', 'prior_mean', 'prior_variance', 'covariance']
tf_lr = tf.placeholder(tf.float32, shape=[])
tf_student_grad_scale = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        loss_gen[0] += loss_gen[i] / args.nr_gpu
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j] / args.nr_gpu

    for j in range(len(grads[0])):
        if any(name in all_params[j].name for name in student_params):
            grads[0][j] *= tf_student_grad_scale

    # training op
    grads_and_vars = zip(grads[0], all_params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops_gpu0 = []
    for u in update_ops:
        if u.name.startswith('gpu_0/train'):
            update_ops_gpu0.append(u)

    with tf.control_dependencies(update_ops_gpu0):
        if hasattr(config, 'optimizer') and config.optimizer == 'nesterov':
            print('using nesterov momentum')
            train_step = tf.train.MomentumOptimizer(learning_rate=tf_lr, momentum=0.9,
                                                    use_nesterov=True).apply_gradients(
                grads_and_vars=grads_and_vars,
                global_step=None, name='nestrov_momentum')
        elif hasattr(config, 'optimizer') and config.optimizer == 'rmsprop':
            print('using rmsprop')
            train_step = tf.train.RMSPropOptimizer(learning_rate=tf_lr).apply_gradients(
                grads_and_vars=grads_and_vars,
                global_step=None, name='nestrov_momentum')
        else:
            train_step = tf.train.AdamOptimizer(learning_rate=tf_lr, beta1=0.95, beta2=0.9995).apply_gradients(
                grads_and_vars=grads_and_vars,
                global_step=None, name='adam')

# convert loss to bits/dim
train_loss = loss_gen[0]

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
train_losses = []

batch_idxs = range(0, config.max_iter)

print_every = 100
niter = 1
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

        xfs = np.split(x_batch, args.nr_gpu)
        feed_dict = {tf_lr: lr, tf_student_grad_scale: student_grad_scale}
        feed_dict.update({xs[i]: xfs[i] for i in range(args.nr_gpu)})
        feed_dict.update({training_phase: 1})
        l, _ = sess.run([train_loss, train_step], feed_dict)
        train_losses.append(l)
        # print(train_losses[-1])

        current_time = time.clock()
        if iteration % print_every == 0:
            avg_train_loss = np.mean(train_losses[-print_every:])
            print('%d/%d train_loss=%6.8f time/batch=%.2f' % (
                niter, config.max_iter, avg_train_loss, current_time - prev_time))
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
            nu = config.student_layer.nu.eval().flatten()
            print('nu', np.median(nu), np.min(nu), np.max(nu))

        niter += 1
