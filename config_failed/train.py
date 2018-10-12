import argparse
import importlib
import json
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import tensorflow as tf

import logger
import utils

matplotlib.use('Agg')

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
parser.add_argument('--resume', type=int, default=0, help='Resume training from a checkpoint?')
parser.add_argument('--tf_seed', type=int, default=0, help='tf rng seed')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
assert args.nr_gpu == len(''.join(filter(str.isdigit, os.environ["CUDA_VISIBLE_DEVICES"])))
# -----------------------------------------------------------------------------
np.random.seed(seed=42)
tf.reset_default_graph()
tf.set_random_seed(args.tf_seed)

# config
configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))
if not args.resume:
    experiment_id = '%s-%s' % (args.config_name.split('.')[-1], time.strftime("%Y_%m_%d", time.localtime()))
    utils.autodir('metadata')
    save_dir = 'metadata/' + experiment_id
    utils.autodir(save_dir)
else:
    save_dir = utils.find_model_metadata('metadata/', args.config_name)
    experiment_id = os.path.dirname(save_dir).split('/')[-1]
    with open(save_dir + '/meta.pkl', 'rb') as f:
        resumed_metadata = pickle.load(f)
        last_lr = resumed_metadata['lr']
        last_iteration = resumed_metadata['iteration']
        print('Last iteration', last_iteration)
        print('Last learning rate', last_lr)

# logs
utils.autodir('logs')
sys.stdout = logger.Logger('logs/%s.log' % experiment_id)
sys.stderr = sys.stdout

print('exp_id', experiment_id)
if args.resume:
    print('Resuming training')

# create the model
model = tf.make_template('model', config.build_model)

# run once for data dependent initialization of parameters
x_init = tf.placeholder(tf.float32, shape=(config.batch_size,) + config.obs_shape)
init_pass = model(x_init, init=True)[0]

all_params = tf.trainable_variables()
n_parameters = 0
for variable in all_params:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    n_parameters += variable_parameters
print('Number of parameters', n_parameters)

# get loss gradients over multiple GPUs
xs = []
grads = []
train_losses = []

# evaluation in case we want to validate
x_in_eval = tf.placeholder(tf.float32, shape=(config.batch_size,) + config.obs_shape)
log_probs = model(x_in_eval)[0]
eval_loss = config.eval_loss(log_probs) if hasattr(config, 'eval_loss') else config.loss(log_probs)

for i in range(args.nr_gpu):
    xs.append(tf.placeholder(tf.float32, shape=(config.batch_size / args.nr_gpu,) + config.obs_shape))
    with tf.device('/gpu:%d' % i):
        # train
        with tf.variable_scope('gpu_%d' % i):
            with tf.variable_scope('train'):
                log_probs = model(xs[i])[0]
                train_losses.append(config.loss(log_probs))
                grads.append(tf.gradients(train_losses[i], all_params))

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
tf_student_grad_scale = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        train_losses[0] += train_losses[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]

    # average over gpus
    train_losses[0] /= args.nr_gpu
    for j in range(len(grads[0])):
        grads[0][j] /= args.nr_gpu

    # scale gradients of student_params
    student_params = ['prior_nu', 'prior_mean', 'prior_var', 'prior_corr']
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
        if hasattr(config, 'optimizer') and config.optimizer == 'rmsprop':
            print('using rmsprop')
            train_step = tf.train.RMSPropOptimizer(learning_rate=tf_lr).apply_gradients(
                grads_and_vars=grads_and_vars,
                global_step=None, name='rmsprop')
        else:
            print('using adam')
            train_step = tf.train.AdamOptimizer(learning_rate=tf_lr, beta1=0.95, beta2=0.9995).apply_gradients(
                grads_and_vars=grads_and_vars,
                global_step=None, name='adam')

train_loss = train_losses[0]

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

print('\n Start training')
train_data_iter = config.train_data_iter
lr = config.learning_rate
student_grad_scale = config.scale_student_grad
batch_idxs = range(0, config.max_iter)
print_every = 100
train_iter_losses = []
if args.resume:
    losses_eval_valid = resumed_metadata['losses_eval_valid']
    losses_eval_train = resumed_metadata['losses_eval_train']
    losses_avg_train = resumed_metadata['losses_avg_train']
else:
    losses_eval_valid, losses_eval_train, losses_avg_train = [], [], []

start_time = time.time()
with tf.Session() as sess:
    if args.resume:
        ckpt_file = save_dir + 'params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    prev_time = time.time()
    for iteration, x_batch in zip(batch_idxs, train_data_iter.generate()):

        if (iteration + 1) % 100 == 0:
            x_batch[:, -1] = x_batch[:, -1] * 0. + 255. + np.random.uniform(size=x_batch[:, -2].shape)

        if hasattr(config, 'learning_rate_schedule') and iteration in config.learning_rate_schedule:
            lr = np.float32(config.learning_rate_schedule[iteration])
        elif hasattr(config, 'lr_decay'):
            lr *= config.lr_decay

        if hasattr(config, 'student_grad_schedule') and iteration in config.student_grad_schedule:
            student_grad_scale = np.float32(config.student_grad_schedule[iteration])
            print('setting student grad scale to %.7f' % config.student_grad_schedule[iteration])

        if args.resume and iteration < last_iteration:
            if iteration % (print_every * 10) == 0:
                print(iteration, 'skipping training')
            continue

        # init
        if iteration == 0:
            print('initializing the model...')
            sess.run(initializer)
            init_loss = sess.run(init_pass, {x_init: x_batch})
            sess.graph.finalize()
        else:
            xfs = np.split(x_batch, args.nr_gpu)
            feed_dict = {tf_lr: lr, tf_student_grad_scale: student_grad_scale}
            feed_dict.update({xs[i]: xfs[i] for i in range(args.nr_gpu)})
            l, _ = sess.run([train_loss, train_step], feed_dict)
            print(iteration, l)
            if (iteration + 1) % 100 == 0:
                print('anomaly batch')
                print(l)
            train_iter_losses.append(l)
            if np.isnan(l):
                print('Loss is NaN')
                sys.exit(0)

            if (iteration + 1) % print_every == 0:
                avg_train_loss = np.mean(train_iter_losses[-print_every:])
                losses_avg_train.append(avg_train_loss)
                print('%d/%d train_loss=%6.8f bits/value=%.3f' % (
                    iteration + 1, config.max_iter, avg_train_loss, avg_train_loss / config.ndim / np.log(2.)))
                corr = config.student_layer.corr.eval().flatten()

            if hasattr(config, 'validate_every') and (iteration + 1) % config.validate_every == 0:
                print('\n Validating ...')
                losses = []
                rng = np.random.RandomState(42)
                for _, x_valid_batch in zip(range(0, config.n_valid_batches),
                                            config.valid_data_iter.generate(rng)):
                    feed_dict = {x_in_eval: x_valid_batch}
                    l = sess.run([eval_loss], feed_dict)
                    losses.append(l)
                avg_loss = np.mean(np.asarray(losses), axis=0)
                losses_eval_valid.append(avg_loss)
                print('valid loss', avg_loss)

                losses = []
                rng = np.random.RandomState(42)
                for _, x_valid_batch in zip(range(0, config.n_valid_batches),
                                            train_data_iter.generate(rng)):
                    feed_dict = {x_in_eval: x_valid_batch}
                    l = sess.run([eval_loss], feed_dict)
                    losses.append(l)
                avg_loss = np.mean(np.asarray(losses), axis=0)
                losses_eval_train.append(avg_loss)
                print('train loss', avg_loss)

            if (iteration + 1) % config.save_every == 0:
                current_time = time.time()
                eta_time = (config.max_iter - iteration) / config.save_every * (current_time - prev_time)
                prev_time = current_time
                print('ETA: ', time.strftime("%H:%M:%S", time.gmtime(eta_time)))
                print('Saving model (iteration %s):' % iteration, experiment_id)
                print('current learning rate:', lr)
                saver.save(sess, save_dir + '/params.ckpt')

                with open(save_dir + '/meta.pkl', 'wb') as f:
                    pickle.dump({'lr': lr,
                                 'losses_train_iter': train_iter_losses,
                                 'iteration': iteration + 1,
                                 'losses_avg_train': losses_avg_train,
                                 'losses_eval_valid': losses_eval_valid,
                                 'losses_eval_train': losses_eval_train}, f)

                corr = config.student_layer.corr.eval().flatten()
                print('0.01', np.sum(corr > 0.01))
                print('0.1', np.sum(corr > 0.1))
                print('0.2', np.sum(corr > 0.2))
                print('0.3', np.sum(corr > 0.3))
                print('0.5', np.sum(corr > 0.5))
                print('0.7', np.sum(corr > 0.7))
                print('corr min-max:', np.min(corr), np.max(corr))
                var = config.student_layer.var.eval().flatten()
                print('var min-max:', np.min(var), np.max(var))

                if hasattr(config.student_layer, 'nu'):
                    nu = config.student_layer.nu.eval().flatten()
                    print('nu median-min-max:', np.median(nu), np.min(nu), np.max(nu))

print('Total time: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
