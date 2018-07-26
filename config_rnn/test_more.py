import argparse
import importlib
import json
import os
import time

import matplotlib
import numpy as np
import tensorflow as tf

import utils
from config_rnn import defaults

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nn_extra_gauss, nn_extra_student

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_name', type=str, required=True, help='Configuration name')
parser.add_argument('--set', type=str, default='test', help='train or test set?')
parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
parser.add_argument('--mask_dims', type=int, default=0, help='keep the dimensions with correlation > eps_corr?')
parser.add_argument('--eps_corr', type=float, default=0., help='minimum correlation')
parser.add_argument('--same_class', type=int, default=1, help='sequences from the same class')
parser.add_argument('--same_image', type=int, default=0, help='sequences from the same image')
parser.add_argument('--n_batches', type=int, default=1000, help='how many batches to average over')
args, _ = parser.parse_known_args()
defaults.set_parameters(args)
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
if args.mask_dims == 0:
    assert args.eps_corr == 0.
# -----------------------------------------------------------------------------
rng = np.random.RandomState(42)
tf.set_random_seed(42)

# config
configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))

save_dir = utils.find_model_metadata('metadata/', args.config_name)
experiment_id = os.path.dirname(save_dir)
print('exp_id', experiment_id)

target_path = save_dir + 'tests/'
utils.autodir(target_path)

# create the model
model = tf.make_template('model', config.build_model)
all_params = tf.trainable_variables()

x_in = tf.placeholder(tf.float32, shape=(args.seq_len,) + config.obs_shape)
model_output = model(x_in)
log_probs, latent_log_probs, latent_log_probs_prior = model_output[0], model_output[1], model_output[2]
z_vec = model_output[3]

saver = tf.train.Saver()

data_iter = config.test_data_iter if args.set == 'test' else config.train_data_iter

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    corr = config.student_layer.corr.eval()
    var = config.student_layer.var.eval()
    nu = config.student_layer.nu.eval()
    print(corr)
    print(nu)
    print(var)
    print('--------------------')

z_var = tf.placeholder(tf.float32, shape=(args.seq_len, args.seq_len, config.ndim))
l_rnn = nn_extra_student.StudentRecurrentLayer(shape=(config.ndim,), nu_init=nu, var_init=var, corr_init=corr)
l_rnn2 = nn_extra_gauss.GaussianRecurrentLayer(shape=(config.ndim,), var_init=var, corr_init=corr)
probs = []
probs_gauss = []
with tf.variable_scope("one_step") as scope:
    l_rnn.reset()
    l_rnn2.reset()
    for i in range(config.seq_len):
        prob_i = l_rnn.get_log_likelihood(z_var[:, i, :])
        probs.append(prob_i)
        l_rnn.update_distribution(z_var[:, i, :])

        prob_i = l_rnn2.get_log_likelihood(z_var[:, i, :])
        probs_gauss.append(prob_i)
        l_rnn2.update_distribution(z_var[:, i, :])
        scope.reuse_variables()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    begin = time.time()
    ckpt_file = save_dir + 'params.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    batch_idxs = range(0, args.n_batches)

    scores = []
    prior_ll = []
    probs_x = []
    for idx, x_batch in zip(batch_idxs, data_iter.generate_diagonal_roll(same_class=args.same_class, noise_rng=rng,
                                                                         same_image=args.same_image)):
        lp_x, lp_z, lp_prior_z, z = sess.run([log_probs, latent_log_probs, latent_log_probs_prior, z_vec],
                                             feed_dict={x_in: x_batch})
        probs_x = lp_z
        print(z.shape)

        print(z)

        # for i in range(probs_x.shape[0]):
        #     fig = plt.figure(figsize=(4, 3))
        #     plt.grid(True, which="both", ls="-", linewidth='0.2')
        #     plt.plot(range(len(probs_x[i])), probs_x[i], 'black', linewidth=1.)
        #     plt.scatter(range(len(probs_x[i])), probs_x[i], s=1.5, c='black')
        #     plt.xlabel('step')
        #     plt.ylabel(r'nll ($\epsilon$=%s)' % args.eps_corr)
        #     plt.savefig(
        #         target_path + '/aaa_%s.png' % i,
        #         bbox_inches='tight', dpi=600)
        #     plt.close()

        probs_x = np.diag(probs_x)
        fig = plt.figure(figsize=(4, 3))
        plt.grid(True, which="both", ls="-", linewidth='0.2')
        plt.plot(range(len(probs_x)), probs_x, 'black', linewidth=1.)
        plt.scatter(range(len(probs_x)), probs_x, s=1.5, c='black')
        plt.xlabel('step')
        plt.ylabel(r'nll ($\epsilon$=%s)' % args.eps_corr)
        plt.savefig(
            target_path + '/aaa_diag.png',
            bbox_inches='tight', dpi=600)
        plt.close()

with tf.Session() as sess:
    sess.run(init)
    assert np.allclose(corr, l_rnn.corr.eval())
    assert np.allclose(var, l_rnn.var.eval())
    assert np.allclose(nu, l_rnn.nu.eval())

    print(z)
    feed_dict = {z_var: z}
    probs_out = sess.run(probs, feed_dict)
    probs_out_gauss = sess.run(probs_gauss, feed_dict)
    probs_out = np.asarray(probs_out)
    probs_out_gauss = np.asarray(probs_out_gauss)
    print(probs_out.shape)
    print(probs_out_gauss.shape)

    for i in range(probs_out.shape[0]):
        fig = plt.figure(figsize=(4, 3))
        plt.grid(True, which="both", ls="-", linewidth='0.2')
        plt.plot(range(len(probs_out_gauss[i])), probs_out_gauss[i], 'red', linewidth=0.7)
        plt.scatter(range(len(probs_out_gauss[i])), probs_out_gauss[i], s=1.5, c='red')
        plt.plot(range(len(probs_out[i])), probs_out[i], 'black', linewidth=1.)
        plt.scatter(range(len(probs_out[i])), probs_out[i], s=1.5, c='black')
        plt.xlabel('step')
        plt.ylabel(r'nll ($\epsilon$=%s)' % args.eps_corr)
        plt.savefig(
            target_path + '/gauss_st_%s.png' % i,
            bbox_inches='tight', dpi=600)
        plt.close()

    probs_out = np.diag(probs_out)
    probs_out_gauss = np.diag(probs_out_gauss)

    fig = plt.figure(figsize=(4, 3))
    plt.grid(True, which="both", ls="-", linewidth='0.2')
    plt.plot(range(len(probs_out_gauss)), probs_out_gauss, 'red', linewidth=0.7)
    plt.scatter(range(len(probs_out_gauss)), probs_out_gauss, s=1.5, c='red')
    plt.plot(range(len(probs_out)), probs_out, 'black', linewidth=1.0)
    plt.scatter(range(len(probs_out)), probs_out, s=1.5, c='black')
    plt.xlabel('step')
    plt.ylabel(r'nll ($\epsilon$=%s)' % args.eps_corr)
    plt.savefig(
        target_path + '/gauss_st_diag.png',
        bbox_inches='tight', dpi=600)
    plt.close()
