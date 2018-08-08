import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

import data_iter
import nn_extra_nvp
import nn_extra_student
from config_rnn import defaults

batch_size = 16
sample_batch_size = 1
n_samples = 4
rng = np.random.RandomState(42)
test_rng = np.random.RandomState(317070)
seq_len = defaults.seq_len
eps_corr = defaults.eps_corr
mask_dims = defaults.mask_dims

nonlinearity = tf.nn.elu
weight_norm = True

train_data_iter = data_iter.BaseExchSeqDataIterator(seq_len=seq_len, batch_size=batch_size,
                                                    dataset='cifar10', set='train', rng=rng)
test_data_iter = data_iter.BaseExchSeqDataIterator(seq_len=seq_len, batch_size=batch_size,
                                                   dataset='cifar10', set='test', rng=test_rng)
obs_shape = train_data_iter.get_observation_size()  # (seq_len, 28,28,1)
print('obs shape', obs_shape)

ndim = np.prod(obs_shape[1:])
corr_init = np.ones((ndim,), dtype='float32') * 0.1
nu_init = 1000

optimizer = 'rmsprop'
learning_rate = 0.001
lr_decay = 0.999995
scale_student_grad = 0.
max_iter = 70000
save_every = 1000
student_grad_schedule = {0: 0., 100: 0.1}

nvp_layers = []
nvp_dense_layers = []
student_layer = None


def build_model(x, init=False, sampling_mode=False):
    global nvp_layers
    global nvp_dense_layers
    with arg_scope([nn_extra_nvp.conv2d_wn, nn_extra_nvp.dense_wn], init=init):
        if len(nvp_layers) == 0:
            build_nvp_model()

        if len(nvp_dense_layers) == 0:
            build_nvp_dense_model()

        global student_layer
        if student_layer is None:
            student_layer = nn_extra_student.StudentRecurrentLayer(shape=(ndim,), corr_init=corr_init, nu_init=nu_init)

        x_shape = nn_extra_nvp.int_shape(x)
        x_bs = tf.reshape(x, (x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4]))
        x_bs_shape = nn_extra_nvp.int_shape(x_bs)

        log_det_jac = tf.zeros(x_bs_shape[0])

        y, log_det_jac = nn_extra_nvp.dequantization_forward_and_jacobian(x_bs, log_det_jac)
        y, log_det_jac = nn_extra_nvp.logit_forward_and_jacobian(y, log_det_jac)

        # construct forward pass
        z = None
        for layer in nvp_layers:
            y, log_det_jac, z = layer.forward_and_jacobian(y, log_det_jac, z)

        z = tf.concat([z, y], 3)
        for layer in nvp_dense_layers:
            z, log_det_jac, _ = layer.forward_and_jacobian(z, log_det_jac, None)

        z_shape = nn_extra_nvp.int_shape(z)
        z_vec = tf.reshape(z, (x_shape[0], x_shape[1], -1))
        log_det_jac = tf.reshape(log_det_jac, (x_shape[0], x_shape[1]))

        log_probs = []
        z_samples = []
        latent_log_probs = []
        latent_log_probs_prior = []

        if mask_dims:
            mask_dim = tf.greater(student_layer.corr, tf.ones_like(student_layer.corr) * eps_corr)
            mask_dim = tf.cast(mask_dim, tf.float32)
        else:
            mask_dim = None

        with tf.variable_scope("one_step") as scope:
            student_layer.reset()
            for i in range(seq_len):
                if sampling_mode:
                    z_sample = student_layer.sample(nr_samples=n_samples)
                    z_samples.append(z_sample)
                else:
                    latent_log_prob = student_layer.get_log_likelihood(z_vec[:, i, :], mask_dim=mask_dim)
                    latent_log_probs.append(latent_log_prob)

                    log_prob = latent_log_prob + log_det_jac[:, i]
                    log_probs.append(log_prob)

                    latent_log_prob_prior = student_layer.get_log_likelihood_under_prior(z_vec[:, i, :],
                                                                                         mask_dim=mask_dim)
                    latent_log_probs_prior.append(latent_log_prob_prior)

                student_layer.update_distribution(z_vec[:, i, :])
                scope.reuse_variables()

        if sampling_mode:
            # one more sample after seeing the last element in the sequence
            z_sample = student_layer.sample(nr_samples=n_samples)
            z_samples.append(z_sample)

            z_samples = tf.concat(z_samples, 1)
            z_samples_shape = nn_extra_nvp.int_shape(z_samples)
            z_samples = tf.reshape(z_samples,
                                   (z_samples_shape[0] * z_samples_shape[1],
                                    z_shape[1], z_shape[2], z_shape[3]))  # (n_samples*seq_len, z_img_shape)

            for layer in reversed(nvp_dense_layers):
                z_samples, _ = layer.backward(z_samples, None)

            x_samples = None
            for layer in reversed(nvp_layers):
                x_samples, z_samples = layer.backward(x_samples, z_samples)

            # inverse logit
            x_samples = 1. / (1 + tf.exp(-x_samples))
            x_samples = tf.reshape(x_samples,
                                   (z_samples_shape[0], z_samples_shape[1], x_shape[2], x_shape[3], x_shape[4]))
            return x_samples

        log_probs = tf.stack(log_probs, axis=1)
        latent_log_probs = tf.stack(latent_log_probs, axis=1)
        latent_log_probs_prior = tf.stack(latent_log_probs_prior, axis=1)

        return log_probs, latent_log_probs, latent_log_probs_prior, z_vec


def build_nvp_model():
    global nvp_layers
    num_scales = 2
    for scale in range(num_scales - 1):
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('checkerboard0', name='Checkerboard%d_1' % scale,
                                           nonlinearity=nonlinearity, weight_norm=weight_norm))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('checkerboard1', name='Checkerboard%d_2' % scale,
                                           nonlinearity=nonlinearity, weight_norm=weight_norm))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('checkerboard0', name='Checkerboard%d_3' % scale,
                                           nonlinearity=nonlinearity, weight_norm=weight_norm))
        nvp_layers.append(nn_extra_nvp.SqueezingLayer(name='Squeeze%d' % scale))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('channel0', name='Channel%d_1' % scale, nonlinearity=nonlinearity,
                                           weight_norm=weight_norm))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('channel1', name='Channel%d_2' % scale, nonlinearity=nonlinearity,
                                           weight_norm=weight_norm))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('channel0', name='Channel%d_3' % scale, nonlinearity=nonlinearity,
                                           weight_norm=weight_norm))
        nvp_layers.append(nn_extra_nvp.FactorOutLayer(scale, name='FactorOut%d' % scale))

    # final layer
    scale = num_scales - 1
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard0', name='Checkerboard%d_1' % scale,
                                       nonlinearity=nonlinearity, weight_norm=weight_norm))
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard1', name='Checkerboard%d_2' % scale,
                                       nonlinearity=nonlinearity, weight_norm=weight_norm))
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard0', name='Checkerboard%d_3' % scale,
                                       nonlinearity=nonlinearity, weight_norm=weight_norm))
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard1', name='Checkerboard%d_4' % scale,
                                       nonlinearity=nonlinearity, weight_norm=weight_norm))
    nvp_layers.append(nn_extra_nvp.FactorOutLayer(scale, name='FactorOut%d' % scale))


def build_nvp_dense_model():
    global nvp_dense_layers

    for i in range(6):
        mask = 'even' if i % 2 == 0 else 'odd'
        name = '%s_%s' % (mask, i)
        nvp_dense_layers.append(
            nn_extra_nvp.CouplingLayerDense(mask, name=name, nonlinearity=nonlinearity, n_units=512,
                                            weight_norm=weight_norm))


def loss(log_probs):
    return -tf.reduce_mean(log_probs)
