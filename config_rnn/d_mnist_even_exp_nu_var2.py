import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import data_iter
import nn_extra_nvp
import nn_extra_student
from config_rnn import defaults

batch_size = 64
sample_batch_size = 1
n_samples = 4
rng = np.random.RandomState(42)
rng_test = np.random.RandomState(317070)
seq_len = defaults.seq_len
eps_corr = defaults.eps_corr
mask_dims = defaults.mask_dims

nonlinearity = tf.nn.elu
weight_norm = True

train_data_iter = data_iter.BaseExchSeqDataIterator(seq_len=seq_len, batch_size=batch_size,
                                                    set='train', rng=rng, digits=[0, 2, 4, 6, 8])
test_data_iter = data_iter.BaseExchSeqDataIterator(seq_len=seq_len, batch_size=batch_size, set='test',
                                                   digits=[1, 3, 5, 7, 9], rng=rng_test)

valid_data_iter = data_iter.BaseExchSeqDataIterator(seq_len=seq_len, batch_size=batch_size,
                                                    set='test', rng=rng, digits=[0, 2, 4, 6, 8])

test_data_iter2 = data_iter.BaseTestBatchSeqDataIterator(seq_len=seq_len,
                                                         set='test',
                                                         rng=rng,
                                                         digits=[1, 3, 5, 7, 9])

obs_shape = train_data_iter.get_observation_size()  # (seq_len, 28,28,1)
print('obs shape', obs_shape)

ndim = np.prod(obs_shape[1:])
corr_init = np.ones((ndim,), dtype='float32') * 0.1
nu_init = 1000

optimizer = 'rmsprop'
learning_rate = 0.001
lr_decay = 0.999995
max_iter = 200000
save_every = 1000

validate_every = 1000
n_valid_batches = 20

scale_student_grad = 0.
student_grad_schedule = {0: 0., 100: 0.1}

nvp_layers = []
nvp_dense_layers = []
student_layer = None


def build_model(x, init=False, sampling_mode=False):
    global nvp_dense_layers
    with arg_scope([nn_extra_nvp.conv2d_wn, nn_extra_nvp.dense_wn], init=init):
        if len(nvp_dense_layers) == 0:
            build_nvp_dense_model()

        global student_layer
        if student_layer is None:
            student_layer = nn_extra_student.StudentRecurrentLayer(shape=(ndim,), corr_init=corr_init, learn_mu=False,
                                                                   nu_init=nu_init, exp_nu=True, square_var=True)

        x_shape = nn_extra_nvp.int_shape(x)
        x_bs = tf.reshape(x, (x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4]))
        x_bs_shape = nn_extra_nvp.int_shape(x_bs)

        log_det_jac = tf.zeros(x_bs_shape[0])

        y, log_det_jac = nn_extra_nvp.dequantization_forward_and_jacobian(x_bs, log_det_jac)
        y, log_det_jac = nn_extra_nvp.logit_forward_and_jacobian(y, log_det_jac)

        for layer in nvp_dense_layers:
            y, log_det_jac, _ = layer.forward_and_jacobian(y, log_det_jac, None)

        z_shape = nn_extra_nvp.int_shape(y)
        z_vec = tf.reshape(y, (x_shape[0], x_shape[1], -1))
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
                                    z_shape[1], z_shape[2], z_shape[3]))  # (bs*seq_len, z_img_shape)

            for layer in reversed(nvp_dense_layers):
                z_samples, _ = layer.backward(z_samples, None)

            # inverse logit
            x_samples = 1. / (1 + tf.exp(-z_samples))
            x_samples = tf.reshape(x_samples,
                                   (z_samples_shape[0], z_samples_shape[1], x_shape[2], x_shape[3], x_shape[4]))
            return x_samples

        log_probs = tf.stack(log_probs, axis=1)
        latent_log_probs = tf.stack(latent_log_probs, axis=1)
        latent_log_probs_prior = tf.stack(latent_log_probs_prior, axis=1)

        return log_probs, latent_log_probs, latent_log_probs_prior


def build_nvp_dense_model():
    global nvp_dense_layers

    for i in range(6):
        mask = 'even' if i % 2 == 0 else 'odd'
        name = '%s_%s' % (mask, i)
        nvp_dense_layers.append(
            nn_extra_nvp.CouplingLayerDense(mask, name=name, nonlinearity=nonlinearity, n_units=256,
                                            weight_norm=weight_norm))


def loss(log_probs):
    return -tf.reduce_mean(log_probs)
