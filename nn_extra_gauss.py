import collections

import numpy as np
import tensorflow as tf

Gaussian = collections.namedtuple('Gaussian', ['mu', 'var'])
State = collections.namedtuple('State', ['num_observations', 'x_sum'])


def inv_softplus(x):
    return np.log(1 - np.exp(-x)) + x


def inv_sigmoid(x):
    return np.log(x) - np.log(1. - x)


class GaussianRecurrentLayer(object):
    def __init__(self, shape,
                 mu_init=0.,
                 var_init=1.,
                 corr_init=0.5,
                 learn_mu=True):
        self.seed_rng = np.random.RandomState(42)

        self._shape = shape

        with tf.variable_scope("gaussian"):

            if learn_mu:
                self.mu = tf.get_variable(
                    "prior_mean",
                    (1,) + shape,
                    tf.float32,
                    tf.constant_initializer(mu_init)
                )
            else:
                self.mu = tf.ones((1,) + shape, name='prior_mean') * mu_init

            self.var_vbl = tf.get_variable(
                "prior_var",
                (1,) + shape,
                tf.float32,
                tf.constant_initializer(inv_softplus(var_init))
            )
            self.var = tf.nn.softplus(self.var_vbl)

            self.prior = Gaussian(
                self.mu,
                self.var
            )

            self.corr_vbl = tf.get_variable(
                "prior_corr",
                (1,) + shape,
                tf.float32,
                tf.constant_initializer(inv_sigmoid(corr_init))
            )
            self.corr = tf.sigmoid(self.corr_vbl)
            self.cov = tf.sigmoid(self.corr_vbl) * self.var

            self.current_distribution = self.prior
            self._state = State(0., 0.)

    @property
    def variables(self):
        return self.mu, self.var_vbl, self.corr_vbl

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        if "tf" in type(self._shape):
            return tf.reduce_prod(self._shape)
        else:
            np.prod(self._shape)

    def reset(self):
        self.current_distribution = self.prior
        self._state = State(0., 0.)

    def update_distribution(self, observation, eps=1e-12):
        mu, sigma = self.current_distribution
        i, x_sum = self._state
        x = observation
        x_zm = x - self.mu
        x_sum_out = x_sum + x_zm
        i += 1
        dd = self.cov / (self.var + self.cov * (i - 1.) + eps)
        mu_out = (1. - dd) * mu + observation * dd
        var_out = (1. - dd) * sigma + (self.var - self.cov) * dd

        self.current_distribution = Gaussian(mu_out, var_out)
        self._state = State(i, x_sum_out)

    def get_log_likelihood(self, observation, mask_dim=None, eps=1e-12):
        x = observation
        mu, var = self.current_distribution
        var += eps
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(x - mu) / (2. * var)
        if mask_dim is not None:
            return tf.reduce_sum(log_pdf * mask_dim, 1)
        else:
            return tf.reduce_sum(log_pdf, 1)

    def get_log_likelihood_under_prior(self, observation, mask_dim=None, eps=1e-12):
        x = observation
        mu, var = self.prior
        var += eps
        log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(x - mu) / (2. * var)
        if mask_dim is not None:
            return tf.reduce_sum(log_pdf * mask_dim, 1)
        else:
            return tf.reduce_sum(log_pdf, 1)

    def sample(self, nr_samples=1):
        mu, var = self.current_distribution

        return tf.random_normal(
            shape=tf.TensorShape([nr_samples]).concatenate(mu.shape),
            mean=mu,
            stddev=tf.sqrt(var),
            seed=self.seed_rng.randint(317070),
            name="Normal_sampler"
        )
