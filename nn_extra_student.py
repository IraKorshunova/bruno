import collections
import numpy as np
import tensorflow as tf

Student = collections.namedtuple('Student', ['mu', 'var', 'nu'])
State = collections.namedtuple('State', ['num_observations', 'beta', 'x_sum', 'k'])


def inv_softplus(x):
    return np.log(1 - np.exp(-x)) + x


def inv_sigmoid(x):
    return np.log(x) - np.log(1. - x)


class StudentRecurrentLayer(object):
    def __init__(self, shape,
                 nu_init=2.5,
                 mu_init=0.,
                 var_init=1.,
                 corr_init=0.5,
                 learn_nu=True,
                 learn_mu=True,
                 min_nu=2.,
                 exp_nu=False,
                 square_var=False,
                 tied_nu=False):
        self.seed_rng = np.random.RandomState(42)

        self._shape = shape

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
        if square_var:
            self.var = tf.square(tf.nn.softplus(self.var_vbl))
        else:
            self.var = tf.nn.softplus(self.var_vbl)

        if learn_nu:
            if tied_nu:
                assert len(shape) == 1
                self.nu_vbl = tf.get_variable(
                    "prior_nu",
                    (1, 1),
                    tf.float32,
                    tf.constant_initializer(np.log(nu_init - min_nu))
                )
                self.nu = tf.tile(tf.exp(self.nu_vbl) + min_nu, (1,) + shape)
            else:
                if exp_nu:
                    self.nu_vbl = tf.get_variable(
                        "prior_nu",
                        (1,) + shape,
                        tf.float32,
                        tf.constant_initializer(np.log(nu_init - min_nu))
                    )
                    self.nu = tf.exp(self.nu_vbl) + min_nu
                else:
                    self.nu_vbl = tf.get_variable(
                        "prior_nu",
                        (1,) + shape,
                        tf.float32,
                        tf.constant_initializer(inv_softplus(nu_init - min_nu))
                    )
                    self.nu = tf.nn.softplus(self.nu_vbl) + min_nu
        else:
            self.nu = tf.ones((1,) + shape, name='prior_nu') * nu_init

        self.prior = Student(
            self.mu,
            self.var,
            self.nu,
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
        self._state = State(0., 0., 0., self.prior.var)

    @property
    def variables(self):
        return (self.mu, self.var_vbl, self.nu_vbl, self.corr_vbl)

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
        self._state = State(0., 0., 0., self.prior.var)

    def update_distribution(self, observation):
        mu, sigma, nu = self.current_distribution
        i, beta, x_sum, k = self._state
        x = observation
        x_zm = x - self.mu
        x_sum_out = x_sum + x_zm
        i += 1
        dd = self.cov / (self.var + self.cov * (i - 1.))
        nu_out = nu + 1
        mu_out = (1. - dd) * mu + observation * dd

        a_i = (self.cov * (i - 2.) + self.var) / ((self.var - self.cov) * (self.cov * (i - 1.) + self.var))
        b_i = -1. * self.cov / ((self.var - self.cov) * (self.cov * (i - 1.) + self.var))
        b_i_prev = -1. * self.cov / ((self.var - self.cov) * (self.cov * (i - 2.) + self.var))

        beta_out = beta + (a_i - b_i) * tf.square(x_zm) + b_i * tf.square(x_sum + x_zm) - b_i_prev * tf.square(x_sum)
        k_out = (1. - dd) * k + (self.var - self.cov) * dd

        sigma_out = (self.nu + beta_out - 2.) / (nu_out - 2.) * k_out

        self.current_distribution = Student(mu_out, sigma_out, nu_out)
        self._state = State(i, beta_out, x_sum_out, k_out)

    # def get_log_likelihood(self, observation, mask_dim=None, eps=1e-12):
    #     x = observation
    #     mu, var, nu = self.current_distribution
    #     var += eps
    #     log_pdf = -0.5 * tf.log(2. * np.pi * var) - tf.square(x - mu) / (2. * var)
    #     if mask_dim is not None:
    #         return tf.reduce_sum(log_pdf * mask_dim, 1)
    #     else:
    #         return tf.reduce_sum(log_pdf, 1)

    def get_log_likelihood(self, observation, mask_dim=None, eps=1e-12):
        x = observation
        mu, var, nu = self.current_distribution
        # var += eps
        ln_gamma_quotient = tf.lgamma((1. + nu) / 2.) - tf.lgamma(nu / 2.)
        ln_nom = (-(1. + nu) / 2.) * tf.log1p(tf.square(x - mu) / ((nu - 2.) * var))
        ln_denom = 0.5 * tf.log((nu - 2.) * np.pi * var)
        log_pdf = ln_gamma_quotient + ln_nom - ln_denom
        if mask_dim is not None:
            return tf.reduce_sum(log_pdf * mask_dim, 1)
        else:
            return tf.reduce_sum(log_pdf, 1)

    def get_log_likelihood_per_dim(self, observation, mask_dim=None, eps=1e-12):
        x = observation
        mu, var, nu = self.current_distribution
        # var += eps
        ln_gamma_quotient = tf.lgamma((1. + nu) / 2.) - tf.lgamma(nu / 2.)
        ln_nom = (-(1. + nu) / 2.) * tf.log1p(tf.square(x - mu) / ((nu - 2.) * var))
        ln_denom = 0.5 * tf.log((nu - 2.) * np.pi * var)
        log_pdf = ln_gamma_quotient + ln_nom - ln_denom
        return log_pdf

    def get_log_likelihood_under_prior(self, observation, mask_dim=None, eps=1e-12):
        x = observation
        mu, var, nu = self.prior
        # var += eps
        ln_gamma_quotient = tf.lgamma((1. + nu) / 2.) - tf.lgamma(nu / 2.)
        ln_nom = (-(1. + nu) / 2.) * tf.log1p((tf.square(x - mu) / ((nu - 2.) * var)))
        ln_denom = 0.5 * tf.log((nu - 2.) * np.pi * var)
        log_pdf = ln_gamma_quotient + ln_nom - ln_denom
        if mask_dim is not None:
            return tf.reduce_sum(log_pdf * mask_dim, 1)
        else:
            return tf.reduce_sum(log_pdf, 1)

    def sample(self, nr_samples=1):
        mu, var, nu = self.current_distribution

        rvs = tf.random_uniform(
            shape=tf.TensorShape([2, nr_samples]).concatenate(mu.shape),
            seed=self.seed_rng.randint(317070),
            name="Student_sampler"
        )
        a = tf.reduce_min(rvs, axis=0)
        b = tf.reduce_max(rvs, axis=0)

        u = b * tf.cos(2 * np.pi * a / b)
        v = b * tf.sin(2 * np.pi * a / b)
        w = tf.square(u) + tf.square(v)
        t = u * tf.sqrt(nu * (tf.pow(w, -2. / nu) - 1) / w)
        t_sample = mu + tf.sqrt(var * (nu - 2) / nu) * t

        return t_sample
