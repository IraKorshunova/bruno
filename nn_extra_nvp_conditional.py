import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def int_shape(x):
    return list(map(int, x.get_shape()))


def logit_forward_and_jacobian(x, sum_log_det_jacobians):
    alpha = 1e-5
    y = x * (1 - alpha) + alpha * 0.5
    jac = tf.reduce_sum(-tf.log(y) - tf.log(1 - y), [1, 2, 3])
    y = tf.log(y) - tf.log(1. - y)
    sum_log_det_jacobians += jac
    return y, sum_log_det_jacobians


def dequantization_forward_and_jacobian(x, sum_log_det_jacobians):
    x_shape = int_shape(x)
    y = x / 256.0
    sum_log_det_jacobians -= tf.log(256.0) * x_shape[1] * x_shape[2] * x_shape[3]
    return y, sum_log_det_jacobians


class Layer():
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z, y_label):
        raise NotImplementedError(str(type(self)))

    def backward(self, y, z, y_label):
        raise NotImplementedError(str(type(self)))


class CouplingLayerConv(Layer):
    def __init__(self, mask_type, name='CouplingLayer', nonlinearity=tf.nn.relu, weight_norm=True, num_filters=64,
                 num_res_blocks=8):
        self.mask_type = mask_type
        self.name = name
        self.nonlinearity = nonlinearity
        self.weight_norm = weight_norm
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks

    def function_s_t(self, x, mask, y_label, name='function_s_t'):
        if self.weight_norm:
            return self.function_s_t_wn(x, mask, y_label, name + '_wn')
        else:
            with tf.variable_scope(name):
                num_filters = self.num_filters
                xs = int_shape(x)
                kernel_size = 3
                n_input_channels = xs[3]

                y = conv2d(x, num_filters, 1, nonlinearity=self.nonlinearity,
                           kernel_initializer=Orthogonal(), name='c1', y_label=y_label)

                skip = y
                for r in range(self.num_res_blocks):
                    y = conv2d(y, num_filters, kernel_size, nonlinearity=self.nonlinearity,
                               kernel_initializer=Orthogonal(), name='c2_%d' % r, y_label=y_label)

                    y = conv2d(y, num_filters, kernel_size, nonlinearity=None,
                               kernel_initializer=Orthogonal(), name='c3_%d' % r, y_label=y_label)

                    y += skip
                    y = self.nonlinearity(y)
                    skip = y

                l_scale = conv2d(y, n_input_channels, 1, nonlinearity=tf.tanh,
                                 kernel_initializer=tf.constant_initializer(0.),
                                 bias_initializer=tf.constant_initializer(0.),
                                 name='conv_out_scale', y_label=y_label)

                l_scale *= 1 - mask

                m_translation = conv2d(y, n_input_channels, 1, nonlinearity=None,
                                       kernel_initializer=tf.constant_initializer(0.),
                                       bias_initializer=tf.constant_initializer(0.),
                                       name='conv_out_translation', y_label=y_label)
                m_translation *= 1 - mask

                return l_scale, m_translation

    def function_s_t_wn(self, x, mask, y_label, name):
        with tf.variable_scope(name):
            num_filters = self.num_filters
            xs = int_shape(x)
            kernel_size = 3
            n_input_channels = xs[3]

            y = conv2d_wn(x, num_filters, 'c1', filter_size=[1, 1], nonlinearity=self.nonlinearity, y_label=y_label)

            skip = y
            for r in range(self.num_res_blocks):
                y = conv2d_wn(y, num_filters, 'c2_%d' % r, filter_size=[kernel_size, kernel_size],
                              nonlinearity=self.nonlinearity, y_label=y_label)
                y = conv2d_wn(y, num_filters, 'c3_%d' % r, filter_size=[kernel_size, kernel_size], nonlinearity=None,
                              y_label=y_label)

                y += skip
                y = self.nonlinearity(y)
                skip = y

            l_scale = conv2d(y, n_input_channels, 1, nonlinearity=tf.tanh,
                             kernel_initializer=tf.constant_initializer(0.),
                             bias_initializer=tf.constant_initializer(0.),
                             name='conv_out_scale', y_label=y_label)
            l_scale *= 1 - mask

            m_translation = conv2d(y, n_input_channels, 1, nonlinearity=None,
                                   kernel_initializer=tf.constant_initializer(0.),
                                   bias_initializer=tf.constant_initializer(0.),
                                   name='conv_out_translation', y_label=y_label)
            m_translation *= 1 - mask

            return l_scale, m_translation

    def get_mask(self, xs, mask_type):

        assert self.mask_type in ['checkerboard0', 'checkerboard1', 'channel0', 'channel1']

        if 'checkerboard' in mask_type:
            unit0 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
            unit1 = -unit0 + 1.0
            unit = unit0 if mask_type == 'checkerboard0' else unit1
            unit = tf.reshape(unit, [1, 2, 2, 1])
            b = tf.tile(unit, [xs[0], xs[1] // 2, xs[2] // 2, xs[3]])
        else:
            white = tf.ones([xs[0], xs[1], xs[2], xs[3] // 2])
            black = tf.zeros([xs[0], xs[1], xs[2], xs[3] // 2])
            if mask_type == 'channel0':
                b = tf.concat([white, black], 3)
            else:
                b = tf.concat([black, white], 3)
        return b

    def forward_and_jacobian(self, x, sum_log_det_jacobians, z, y_label=None):
        with tf.variable_scope(self.name):
            xs = int_shape(x)
            b = self.get_mask(xs, self.mask_type)

            # masked half of x
            x1 = x * b
            l, m = self.function_s_t(x1, b, y_label)
            y = x1 + tf.multiply(1. - b, x * tf.exp(l) + m)
            log_det_jacobian = tf.reduce_sum(l, [1, 2, 3])
            sum_log_det_jacobians += log_det_jacobian

            return y, sum_log_det_jacobians, z

    def backward(self, y, z, y_label=None):
        with tf.variable_scope(self.name, reuse=True):
            ys = int_shape(y)
            b = self.get_mask(ys, self.mask_type)

            y1 = y * b
            l, m = self.function_s_t(y1, b, y_label)
            x = y1 + tf.multiply(y * (1. - b) - m, tf.exp(-l))
            return x, z


class CouplingLayerDense(CouplingLayerConv):
    def __init__(self, mask_type, name='CouplingDense', nonlinearity=tf.nn.relu, n_units=1024, weight_norm=True):
        super(CouplingLayerDense, self).__init__(mask_type, name, nonlinearity, weight_norm)
        self.mask_type = mask_type
        self.name = name
        self.nonlinearity = nonlinearity
        self.n_units = n_units
        self.weight_norm = weight_norm

    def get_mask(self, xs, mask_type):

        assert self.mask_type in ['even', 'odd']

        ndim = tf.reduce_prod(xs[1:])

        b = tf.range(ndim)
        if 'even' in mask_type:
            # even = checkerboard 0
            b = tf.cast(tf.mod(b, 2), tf.float32)
        elif 'odd' in mask_type:
            # odd = checkerboard 1
            b = 1. - tf.cast(tf.mod(b, 2), tf.float32)

        b_mask = tf.ones((xs[0], ndim))
        b_mask = b_mask * b

        b_mask = tf.reshape(b_mask, xs)

        bs = int_shape(b_mask)
        assert bs == xs

        return b_mask

    def function_s_t(self, x, mask, y_label, name='function_s_t_dense'):
        if self.weight_norm:
            return self.function_s_t_wn(x, mask, y_label, name + '_wn')
        else:
            with tf.variable_scope(name):
                xs = int_shape(x)
                y = tf.reshape(x, (xs[0], -1))
                ndim = int_shape(y)[-1]

                y = dense(y, num_units=self.n_units, nonlinearity=self.nonlinearity,
                          kernel_initializer=Orthogonal(),
                          bias_initializer=tf.constant_initializer(0.01), name='d1', y_label=y_label)
                y = dense(y, num_units=self.n_units, nonlinearity=self.nonlinearity,
                          kernel_initializer=Orthogonal(),
                          bias_initializer=tf.constant_initializer(0.01), name='d2', y_label=y_label)

                l_scale = dense(y, num_units=ndim, nonlinearity=tf.tanh,
                                kernel_initializer=tf.constant_initializer(0.),
                                bias_initializer=tf.constant_initializer(0.), name='d_scale', y_label=y_label)
                l_scale = tf.reshape(l_scale, shape=xs)
                l_scale *= 1 - mask

                m_translation = dense(y, num_units=ndim, nonlinearity=None,
                                      kernel_initializer=tf.constant_initializer(0.),
                                      bias_initializer=tf.constant_initializer(0.), name='d_translate', y_label=y_label)
                m_translation = tf.reshape(m_translation, shape=xs)
                m_translation *= 1 - mask

                return l_scale, m_translation

    def function_s_t_wn(self, x, mask, y_label, name):
        with tf.variable_scope(name):
            xs = int_shape(x)
            y = tf.reshape(x, (xs[0], -1))
            ndim = int_shape(y)[-1]

            y = dense_wn(y, units=self.n_units, name='d1', activation=self.nonlinearity, y_label=y_label)
            y = dense_wn(y, units=self.n_units, name='d2', activation=self.nonlinearity, y_label=y_label)

            l_scale = dense(y, num_units=ndim, nonlinearity=tf.tanh,
                            kernel_initializer=tf.constant_initializer(0.),
                            bias_initializer=tf.constant_initializer(0.), name='d_scale', y_label=y_label)
            l_scale = tf.reshape(l_scale, shape=xs)
            l_scale *= 1 - mask

            m_translation = dense(y, num_units=ndim, nonlinearity=None,
                                  kernel_initializer=tf.constant_initializer(0.),
                                  bias_initializer=tf.constant_initializer(0.), name='d_translate',
                                  y_label=y_label)
            m_translation = tf.reshape(m_translation, shape=xs)
            m_translation *= 1 - mask

            return l_scale, m_translation


class SqueezingLayer(Layer):
    def __init__(self, name="Squeeze"):
        self.name = name

    def forward_and_jacobian(self, x, sum_log_det_jacobians, z, y_label=None):
        xs = int_shape(x)
        assert xs[1] % 2 == 0 and xs[2] % 2 == 0
        y = tf.space_to_depth(x, 2)
        if z is not None:
            z = tf.space_to_depth(z, 2)

        return y, sum_log_det_jacobians, z

    def backward(self, y, z, y_label=None):
        ys = int_shape(y)
        assert ys[3] % 4 == 0
        x = tf.depth_to_space(y, 2)

        if z is not None:
            z = tf.depth_to_space(z, 2)

        return x, z


class FactorOutLayer(Layer):
    def __init__(self, scale, name='FactorOut'):
        self.scale = scale
        self.name = name

    def forward_and_jacobian(self, x, sum_log_det_jacobians, z, y_label=None):

        xs = int_shape(x)
        split = xs[3] // 2

        # The factoring out is done on the channel direction.
        # Haven't experimented with other ways of factoring out.
        new_z = x[:, :, :, :split]
        x = x[:, :, :, split:]

        if z is not None:
            z = tf.concat([z, new_z], 3)
        else:
            z = new_z

        return x, sum_log_det_jacobians, z

    def backward(self, y, z, y_label=None):

        # At scale 0, 1/2 of the original dimensions are factored out
        # At scale 1, 1/4 of the original dimensions are factored out
        # ....
        # At scale s, (1/2)^(s+1) are factored out
        # Hence, at backward pass of scale s, (1/2)^(s) of z should be factored in

        zs = int_shape(z)
        if y is None:
            split = zs[3] // (2 ** self.scale)
        else:
            split = int_shape(y)[3]
        new_y = z[:, :, :, -split:]
        z = z[:, :, :, :-split]

        assert (int_shape(new_y)[3] == split)

        if y is not None:
            x = tf.concat([new_y, y], 3)
        else:
            x = new_y

        return x, z


class Orthogonal(object):
    """
    Lasagne orthogonal init from OpenAI
    """

    def __init__(self, scale=1.):
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)

    def get_config(self):
        return {
            'scale': self.scale
        }


def dense(x, num_units, name, nonlinearity=None, kernel_initializer=Orthogonal(),
          bias_initializer=tf.constant_initializer(0.), y_label=None):
    with tf.variable_scope(name):
        if y_label is None:
            return tf.layers.dense(x, units=num_units, activation=nonlinearity,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer, name=name)
        else:
            ndim = int(int_shape(x)[-1] / 2)
            h = tf.layers.dense(y_label, units=ndim, activation=tf.nn.leaky_relu,
                                use_bias=True,
                                kernel_initializer=Orthogonal(), name='label_W1')
            o1 = tf.concat([h, x], axis=-1)
            output = tf.layers.dense(o1, units=num_units, activation=None,
                                     use_bias=True,
                                     kernel_initializer=Orthogonal(), name=name)

            if nonlinearity is not None:
                output = nonlinearity(output)
            return output


def conv2d(x, num_filters, kernel_size, name, pad='same', nonlinearity=None,
           kernel_initializer=tf.constant_initializer(0.),
           bias_initializer=tf.constant_initializer(0.), y_label=None):
    with tf.variable_scope(name):
        if y_label is None:
            return tf.layers.conv2d(x, num_filters, kernel_size, padding=pad, activation=nonlinearity,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    name=name)
        else:
            h = tf.layers.dense(y_label, units=num_filters, activation=None, use_bias=False,
                                kernel_initializer=Orthogonal(), name='label_h')
            output = tf.layers.conv2d(x, num_filters, kernel_size, padding=pad, activation=None,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=True,
                                      name=name)
            output = output + h[:, None, None, :]

            if nonlinearity is not None:
                output = nonlinearity(output)
            return output


@add_arg_scope
def conv2d_wn(x, num_filters, name, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None, init_scale=1.,
              init=False, ema=None, y_label=None, trainable_bias=True):
    with tf.variable_scope(name):
        if y_label is None:
            V = get_var_maybe_avg('V', ema, shape=filter_size + [int(x.get_shape()[-1]), num_filters], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                                  initializer=tf.constant_initializer(1.), trainable=True)
            b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.), trainable=trainable_bias)

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

            if init:
                m_init, v_init = tf.nn.moments(x, [0, 1, 2])
                scale_init = init_scale / tf.sqrt(v_init + 1e-10)
                with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                    x = tf.identity(x)

            if nonlinearity is not None:
                x = nonlinearity(x)
            return x

        if y_label is not None:
            h = dense_wn(y_label, units=num_filters, activation=None, use_bias=False,
                         init=init, name='label_h')

            output = conv2d_wn(x, num_filters, name='conv_h', nonlinearity=None, init=init, trainable_bias=True)

            output = output + h[:, None, None, :]

            if nonlinearity is not None:
                output = nonlinearity(output)
            return output


@add_arg_scope
def dense_wn(x, units, name, activation=None, use_bias=True, init_scale=1., init=False, ema=None, y_label=None):
    with tf.variable_scope(name):
        if y_label is None:
            V = get_var_maybe_avg('V', ema, shape=[int(x.get_shape()[1]), units], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            g = get_var_maybe_avg('g', ema, shape=[units], dtype=tf.float32,
                                  initializer=tf.constant_initializer(1.), trainable=True)
            b = get_var_maybe_avg('b', ema, shape=[units], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.), trainable=use_bias)

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.norm(V, axis=0)

            x = tf.reshape(scaler, [1, units]) * x + tf.reshape(b, [1, units])
            if init:
                m_init, v_init = tf.nn.moments(x, [0])
                scale_init = init_scale / tf.sqrt(v_init + 1e-10)
                with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                    x = tf.identity(x)

            if activation is not None:
                x = activation(x)
            return x
        else:
            ndim = int(int_shape(x)[-1] / 2)
            h = dense_wn(y_label, units=ndim, activation=tf.nn.leaky_relu, use_bias=True,
                         init=init, name='label_h')

            o1 = tf.concat([h, x], axis=-1)
            output = dense_wn(o1, units=units, activation=None, use_bias=True, init=init, name=name)

            if activation is not None:
                output = activation(output)
            return output


def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars
