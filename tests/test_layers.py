import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag

import nn_extra_gauss
import nn_extra_student

rng = np.random.RandomState(41)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 1
seq_len = 100
p = 1
recursive = True

g_nu = np.ones((p,), dtype='float32') * 30
g_var = np.ones((p,), dtype='float32') * 0.13
g_corr = np.ones((p,), dtype='float32') * 0.002
g_cov = g_corr * g_var
g_mu = np.ones((p,), dtype='float32') * 0.

x_cov = np.ones((p,), dtype='float32') * 0.05
x_var = np.ones((p,), dtype='float32') * 0.1
x_mu = -np.ones((p,), dtype='float32') * 0.17


# create covariance matrix
def create_covariance_matrix(n, p, cov_u, var_u):
    if p == 1:
        K = np.ones((n, n)) * cov_u
        np.fill_diagonal(K, var_u)
    else:
        v = np.eye(p)
        diagonal = [v] * n
        K_v = block_diag(*diagonal)

        r = np.ones((p, p)) * cov_u
        np.fill_diagonal(r, var_u)
        diagonal = [r] * n
        K_r = block_diag(*diagonal)

        K = np.kron(np.ones((n, n)), r)

        K = K - K_r + K_v

    return K


K = create_covariance_matrix(seq_len, p, x_cov, x_var)
xs = []
for i in range(batch_size):
    phi = np.tile(x_mu, (seq_len, 1)).flatten()
    x1 = rng.multivariate_normal(phi, K)
    # x1 = multivariate_student.sample(phi=phi, K=K, nu=30, rng=rng)
    x1 = x1.reshape((seq_len, p))
    x1 = np.float32(x1)
    xs.append(x1[None, :, :])

x = np.concatenate(xs, axis=0)
print('shape x', x.shape)
x[0][-1] -= 10
x[0][-10] -= 5
x[0][-20] -= 3

x[0, :, 0] = np.array([-0.35260147, 0.22195661, -0.07069921, -0.3709321, 0.00873423,
                       0.13817436, -0.3873244, 0.47131354, -0.683796, 0.3346839,
                       -0.28066206, -0.08021778, 0.18474007, -0.6954666, 0.25559473,
                       0.05053151, 0.0854584, 0.03379792, 0.17560339, -0.2948599,
                       0.09350818, 0.2055502, -0.23348409, 0.4034475, -0.48395532,
                       -0.44995946, 0.01072145, -0.02041304, 0.07376307, 0.18082517,
                       0.21721125, 0.04213548, 0.49464414, 0.36044407, 0.22032273,
                       -0.10874593, -0.1162864, 0.49828446, 0.35410953, 0.03654575,
                       -0.27813053, 0.09882653, 0.1532216, -0.09235597, 0.03861177,
                       0.07748926, -0.01355469, 0.26316696, -0.49216962, -0.48428404,
                       0.22735786, 0.15967107, 0.18991512, -0.4516616, -0.34925675,
                       0.12077135, 0.46243125, 0.17621005, -0.59641767, 0.3900391,
                       -0.5424466, -0.4373713, 0.13044262, -0.01445735, 0.4255762,
                       0.59966636, 0.26922202, 0.43725386, -0.31793606, 0.2314139,
                       0.22227955, -0.34927458, 0.22491741, -0.04603827, 0.1691637,
                       0.19004703, -0.19159842, -0.5567625, -0.35111427, -0.07074428,
                       -0.31092787, -0.16381532, 0.01810074, -0.03907657, 0.19226861,
                       -0.37199593, 0.04148138, -0.3251711, -0.03755003, -0.11645389,
                       -0.28840822, -0.00146043, 0.42054844, -0.6171824, -0.2661503,
                       0.4482042, 0.10480976, 0.51177275, -0.36583614, -0.7287407],
                      dtype=np.float32)
# x[1, :, 0] = x[0, :, 0]

x_var = tf.placeholder(tf.float32, shape=(batch_size, seq_len, p))
l_rnn = nn_extra_student.StudentRecurrentLayer(shape=(p,), nu_init=g_nu, mu_init=g_mu, var_init=g_var, corr_init=g_corr)
l_rnn2 = nn_extra_gauss.GaussianRecurrentLayer(shape=(p,), mu_init=g_mu, var_init=g_var, corr_init=g_corr)

probs = []
probs_gauss = []
with tf.variable_scope("one_step") as scope:
    l_rnn.reset()
    l_rnn2.reset()
    for i in range(seq_len):
        prob_i = l_rnn.get_log_likelihood(x_var[:, i, :])
        probs.append(prob_i)
        l_rnn.update_distribution(x_var[:, i, :])

        prob_i = l_rnn2.get_log_likelihood(x_var[:, i, :])
        probs_gauss.append(prob_i)
        l_rnn2.update_distribution(x_var[:, i, :])
        scope.reuse_variables()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    feed_dict = {x_var: x}
    probs_out = sess.run(probs, feed_dict)
    probs_out_gauss = sess.run(probs_gauss, feed_dict)

for i in range(seq_len):
    print('step', i)
    print('prob', probs_out[i])
    print('prob_gauss', probs_out_gauss[i])
    print(x[0, i])

probs_out_1 = np.asarray(probs_out)
probs_out_gauss_1 = np.asarray(probs_out_gauss)
target_path = '/mnt/storage/users/ikorshun/bruno/metadata/'
fig = plt.figure(figsize=(4, 3))
plt.grid(True, which="both", ls="-", linewidth='0.2')
plt.plot(range(len(probs_out_1)), probs_out_1, 'black', linewidth=1.)
plt.scatter(range(len(probs_out_1)), probs_out_1, s=1.5, c='black')
plt.plot(range(len(probs_out_gauss_1)), probs_out_gauss_1, 'red', linewidth=1.)
plt.scatter(range(len(probs_out_gauss_1)), probs_out_gauss_1, s=1.5, c='red')
plt.xlabel('step')
plt.savefig(
    target_path + '/nll_plot.png',
    bbox_inches='tight', dpi=600)
