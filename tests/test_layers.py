import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag

import nn_extra_gauss
import nn_extra_student
from tests import utils_stats
from tests.multivariate_student import multivariate_student

rng = np.random.RandomState(41)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 1000
seq_len = 100
p = 1
mu = np.ones((p,), dtype='float32') * 0.
recursive = True

g_nu = np.ones((p,), dtype='float32') * 150.
g_var = np.ones((p,), dtype='float32') * 1.
g_corr = np.ones((p,), dtype='float32') * 0.9
g_cov = g_corr

x_cov = np.ones((p,), dtype='float32') * 0.01
x_var = np.ones((p,), dtype='float32') * 1.


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
    phi = np.tile(mu, (seq_len, 1)).flatten()
    x1 = rng.multivariate_normal(phi, K)
    x1 = x1.reshape((seq_len, p))
    x1 = np.float32(x1)
    xs.append(x1[None, :, :])

x = np.concatenate(xs, axis=0)
print('shape x', x.shape)

x_var = tf.placeholder(tf.float32, shape=(batch_size, seq_len, p))
l_rnn = nn_extra_student.StudentRecurrentLayer(shape=(p,), nu_init=g_nu, mu_init=mu, var_init=g_var, corr_init=g_corr)
l_rnn2 = nn_extra_gauss.GaussianRecurrentLayer(shape=(p,), mu_init=mu, var_init=g_var, corr_init=g_corr)

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
    print('x_step', x[:, i, :])

probs_out_1 = np.asarray(probs_out).mean(axis=1)
probs_out_gauss_1 = np.asarray(probs_out_gauss).mean(axis=1)
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

print('---------------')  # numpy
print('\n ****** NUMPY ******')
x1 = xs[0][0]
x2 = xs[1][0]
print(x1.shape, x2.shape)

probs_k1, probs_k2 = [], []
for k in range(p):
    probs_k1.append(
        multivariate_student.pdf(x1[0, k:k + 1], phi=mu[k:k + 1], K=g_var[None, k:k + 1], nu=g_nu[k]))
    probs_k2.append(
        multivariate_student.pdf(x2[0, k:k + 1], phi=mu[k:k + 1], K=g_var[None, k:k + 1], nu=g_nu[k:k + 1]))
prob_1 = np.prod(probs_k1)
prob_2 = np.prod(probs_k2)
print('step', 0)
print('prob', np.log(prob_1), np.log(prob_2))
assert np.isclose(np.log(prob_1), probs_out[0][0], atol=1e-3, rtol=1e-03)
assert np.isclose(np.log(prob_2), probs_out[0][1], atol=1e-3, rtol=1e-03)
print('x_step', x1[0, :], x2[0, :])
print('---------------')

for i in range(1, seq_len):
    p_next1 = utils_stats.predictive_mdim_student_prob(x_1n=x1[:i], x_next=x1[i],
                                                       cov_u=g_cov,
                                                       var_u=g_var,
                                                       mu=mu, nu=g_nu, recursive=recursive,
                                                       verbose=False)
    p_next2 = utils_stats.predictive_mdim_student_prob(x_1n=x2[:i], x_next=x2[i],
                                                       cov_u=g_cov,
                                                       var_u=g_var,
                                                       mu=mu, nu=g_nu, recursive=recursive,
                                                       verbose=False)
    print('step', i)
    print('prob', np.log(p_next1), np.log(p_next2))
    assert np.isclose(np.log(p_next1), probs_out[i][0], atol=1e-3, rtol=1e-03)
    assert np.isclose(np.log(p_next2), probs_out[i][1], atol=1e-3, rtol=1e-03)
    print('x_step', x1[i, :], x2[i, :])
    print('---------------')

# check with joint probability
print('\n ********* JOINT PROBABILITY CHECK ********')
for j in range(2, seq_len + 1):
    probs_i1, probs_i2 = [], []
    for i in range(p):
        x_2n = x1[:j, i:i + 1].T
        K = create_covariance_matrix(j, 1, g_cov[i], g_var[i])
        pp_joint_n2 = multivariate_student.pdf(x_2n, phi=np.zeros_like(x_2n) + mu[i], K=K, nu=g_nu[i])

        x_1n = x1[:j - 1, i:i + 1].T
        K = create_covariance_matrix(j - 1, 1, g_cov[i], g_var[i])
        pp_joint_n1 = multivariate_student.pdf(x_1n, phi=np.zeros_like(x_1n) + mu[i], K=K, nu=g_nu[i])

        probs_i1.append(pp_joint_n2 / pp_joint_n1)

        x_2n = x2[:j, i:i + 1].T
        K = create_covariance_matrix(j, 1, g_cov[i], g_var[i])
        pp_joint_n2 = multivariate_student.pdf(x_2n, phi=np.zeros_like(x_2n) + mu[i], K=K, nu=g_nu[i])

        x_1n = x2[:j - 1, i:i + 1].T
        K = create_covariance_matrix(j - 1, 1, g_cov[i], g_var[i])
        pp_joint_n1 = multivariate_student.pdf(x_1n, phi=np.zeros_like(x_1n) + mu[i], K=K, nu=g_nu[i])

        probs_i2.append(pp_joint_n2 / pp_joint_n1)

    print('step', j - 1)
    print('prob', np.log(np.prod(probs_i1)), np.log(np.prod(probs_i2)))
    assert np.isclose(np.log(np.prod(probs_i1)), probs_out[j - 1][0], atol=1e-3, rtol=1e-03)
    assert np.isclose(np.log(np.prod(probs_i2)), probs_out[j - 1][1], atol=1e-3, rtol=1e-03)
    print('x_step', x1[j - 1, :], x2[j - 1, :])
    print('---------------')

# check with joint probability
print('\n ********* JOINT PROBABILITY CHECK GAUSS********')
for j in range(2, seq_len + 1):
    probs_i1, probs_i2 = [], []
    for i in range(p):
        pp = utils_stats.gaussian_predictive_theoretical_prob_slow(x1[:j - 1, i], x1[:j, i], mu[i], g_cov[i], g_var[i])
        probs_i1.append(pp)
        pp2 = utils_stats.gaussian_predictive_theoretical_prob_slow(x2[:j - 1, i], x2[:j, i], mu[i], g_cov[i], g_var[i])
        probs_i2.append(pp2)

    print('step', j - 1)
    print('prob', np.log(np.prod(probs_i1)), np.log(np.prod(probs_i2)))
    assert np.isclose(np.log(np.prod(probs_i1)), probs_out_gauss[j - 1][0], atol=1e-3, rtol=1e-03)
    assert np.isclose(np.log(np.prod(probs_i2)), probs_out_gauss[j - 1][1], atol=1e-3, rtol=1e-03)
    print('x_step', x1[j - 1, :], x2[j - 1, :])
    print('---------------')
