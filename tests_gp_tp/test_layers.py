import math

import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag

import nn_extra_gauss
import nn_extra_student
from tests_gp_tp import utils_stats
from tests_gp_tp.multivariate_student import multivariate_student

rng = np.random.RandomState(41)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 1
seq_len = 100
p = 1
recursive = True

plot_name = '/nll_plot0.png'

g_nu = np.ones((p,), dtype='float32') * 10000
g_var = np.ones((p,), dtype='float32') * 1.
g_corr = np.ones((p,), dtype='float32') * 0.01
g_cov = g_corr * g_var
g_mu = np.ones((p,), dtype='float32') * 0.

x_cov = np.ones((p,), dtype='float32') * 0.05
x_var = np.ones((p,), dtype='float32') * 0.1
x_mu = np.ones((p,), dtype='float32') * 0.


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
    x1 = x1.reshape((seq_len, p))
    x1 = np.float32(x1)
    xs.append(x1[None, :, :])

x = np.concatenate(xs, axis=0)
print('shape x', x.shape)

x_var_tf = tf.placeholder(tf.float32, shape=(batch_size, seq_len, p))
l_rnn = nn_extra_student.StudentRecurrentLayer(shape=(p,), nu_init=g_nu, mu_init=g_mu, var_init=g_var, corr_init=g_corr)
l_rnn2 = nn_extra_gauss.GaussianRecurrentLayer(shape=(p,), mu_init=g_mu, var_init=g_var, corr_init=g_corr)

probs = []
probs_gauss = []
with tf.variable_scope("one_step") as scope:
    l_rnn.reset()
    l_rnn2.reset()
    for i in range(seq_len):
        prob_i = l_rnn.get_log_likelihood(x_var_tf[:, i, :])
        probs.append(prob_i)
        l_rnn.update_distribution(x_var_tf[:, i, :])

        prob_i = l_rnn2.get_log_likelihood(x_var_tf[:, i, :])
        probs_gauss.append(prob_i)
        l_rnn2.update_distribution(x_var_tf[:, i, :])
        scope.reuse_variables()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    feed_dict = {x_var_tf: x}
    probs_out = sess.run(probs, feed_dict)
    probs_out_gauss = sess.run(probs_gauss, feed_dict)

for i in range(seq_len):
    print('step', i)
    print('prob', probs_out[i])
    print('prob_gauss', probs_out_gauss[i])
    print(x[0, i])

probs_out_1 = np.asarray(probs_out)
probs_out_gauss_1 = np.asarray(probs_out_gauss)
fig = plt.figure(figsize=(4, 3))
plt.grid(True, which="both", ls="-", linewidth='0.2')
plt.plot(range(len(probs_out_1)), probs_out_1, 'black', linewidth=1., label='TP')
plt.scatter(range(len(probs_out_1)), probs_out_1, s=1.5, c='black')
plt.plot(range(len(probs_out_gauss_1)), probs_out_gauss_1, 'red', linewidth=1., label='GP')
plt.scatter(range(len(probs_out_gauss_1)), probs_out_gauss_1, s=1.5, c='red')
plt.xlabel('step')
plt.ylabel('predictive log-probability')
plt.legend()
plt.title('Model: ν=%s μ=%s v=%s ρ=%s \n Data: μ=%s v=%s  ρ=%s' % (g_nu[0], g_mu[0], g_var[0], g_cov[0], x_mu[0],
                                                                   x_var[0], x_cov[0]))
plt.savefig('metadata/' + plot_name, bbox_inches='tight', dpi=600)

print('---------------')  # numpy
print('\n ****** NUMPY ******')
x1 = x[0]
print(x1.shape)

probs_k1, probs_k2 = [], []
for k in range(p):
    probs_k1.append(
        multivariate_student.pdf(x1[0, k:k + 1], phi=g_mu[k:k + 1], K=g_var[None, k:k + 1], nu=g_nu[k]))
prob_1 = np.prod(probs_k1)
print('step', 0)
print('prob', np.log(prob_1))
assert np.isclose(np.log(prob_1), probs_out[0][0], atol=1e-3, rtol=1e-03)
print('x_step', x1[0, :])
print('---------------')

for i in range(1, seq_len):
    p_next1 = utils_stats.predictive_mdim_student_prob(x_1n=x1[:i], x_next=x1[i],
                                                       cov_u=g_cov,
                                                       var_u=g_var,
                                                       mu=g_mu, nu=g_nu, recursive=recursive,
                                                       verbose=False)
    print('step', i)
    print('prob', np.log(p_next1))
    print(np.log(p_next1), probs_out[i][0])
    print('x_step', x1[i, :])
    assert np.isclose(np.log(p_next1), probs_out[i][0], atol=1e-3, rtol=1e-03)
    print('---------------')

# check with joint probability
print('\n ********* JOINT PROBABILITY CHECK ********')
for j in range(2, seq_len + 1):
    probs_i1 = []
    for i in range(p):
        x_2n = x1[:j, i:i + 1].T
        K = create_covariance_matrix(j, 1, g_cov[i], g_var[i])
        pp_joint_n2 = multivariate_student.pdf(x_2n, phi=np.zeros_like(x_2n) + g_mu[i], K=K, nu=g_nu[i])

        x_1n = x1[:j - 1, i:i + 1].T
        K = create_covariance_matrix(j - 1, 1, g_cov[i], g_var[i])
        pp_joint_n1 = multivariate_student.pdf(x_1n, phi=np.zeros_like(x_1n) + g_mu[i], K=K, nu=g_nu[i])

        probs_i1.append(pp_joint_n2 / pp_joint_n1)

    print('step', j - 1)
    print('prob', np.log(np.prod(probs_i1)))
    assert np.isclose(np.log(np.prod(probs_i1)), probs_out[j - 1][0], atol=1e-3, rtol=1e-03)
    print('x_step', x1[j - 1, :])
    print('---------------')

# check with joint probability
print('\n ********* JOINT PROBABILITY CHECK GAUSS********')
for j in range(2, seq_len + 1):
    probs_i1 = []
    for i in range(p):
        pp = utils_stats.gaussian_predictive_theoretical_prob_slow(x1[:j - 1, i], x1[:j, i], g_mu[i], g_cov[i],
                                                                   g_var[i])
        probs_i1.append(pp)

    print('step', j - 1)
    print('prob', np.log(np.prod(probs_i1)))
    assert np.isclose(np.log(np.prod(probs_i1)), probs_out_gauss[j - 1][0], atol=1e-3, rtol=1e-03)
    print('x_step', x1[j - 1, :])
    print('---------------')


def student_pdf_1d(X, mu, var, nu):
    num = math.gamma((1. + nu) / 2.) * pow(
        1. + (1. / (nu - 2)) * (1. / var * (X - mu) ** 2), -(1. + nu) / 2.)
    denom = math.gamma(nu / 2.) * np.sqrt((nu - 2) * math.pi * var)
    return num / denom


def gauss_pdf_1d(X, mu, var):
    return 1. / np.sqrt(2. * np.pi * var) * np.exp(- (X - mu) ** 2 / (2. * var))


print(student_pdf_1d(-0.35260147, g_mu[0], g_var[0], g_nu[0]))
print(gauss_pdf_1d(-0.35260147, g_mu[0], g_var[0]))
print(sum(probs_out), sum(probs_out_gauss))
