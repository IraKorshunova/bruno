import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag

import nn_extra_gauss
import nn_extra_student
from tests import utils_stats
from tests.multivariate_student import multivariate_student

rng = np.random.RandomState(41)

batch_size = 2
seq_len = 10
p = 1
cov_u = np.ones((p,), dtype='float32') * 0.001
var_u = np.ones((p,), dtype='float32') * 1.
mu = np.ones((p,), dtype='float32') * 0.
corr_u = cov_u / var_u
nu = 100.
recursive = True


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


K = create_covariance_matrix(seq_len, p, cov_u, var_u)

phi = np.tile(mu, (seq_len, 1)).flatten()
x1 = multivariate_student.sample(phi=phi, K=K, nu=10, rng=rng)
x1 = x1.reshape((seq_len, p))
x1 = np.float32(x1)


x2 = multivariate_student.sample(phi=phi, K=K, nu=3, rng=rng)
x2 = x2.reshape((seq_len, p))
x2 = np.float32(x2)

x = np.concatenate((x1[None, :, :], x2[None, :, :]), axis=0)
print('shape x', x.shape)

nu = np.ones((p,), dtype='float32') * nu
print(nu)
mu = np.ones((p,), dtype='float32') * mu
var_u = np.ones((p,), dtype='float32') * var_u
cov_u = np.ones((p,), dtype='float32') * cov_u
print(nu, var_u, cov_u)
print(x)

x_var = tf.placeholder(tf.float32, shape=(batch_size, seq_len, p))
l_rnn = nn_extra_student.StudentRecurrentLayer(shape=(p,), nu_init=nu, mu_init=mu, var_init=var_u, corr_init=corr_u)
l_rnn2 = nn_extra_gauss.GaussianRecurrentLayer(shape=(p,), mu_init=mu, var_init=var_u, corr_init=corr_u)

probs = []
probs_gauss = []
with tf.variable_scope("one_step") as scope:
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

print('---------------')  # numpy
print('\n ****** NUMPY ******')

probs_k1, probs_k2 = [], []
for k in range(p):
    probs_k1.append(
        multivariate_student.pdf(x1[0, k:k + 1], phi=mu[k:k + 1], K=var_u[None, k:k + 1], nu=nu[k]))
    probs_k2.append(
        multivariate_student.pdf(x2[0, k:k + 1], phi=mu[k:k + 1], K=var_u[None, k:k + 1], nu=nu[k:k + 1]))
prob_1 = np.prod(probs_k1)
prob_2 = np.prod(probs_k2)
print('step', 0)
print('prob', np.log(prob_1), np.log(prob_2))
print('x_step', x1[0, :], x2[0, :])
print('---------------')

for i in range(1, seq_len):
    p_next1 = utils_stats.predictive_mdim_student_prob(x_1n=x1[:i], x_next=x1[i],
                                                       cov_u=cov_u,
                                                       var_u=var_u,
                                                       mu=mu, nu=nu, recursive=recursive,
                                                       verbose=False)
    p_next2 = utils_stats.predictive_mdim_student_prob(x_1n=x2[:i], x_next=x2[i],
                                                       cov_u=cov_u,
                                                       var_u=var_u,
                                                       mu=mu, nu=nu, recursive=recursive,
                                                       verbose=False)
    print('step', i)
    print('prob', np.log(p_next1), np.log(p_next2))
    print('x_step', x1[i, :], x2[i, :])
    print('---------------')

# check with joint probability
print('\n ********* JOINT PROBABILITY CHECK ********')
for j in range(2, seq_len + 1):
    probs_i1, probs_i2 = [], []
    for i in range(p):
        x_2n = x1[:j, i:i + 1].T
        K = create_covariance_matrix(j, 1, cov_u[i], var_u[i])
        pp_joint_n2 = multivariate_student.pdf(x_2n, phi=np.zeros_like(x_2n) + mu[i], K=K, nu=nu[i])

        x_1n = x1[:j - 1, i:i + 1].T
        K = create_covariance_matrix(j - 1, 1, cov_u[i], var_u[i])
        pp_joint_n1 = multivariate_student.pdf(x_1n, phi=np.zeros_like(x_1n) + mu[i], K=K, nu=nu[i])

        probs_i1.append(pp_joint_n2 / pp_joint_n1)

        x_2n = x2[:j, i:i + 1].T
        K = create_covariance_matrix(j, 1, cov_u[i], var_u[i])
        pp_joint_n2 = multivariate_student.pdf(x_2n, phi=np.zeros_like(x_2n) + mu[i], K=K, nu=nu[i])

        x_1n = x2[:j - 1, i:i + 1].T
        K = create_covariance_matrix(j - 1, 1, cov_u[i], var_u[i])
        pp_joint_n1 = multivariate_student.pdf(x_1n, phi=np.zeros_like(x_1n) + mu[i], K=K, nu=nu[i])

        probs_i2.append(pp_joint_n2 / pp_joint_n1)

    print('step', j - 1)
    print('prob', np.log(np.prod(probs_i1)), np.log(np.prod(probs_i2)))
    print('x_step', x1[j - 1, :], x2[j - 1, :])
    print('---------------')

# check with joint probability
print('\n ********* JOINT PROBABILITY CHECK GAUSS********')
for j in range(2, seq_len + 1):
    probs_i1, probs_i2 = [], []
    for i in range(p):
        pp = utils_stats.gaussian_predictive_theoretical_prob_slow(x1[:j - 1, i], x1[:j, i], mu[i], cov_u[i], var_u[i])
        probs_i1.append(pp)
        pp2 = utils_stats.gaussian_predictive_theoretical_prob_slow(x2[:j - 1, i], x2[:j, i], mu[i], cov_u[i], var_u[i])
        probs_i2.append(pp2)

    print('step', j - 1)
    print('prob', np.log(np.prod(probs_i1)), np.log(np.prod(probs_i2)))
    print('x_step', x1[j - 1, :], x2[j - 1, :])
    print('---------------')
