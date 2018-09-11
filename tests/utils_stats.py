import numpy as np
import scipy
import scipy.special
import scipy.stats

from tests.multivariate_student import multivariate_student

rng = np.random.RandomState(42)


def normalize_logprobs(logprobs):
    logprobs = np.array(logprobs)
    logprobs -= np.max(logprobs)
    p = np.exp(logprobs)
    p /= np.sum(p)
    return p


def logit(x):
    return np.log(x / (1. - x))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def inv_softplus(x):
    return np.log(np.exp(x) - 1.)


def inv_sigmoid(x):
    """
    ==logit
    """
    return np.log(x) - np.log(1. - x)


def sample_binary_exchangeable_sequence(p_dist, p_dist_params, size):
    """
    Sample exchangeable sequence {x_1,..,x_n} by
    1) sampling Bernoulli parameter theta from p_dist, which has parameters p_dist_params
    2) sampling x_i from Bernoulli(theta)
    """
    if p_dist is None:
        theta = p_dist_params
        seq_x = list(scipy.stats.bernoulli.rvs(p=theta, size=size))
    else:
        theta = p_dist.rvs(size=1, **p_dist_params)
        seq_x = scipy.stats.bernoulli.rvs(p=theta, size=size)
    return seq_x


def sample_categorical_exchangeable_sequence(p_dist, p_dist_params, size, return_theta=True):
    """
    Sample exchangeable sequence {x_1,..,x_n} by
    1) sampling probability vector theta from p_dist, which has parameters p_dist_params
    2) sampling x_i from Categorical(theta)
    """
    theta = p_dist.rvs(size=1, **p_dist_params)[0]
    ndim = len(theta)
    seq_x_1hot = rng.multinomial(n=1, pvals=theta, size=size)
    seq_x = list(np.dot(seq_x_1hot, np.arange(ndim)))
    if return_theta:
        return seq_x, theta
    else:
        return seq_x


def sample_gaussian_exchangeable_sequence(p_dist_params, size, return_theta=False):
    theta = rng.normal(size=1, **p_dist_params)[0]
    # print 'true mean', theta
    std = p_dist_params['scale']
    # print 'true variance', std ** 2
    var_x = 1. - std ** 2
    std_x = np.sqrt(var_x)
    seq_x = list(rng.normal(loc=theta, scale=std_x, size=size))
    # print 'seq mean', np.mean(seq_x)
    # print 'seq variance', np.var(seq_x), 1. - std ** 2
    if return_theta:
        return seq_x, theta
    else:
        return seq_x


def sample_multivar_normal_exchangeable_sequence(p_dist_params, size):
    mean_theta, cov_theta = p_dist_params['mean'], p_dist_params['cov']
    n = len(mean_theta)
    theta = rng.multivariate_normal(size=1, mean=mean_theta, cov=cov_theta)[0]
    seq_x = list(rng.multivariate_normal(mean=theta, cov=np.eye(n), size=size))
    return seq_x


def beta_bernoulli_loglikelihood(sequence, a, b):
    n = len(sequence)
    k = sum(sequence)
    return np.log(scipy.special.beta(k + a, n - k + b) / scipy.special.beta(a, b))


def gaussian_predictive_theoretical_prob_slow(x_1n, x_1n1, mu, cov, var):
    n = len(x_1n)
    cov_matrix_x1n = np.ones((n, n)) * cov
    cov_matrix_x1n[np.diag_indices(n)] = var
    pdf_x_1n = scipy.stats.multivariate_normal.pdf(x_1n, mean=np.ones(n) * mu, cov=cov_matrix_x1n)

    cov_matrix_x1n1 = np.ones((n + 1, n + 1)) * cov
    cov_matrix_x1n1[np.diag_indices(n + 1)] = var
    pdf_x_1n1 = scipy.stats.multivariate_normal.pdf(x_1n1, mean=np.ones(n + 1) * mu, cov=cov_matrix_x1n1)
    return pdf_x_1n1 / pdf_x_1n


def predictive_mdim_gaussian_dist(X, cov_u, cov_v, var_u, var_v, mu, recursive=False):
    n = X.shape[0]
    p = X.shape[1]

    mu_a = np.ones((p,)) * mu

    # U is an (n x n) convariance matrix (intersamples)
    # V is a (p x p) convariance matrix (insample)
    v = np.ones((p, p)) * cov_v
    np.fill_diagonal(v, var_v)

    if recursive:
        print('using recursive implementation')
        mu_a_given_b = mu_a.reshape((1, p))
        cov_a_given_b = var_u * v
        ident_m = np.eye(p)
        for i in range(1, n + 1):
            dd_i = cov_u / (var_u + cov_u * (i - 1)) * ident_m
            mu_a_given_b = mu_a_given_b.dot(ident_m - dd_i) + X[i - 1, :].dot(dd_i)
            cov_a_given_b = cov_a_given_b.dot(ident_m - dd_i) + (var_u - cov_u) * v.dot(dd_i)
        mu_a_given_b = mu_a_given_b.flatten()
    else:
        dd = cov_u / (var_u + cov_u * (n - 1)) * np.eye(p)
        mu_a_given_b = mu_a + dd.dot(np.sum(X - mu, axis=0))
        cov_a_given_b = var_u * v - n * dd.dot(cov_u * v)
    return mu_a_given_b, cov_a_given_b


def predictive_mdim_gaussian_prob(x_1n, x_next, cov_u, cov_v, var_u, var_v, mu, recursive=False):
    mu_a_given_b, cov_a_given_b = predictive_mdim_gaussian_dist(x_1n, cov_u, cov_v, var_u, var_v, mu,
                                                                recursive=recursive)
    print(mu_a_given_b, cov_a_given_b)
    prob = scipy.stats.multivariate_normal.pdf(x_next, mean=mu_a_given_b, cov=cov_a_given_b)
    return prob


def get_likelihood_under_gaussian(x, mu, sigma):
    return scipy.stats.multivariate_normal.logpdf(x, mean=mu, cov=sigma)


# def predictive_mdim_student_dist(X, nu, cov_u, cov_v, var_u, var_v, mu, recursive=False):
#     n = X.shape[0]
#     p = X.shape[1]
#
#     mu_a = np.ones((p,)) * mu
#
#     # U is an (n x n) convariance matrix (intersamples)
#     u = np.ones((n, n)) * cov_u
#     np.fill_diagonal(u, var_u)
#
#     # V is a (p x p) convariance matrix (insample)
#     v = np.ones((p, p)) * cov_v
#     np.fill_diagonal(v, var_v)
#     dd = cov_u / (var_u + cov_u * (n - 1)) * np.eye(p)
#     n_b = n * p
#     X_zm = X - mu
#     A = (cov_u * (n - 2) + var_u) / ((var_u - cov_u) * (cov_u * (n - 1) + var_u)) * np.linalg.inv(v)
#     B = -1 * cov_u / ((var_u - cov_u) * (cov_u * (n - 1) + var_u)) * np.linalg.inv(v)
#
#     if recursive:
#         print 'recursive implementation'
#         nu_a_given_b = nu
#         mu_a_given_b = mu_a.reshape((1, p))
#         beta = 0
#         K_aa = var_u * v
#         ident_m = np.eye(p)
#         xb = np.zeros((1, p))
#         for i in range(1, n + 1):
#
#             # A = (cov_u * (i - 2) + var_u) / ((var_u - cov_u) * (cov_u * (i - 1) + var_u)) * np.linalg.inv(v)
#             # B = -1 * cov_u / ((var_u - cov_u) * (cov_u * (i - 1) + var_u)) * np.linalg.inv(v)
#             nu_a_given_b += p
#             dd_i = cov_u / (var_u + cov_u * (i - 1)) * ident_m
#             mu_a_given_b = mu_a_given_b.dot(ident_m - dd_i) + X[i - 1, :].dot(dd_i)
#             beta += (X_zm[i - 1:i].dot(A) + 2 * xb).dot(np.transpose(X_zm[i - 1:i]))
#             xb += X_zm[i - 1:i].dot(B)
#             K_aa = K_aa.dot(ident_m - dd_i) + (var_u - cov_u) * v.dot(dd_i)
#             cov_a_given_b = (nu + beta - 2.) / (nu_a_given_b - 2.) * K_aa
#         mu_a_given_b = mu_a_given_b.flatten()
#     else:
#         nu_a_given_b = nu + n_b
#         mu_a_given_b = mu_a + dd.dot(np.sum(X - mu, axis=0))
#         beta = np.sum(np.diagonal(X_zm.dot(A).dot(X_zm.transpose()))) \
#                + np.sum(X_zm.dot(B).dot(X_zm.transpose())) - \
#                np.sum(np.diagonal(X_zm.dot(B).dot(X_zm.transpose())))
#         cov_a_given_b = (nu + beta - 2.) / (nu + n_b - 2.) * (var_u * v - n * dd.dot(cov_u * v))
#     return nu_a_given_b, mu_a_given_b, cov_a_given_b


def predictive_1d_student_dist(X, nu, cov, var, mu, recursive=False):
    n = X.shape[0]
    p = X.shape[1]
    assert p == 1

    mu_a = np.ones((p,)) * mu
    X_zm = X - mu

    if recursive:
        # print 'recursive implementation'
        if p > 1:
            raise NotImplementedError('recursive implementation works only for 1D case')
        nu_a_given_b = nu
        mu_a_given_b = mu_a.reshape((1, p))
        beta = 0
        K_aa = var
        x_cum_sum = 0
        for i in range(1, n + 1):
            nu_a_given_b += p
            dd_i = cov / (var + cov * (i - 1))
            mu_a_given_b = (1. - dd_i) * mu_a_given_b + X[i - 1] * dd_i

            a_i = (cov * (i - 2) + var) / ((var - cov) * (cov * (i - 1) + var))
            b_i = -1. * cov / ((var - cov) * (cov * (i - 1) + var))
            b_i_prev = -1. * cov / ((var - cov) * (cov * (i - 2) + var))
            beta += (a_i - b_i) * X_zm[i - 1] ** 2 \
                    + b_i * (x_cum_sum + X_zm[i - 1]) ** 2 \
                    - b_i_prev * x_cum_sum ** 2
            x_cum_sum += X_zm[i - 1]
            K_aa = (1 - dd_i) * K_aa + (var - cov) * dd_i
            cov_a_given_b = (nu + beta - 2.) / (nu_a_given_b - 2.) * K_aa
        cov_a_given_b = np.array(cov_a_given_b).reshape((1, 1))
        mu_a_given_b = mu_a_given_b.flatten()
    else:
        v = np.eye(p)
        n_b = n * p
        dd = cov / (var + cov * (n - 1)) * np.eye(p)
        A = (cov * (n - 2) + var) / ((var - cov) * (cov * (n - 1) + var)) * np.linalg.inv(v)
        B = -1 * cov / ((var - cov) * (cov * (n - 1) + var)) * np.linalg.inv(v)
        nu_a_given_b = nu + n_b
        mu_a_given_b = mu_a + dd.dot(np.sum(X - mu, axis=0))
        beta = np.sum(np.diagonal(X_zm.dot(A).dot(X_zm.transpose()))) \
               + np.sum(X_zm.dot(B).dot(X_zm.transpose())) - \
               np.sum(np.diagonal(X_zm.dot(B).dot(X_zm.transpose())))
        cov_a_given_b = (nu + beta - 2.) / (nu + n_b - 2.) * (var * v - n * dd.dot(cov * v))
    return nu_a_given_b, mu_a_given_b, cov_a_given_b


def predictive_mdim_student_prob(x_1n, x_next, cov_u, var_u, mu, nu, recursive=False, verbose=False):
    p = x_1n.shape[-1]
    if p == 1:
        nu_a_given_b, mu_a_given_b, cov_a_given_b = predictive_1d_student_dist(x_1n,
                                                                               nu=nu[0],
                                                                               cov=cov_u[0],
                                                                               var=var_u[0],
                                                                               mu=mu[0], recursive=recursive)
        if verbose:
            print(nu_a_given_b)
            print(mu_a_given_b)
            print(cov_a_given_b)

        prob = multivariate_student.pdf(x_next, phi=mu_a_given_b, K=cov_a_given_b, nu=nu_a_given_b)
        prob_1d = multivariate_student.pdf_1d(x_next, phi=mu_a_given_b, K=cov_a_given_b, nu=nu_a_given_b)

        if verbose:
            print('****', prob, prob_1d)

    else:
        probs_i = []
        for i in range(p):
            nu_a_given_b, mu_a_given_b, cov_a_given_b = predictive_1d_student_dist(x_1n[:, i:i + 1],
                                                                                   nu=nu[i],
                                                                                   cov=cov_u[i],
                                                                                   var=var_u[i],
                                                                                   mu=mu[i],
                                                                                   recursive=recursive)
            if verbose:
                print(nu_a_given_b)
                print(mu_a_given_b)
                print(cov_a_given_b)

            probs_i.append(
                multivariate_student.pdf(x_next[i:i + 1], phi=mu_a_given_b, K=cov_a_given_b, nu=nu_a_given_b))
        prob = np.prod(probs_i)
    return prob


def get_student_logprob(x, nu, mu, var, epsilon=1e-12):
    """
    Returns log probability per each example in the batch
    :param self:
    :param x: (batch_size, ndim)
    :param nu: (ndim,)
    :param mu: (ndim,)
    :param var: (ndim,)
    :param epsilon:
    :return:
    """
    gamma_quotient = np.exp(scipy.special.gammaln((1. + nu) / 2.) - scipy.special.gammaln(nu / 2.))
    num = gamma_quotient * np.power(1. + (1. / (nu - 2)) * (1. / var * (x - mu) ** 2),
                                    -(1. + nu) / 2.)
    denom = np.power((nu - 2) * np.pi * var, 0.5)
    log_pdf = np.sum(np.log(num / denom + epsilon))
    return log_pdf
