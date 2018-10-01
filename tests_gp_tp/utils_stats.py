import math

import numpy as np
import scipy
import scipy.special
import scipy.stats

rng = np.random.RandomState(42)


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

        prob = mvt_pdf(x_next, phi=mu_a_given_b, K=cov_a_given_b, nu=nu_a_given_b)
        prob_1d = student_pdf_1d(x_next, phi=mu_a_given_b, K=cov_a_given_b, nu=nu_a_given_b)

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
                mvt_pdf(x_next[i:i + 1], phi=mu_a_given_b, K=cov_a_given_b, nu=nu_a_given_b))
        prob = np.prod(probs_i)
    return prob


def mvt_pdf(X, phi, K, nu):
    """
    Multivariate student-t density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        K = scale matrix (dxd numpy array)
        nu = degrees of freedom
    """
    d = X.shape[-1]
    num = math.gamma((d + nu) / 2.) * pow(
        1. + (1. / (nu - 2)) * ((X - phi).dot(np.linalg.inv(K)).dot(np.transpose(X - phi))), -(d + nu) / 2.)
    denom = math.gamma(nu / 2.) * pow((nu - 2) * math.pi, d / 2.) * pow(np.linalg.det(K), 0.5)
    return num / denom


def student_pdf_1d(X, phi, K, nu):
    """
    Univariate student-t density:
    output:
        the density of the given element
    input:
        x = parameter scalar
        mu = mean
        var = scale matrix
        nu = degrees of freedom
    """
    num = math.gamma((1. + nu) / 2.) * pow(
        1. + (1. / (nu - 2)) * (1. / K * (X - phi) ** 2), -(1. + nu) / 2.)
    denom = math.gamma(nu / 2.) * pow((nu - 2) * math.pi * K, 0.5)
    return num / denom
