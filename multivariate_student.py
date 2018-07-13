import math
import numpy as np
from numpy.linalg import slogdet
import scipy.stats
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class multivariate_student(object):
    @staticmethod
    def sample(phi, K, nu, samples=None, rng=None):
        """
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        N = degrees of freedom
        M = # of samples to produce
        """
        rng = np.random if rng is None else rng
        if samples is None:
            return multivariate_student.sample(phi, K, nu, samples=1, rng=rng)[0]
        d = len(K)
        g = np.tile(rng.gamma(nu / 2., 2. / nu, samples), (d, 1)).T
        Z = rng.multivariate_normal(np.zeros(d), K * (nu - 2.) / nu, samples)
        return phi + Z / np.sqrt(g)

    @staticmethod
    def sample_1d(phi, K, nu, n_samples=None, rng=None):
        """
        Output:
        Produce n_samples samples of 1D student t distribution
        Input:
        mu = mean
        K = scaling parameter
        nu = degrees of freedom
        n_samples = # of samples to produce
        """
        sigma = np.sqrt(K * (nu - 2.) / nu)
        rng = np.random if rng is None else rng
        if n_samples is None:
            return multivariate_student.sample_1d(phi, K, nu, n_samples=1, rng=rng)[0]

        nu = int(nu)
        x_norm = rng.normal(loc=0, scale=1., size=n_samples * (nu + 1))
        num = x_norm[-n_samples:]
        denom = np.reshape(x_norm[:n_samples * nu], (n_samples, nu)) ** 2
        denom = np.sum(denom, axis=1)
        denom = np.sqrt(1. / nu * denom)
        samples = phi + sigma * num / denom
        return samples

    @staticmethod
    def sample_1d_bailey(phi, K, nu, n_samples=None, rng=None):
        """
        https://www.researchgate.net/profile/William_Shaw9/publication/247441442_Sampling_Student%27%27s_T_distribution-use_of_the_inverse_cumulative_distribution_function/links/55bbbc7908ae9289a09574f6/Sampling-Students-T-distribution-use-of-the-inverse-cumulative-distribution-function.pdf
        Output:
        Produce n_samples samples of 1D student t distribution
        Input:
        mu = mean
        K = scaling parameter
        nu = degrees of freedom
        n_samples = # of samples to produce
        """
        sigma = np.sqrt(K * (nu - 2.) / nu)
        rng = np.random if rng is None else rng
        samples = []
        for _ in range(n_samples):
            w = 2.
            while w > 1.:
                u = 2 * rng.uniform() - 1
                v = 2 * rng.uniform() - 1
                w = u ** 2 + v ** 2
            t = u * np.sqrt(nu * (np.power(w, -2. / nu) - 1) / w)
            s = phi + sigma * t
            samples.append(s)
        return np.array(np.squeeze(samples))[None, :]

    @staticmethod
    def sample_1d_bailey_v2(phi, K, nu, n_samples=None, rng=None):
        """
        https://www.researchgate.net/profile/William_Shaw9/publication/247441442_Sampling_Student%27%27s_T_distribution-use_of_the_inverse_cumulative_distribution_function/links/55bbbc7908ae9289a09574f6/Sampling-Students-T-distribution-use-of-the-inverse-cumulative-distribution-function.pdf
        https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/5838991#5838991

        Output:
        Produce n_samples samples of 1D student t distribution
        Input:
        mu = mean
        K = scaling parameter
        nu = degrees of freedom
        n_samples = # of samples to produce
        """
        rng = np.random if rng is None else rng
        samples = []
        sigma = np.sqrt(K * (nu - 2.) / nu)
        for _ in range(n_samples):
            r1 = rng.uniform()
            r2 = rng.uniform()
            a = min(r1, r2)
            b = max(r1, r2)
            u = b * np.cos(2 * np.pi * a / b)
            v = b * np.sin(2 * np.pi * a / b)
            w = u ** 2 + v ** 2
            t = u * np.sqrt(nu * (np.power(w, -2. / nu) - 1) / w)
            s = phi + sigma * t
            samples.append(s)
        return np.array(np.squeeze(samples))[None, :]

    @staticmethod
    def pdf(X, phi, K, nu):
        """
        Multivariate t-student density:
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

    @staticmethod
    def pdf_1d(X, phi, K, nu):
        """
        Univariate t-student density:
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

    @staticmethod
    def log_pdf(X, phi, K, nu):
        """
        Multivariate t-student log density:
        output:
            the log density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            K = scale matrix (dxd numpy array)
            nu = degrees of freedom
        """

        d = len(X)
        sign, logdet = slogdet(K)

        logp = math.lgamma((d + nu) / 2.) - 0.5 * (d + nu) * np.log(
            1. + (1. / (nu - 2)) * np.dot(np.dot((X - phi), np.linalg.inv(K)), (X - phi)))
        logz = math.lgamma(nu / 2.) + d / 2. * np.log((nu - 2) * math.pi) + 0.5 * logdet

        return logp - logz


def test_1d():
    nu = 34
    mu = 0
    K = np.eye(1) * 1.
    n_samples = 10000

    plt.figure()
    x_range = np.linspace(-5., 5., 1000)
    y = multivariate_student.pdf_1d(x_range, mu, K, nu)[0]

    integral = 0.
    for i in range(len(y)):
        integral += y[i] * (x_range[1] - x_range[0])
    print('integral', integral)

    X = multivariate_student.sample_1d(mu, K, nu, n_samples=n_samples).T
    plt.hist(X, 100, normed=True)
    plt.plot(x_range, y, 'r')
    plt.savefig('/mnt/storage/users/ikorshun/exch-rnn/samples/debug'
                + '/student_t.png', format='png')

    plt.figure()
    X = multivariate_student.sample_1d_bailey(mu, K, nu, n_samples=n_samples).T
    plt.hist(X, 100, normed=True)
    plt.plot(x_range, y, 'r')
    plt.savefig('/mnt/storage/users/ikorshun/exch-rnn/samples/debug'
                + '/student_t_bailey.png', format='png')

    plt.figure()
    X = multivariate_student.sample_1d_bailey_v2(mu, K, nu, n_samples=n_samples).T
    plt.hist(X, 100, normed=True)
    plt.plot(x_range, y, 'r')
    plt.savefig('/mnt/storage/users/ikorshun/exch-rnn/samples/debug'
                + '/student_t_bailey_v2.png', format='png')
