"""
Copyright Systems & Technology Research

THis module contains several utility functions that compute parameters for the bound function
"""
import numpy as np
import math
from scipy.stats import multivariate_normal as mvnorm
import scipy.optimize as opt
from scipy.sparse import diags
import cupy as cp
from bayesiancoresets.util.norm_calc import NormCalc
import logging

log = logging.getLogger("str_lwll")

# Calculated ||l(w)||, the norm of sum(w_k * l_k)
# likelihood: The matrix of finite projection of likelihood functions
# w: The combination of likelihood functions we take
def normLike(likelihood, w):
    l = likelihood.dot_product(w)
    return math.sqrt(np.mean(l ** 2))


# Calculates eta for the coreset bound
# likelihood: The matrix of finite projection of likelihood functions
# sigma: the vector of magnitutes of likelihood functions
def paramEta(likelihood, p, sigma):
    l = normLike(likelihood, p)
    eta = 1 - (l * l / (sigma * sigma))
    print(eta)
    return math.sqrt(eta)


# Calculates eta bar for the coreset bound
# likelihood: The matrix of finite projection of likelihood functions
# sigman: the vector of magnitutes of likelihood functions
def paramEtaBar(likelihood, sigman, p):
    etaBar = 0
    for i in range(len(sigman)):
        if i % 100 == 0:
            print(i)
        for j in range(i + 1, len(sigman)):
            w = np.zeros(len(sigman))
            w[i] = p[i] / sigman[i]
            w[j] = -1 * p[j] / sigman[j]
            temp = normLike(likelihood, w)
            if temp > etaBar:
                etaBar = temp
    print(etaBar)
    return math.sqrt(etaBar)


# Takes an N-dimensional vector x and finds the
# closest point on the standard (N-1)-simplex
# When x is a KxN matrix, it returns a
# KxN matrix of projections
def simplexProj(x):
    n, d = x.shape
    y = -np.sort(-x, axis=1)
    s = np.diag(1 / np.arange(1, d + 1))
    s = np.matmul(np.cumsum(y, axis=1) - 1, s)
    col = np.sum(y > s, axis=1) - 1
    row = np.arange(0, n)
    scale = np.zeros(n)
    scale = s[row, col]
    s = np.subtract(x.T, scale).T
    s = np.clip(s, 0, 1)
    return s


# Calculates nu for the coreset bound
# p: The optimal probability vecot
# sigman: the vecotr of likelihood magnitutes
# etaBar: the value of etaBar
def paramNu(p, sigman, etaBar):
    sigma = np.sum(sigman)
    scale = diags(sigman) / sigma
    x = scale.dot(np.ones(p.shape))
    n = len(sigman)
    opt = np.zeros((n, n - 1))
    perp = np.zeros(n)
    for i in range(n):
        opt[i, 0:i] = x[0:i]
        opt[i, i:n] = x[(i + 1) : n]
        perp[i] = p[i]
    proj = simplexProj(opt)
    scale = diags(sigma / sigman)
    opt = scale.dot(opt)
    proj = scale.dot(proj)
    dist = (opt - proj) ** 2
    r = np.sqrt(np.sum(dist, axis=1) + perp ** 2)
    dist = np.amin(r)
    nu = dist * dist / (sigma * sigma * etaBar * etaBar)
    nu = 1 - nu
    return math.sqrt(nu)


# Get all parameters for the bound
# likelihood: THe matrix of random vectors
# p: The probability vector
def get_bound_params(likelihood, p):
    sigman = np.sqrt(likelihood.get_norm(p))
    sigma = sum(sigman)
    xi = (np.amax(sigman) - np.amin(sigman)) / 2
    log.info("xi: {xi}")

    eta = paramEta(likelihood, p, sigma)
    log.info(f"Eta; {eta}")
    if len(p) < 3000:
        etaBar = paramEtaBar(likelihood, sigman, p)
    else:
        etaBar = math.sqrt(2)
    log.info(f"Eta bar: {etaBar}")
    nu = paramNu(p, sigman, etaBar)
    log.info(f"Nu: {nu}")
    return [sigma, eta, etaBar, nu, xi]


# calculate L-finfinity norm between two Gaussian pdfs
# m_prior: mean of prior distribution
# mu_post: mean of posterior distribution
# cov_prior: covariance matrix of prior distribution
# cov_post: covariance matrix of posterior distribution
def inf_norm(mu_prior, mu_post, cov_prior, cov_post):
    ratio = (
        lambda x: -1
        * mvnorm.pdf(x, mu_post, cov_post)
        / mvnorm.pdf(x, mu_prior, cov_prior)
    )
    print(ratio(mu_post))
    mn = mu_prior - 5 * np.diag(cov_prior)
    mx = mu_prior + 5 * np.diag(cov_prior)
    init = (mu_prior + mu_post) / 2
    if np.array_equal(mu_prior, mu_post):
        init = init + 0.1
    bounds = opt.Bounds(mn, mx)
    res = opt.minimize(ratio, init, bounds=bounds)
    return abs(res.fun)


# calculate L-2 norm between two Gaussian pdfs
# m_prior: mean of prior distribution
# mu_post: mean of posterior distribution
# cov_prior: covariance matrix of prior distribution
# cov_post: covariance matrix of posterior distribution
def l2_norm(mu_prior, mu_post, cov_prior, cov_post):
    th_samp = np.random.multivariate_normal(mu_prior, cov_prior, 9999999)
    y = mvnorm.pdf(th_samp, mu_post, cov_post) / mvnorm.pdf(
        th_samp, mu_prior, cov_prior
    )
    norm = math.sqrt(np.sum(y * y))
    return norm


# Calculas an upper bound on the L2 norm for the neural netowkr
# This is only valid for the modified laplace approximation posterior used in the derandomized PAC-Bayes bounds paper
# mu_prior: THe mean of the prior distribution
# mu_post: The mean of the posterior distribution
# stdev_prior: standard deviation of the prior distribution
# stdev_post: standard deviation of the posterior distribution
# returns an upper bound on the L2 norm of nu/pi_0
def l2_norm_neural_net(
    mu_prior, mu_post, stdev_prior, stdev_post,
):
    kl = (
        np.sum(np.log((stdev_prior / stdev_post) ** 2))
        + np.sum((mu_prior - mu_post) ** 2)
    ) / 2
    return math.exp(kl)
