import scipy
import numpy as np
from typing import Tuple, Union
import torch
import numpy as np
from scipy.linalg import sqrtm

def kl_mvn_reduced(to: Tuple[np.ndarray, np.ndarray],
                   fr: Tuple[np.ndarray, np.ndarray]) -> float:
    '''
    # mu_to, cov_to are the mean and covariance matrix of to (distribution P).
    # mu_fr, cov_fr are the mean and covariance matrix of fr (distribution Q).
    :param to: Tuple((d,), (d,d))
    :param fr: Tuple((d,), (d,d))
    :return:
    '''
    mu_to, cov_to = to
    mu_fr, cov_fr = fr
    assert mu_to.shape == mu_fr.shape
    assert cov_to.shape == cov_fr.shape

    # Compute the difference in means
    mu_d = mu_fr - mu_to

    # Extract the diagonal elements (variances)
    var_to = np.diag(cov_to)
    var_fr = np.diag(cov_fr)

    # KL divergence formula for diagonal covariance matrices
    term1 = np.sum(var_fr / var_to)  # Sum of (cov_fr[i,i] / cov_to[i,i])
    term2 = np.sum(np.log(var_to / var_fr))  # Sum of log(cov_to[i,i] / cov_fr[i,i])
    term3 = np.sum(mu_d**2 / var_fr)  # Sum of (mu_fr[i] - mu_to[i])^2 / cov_fr[i,i]

    return 0.5 * (term1 + term2 + term3 - len(mu_d))

def kl_mvn_full(to: Tuple[np.ndarray, np.ndarray],
                fr: Tuple[np.ndarray, np.ndarray]) -> float:
    '''
    # mu_to, cov_to are the mean and covariance matrix of to (distribution P).
    # mu_fr, cov_fr are the mean and covariance matrix of fr (distribution Q).
    :param to: Tuple((d,), (d,d))
    :param fr: Tuple((d,), (d,d))
    :return:
    '''
    mu_to, cov_to = to
    mu_fr, cov_fr = fr
    assert mu_to.shape == mu_fr.shape
    assert cov_to.shape == cov_fr.shape

    mu_d = mu_fr - mu_to

    # Cholesky factorization of of cov_fr
    c, lower = scipy.linalg.cho_factor(cov_fr)

    def solve(B): # this will compute A^(-1)B, where c and lower are Cholesky factorization of A
        return scipy.linalg.cho_solve((c, lower), B)

    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(cov_to)) # tr(cov_fr^(-1)cov_to)
    term2 = logdet(cov_fr) - logdet(cov_to)
    term3 = mu_d.T @ solve(mu_d) # (mu_fr - m_to)^T cov_fr^(-1) (mu_fr - m_to)
    return (term1 + term2 + term3 - len(mu_d)) / 2.

def fit_mu_sigma(X: np.ndarray,
                 ignore_cov: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit multivariate gaussian parameters
    :param X: (n, d)
    :param ignore_cov: whether to assume independence of dimensions
    :return:
    '''
    assert len(X.shape) == 2
    # Calculate the mean vector (mu)
    mu = np.mean(X, axis=0)
    # Calculate the covariance matrix (sigma)
    if ignore_cov:
        sigma_diag = np.var(X, axis=0)
        sigma = np.diag(sigma_diag)
    else:
        sigma = np.cov(X, rowvar=False)  # `rowvar=False` treats columns as features
    return mu, sigma

def kl_div(X: Union[np.ndarray, torch.FloatTensor],
           Y: Union[np.ndarray, torch.FloatTensor],
           ignore_cov: bool=True) -> float:
    '''
    Compute kl divergence between two embedding collections
    :param X:  (n1, d)
    :param Y:  (n2, d)
    :return:
    '''
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    mu1, sigma1 = fit_mu_sigma(X, ignore_cov)
    mu2, sigma2 = fit_mu_sigma(Y, ignore_cov)

    if ignore_cov:
        div = kl_mvn_reduced((mu1, sigma1), (mu2, sigma2))
    else:
        div = kl_mvn_full((mu1, sigma1), (mu2, sigma2))
    return div


def wasserstein_distance(X: Union[np.ndarray, torch.FloatTensor],
                         Y: Union[np.ndarray, torch.FloatTensor],
                         ignore_cov: bool=True) -> float:

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    mu1, sigma1 = fit_mu_sigma(X, ignore_cov)
    mu2, sigma2 = fit_mu_sigma(Y, ignore_cov)

    if ignore_cov:
        mean_diff = mu1 - mu2
        mean_dist = np.sum(mean_diff ** 2)
        var_dist = np.sum(sigma1 ** 2 + sigma2 ** 2 - 2 * sigma1 * sigma2)
        return np.sqrt(mean_dist + var_dist)
    else:
        mean_diff = mu1 - mu2
        mean_dist = np.dot(mean_diff, mean_diff)
        sqrt_sigma1 = sqrtm(sigma1)
        try:
            sqrt_term = sqrtm(np.dot(np.dot(sqrt_sigma1, sigma2), sqrt_sigma1))
            trace_term = np.trace(sigma1 + sigma2 - 2 * sqrt_term)
        except np.linalg.LinAlgError:
            trace_term = np.trace(sigma1 + sigma2)

        return np.sqrt(mean_dist + trace_term)