import torch
import math


def gauss_density_1d(x, mean=0, std=1, log=True):
    """Gaussian density on R^1"""
    log_prob = -0.5 * (((x - mean) / std) ** 2 +
                       math.log(2 * math.pi) +
                       2 * torch.log(std))
    if log:
        return log_prob
    return torch.exp(log_prob)


def gmm_density_1d(x, means=None, stds=None, weights=None, log=True):
    """GMM Density on R^1"""
    if means is None:
        means = torch.Tensor([0., 0.])
    if stds is None:
        stds = torch.Tensor([1., 1.])
    if weights is None:
        weights = torch.Tensor([1 / 2, 1 / 2])

    log_probs = -0.5 * (((x[:, None] - means[None, :]) / stds[None, :]) ** 2 +
                        math.log(2 * math.pi) +
                        torch.log(stds)[None, :]) + torch.log(weights)
    log_probs = torch.logsumexp(log_probs, dim=1)
    if log:
        return log_probs
    return torch.exp(log_probs)


def __full_gauss_density_nd(x, mean=None, cov=None, log=True):
    """Gaussian density on R^n with full covariances"""
    if mean is None:
        mean = torch.zeros(x.shape[1])

    if cov is None:
        cov = torch.diag(torch.ones(x.shape[1]))

    log_prob = -0.5 * ((x - mean) @ torch.linalg.inv(cov) @ (x - mean).T +
                       torch.log(torch.linalg.det(cov)) +
                       x.shape[1] * math.log(2 * math.pi))

    if log:
        return log_prob
    return torch.exp(log_prob)


def __diag_gauss_density_nd(x, mean, cov, log=True):
    """Gaussian density on R^n with diagonal covariances"""
    d = x.shape[1]
    std = torch.sqrt(cov)
    z = (x - mean) / std
    log_prob = -0.5 * (
        d * math.log(2 * math.pi) +
        2 * torch.log(std).sum() +
        torch.linalg.norm(z, dim=1) ** 2
    )

    if log:
        return log_prob
    return log_prob.exp()


def __full_gmm_density_nd(x, means=None, covs=None, weights=None, log=True):
    """GMM density on R^n with full covariances"""
    if weights is None:
        weights = torch.ones(2) / 2

    if means is None:
        means = torch.zeros(2, x.shape[1])

    if covs is None:
        covs = torch.stack([torch.eye(x.shape[1]) for _ in range(2)])

    inv_cov = torch.linalg.inv(covs)
    dist = torch.einsum('knd,kdD,knD->kn',
                        x.unsqueeze(0) - means.unsqueeze(1),
                        inv_cov,
                        x.unsqueeze(0) - means.unsqueeze(1))
    log_probs = (-0.5 * (dist +
                         x.shape[1] * math.log(2 * math.pi) +
                         torch.log(torch.linalg.det(covs)).unsqueeze(1)) +
                 torch.log(weights).unsqueeze(1))
    log_probs = torch.logsumexp(log_probs, dim=0)
    if log:
        return log_probs
    return torch.exp(log_probs)


def __diag_gmm_density_nd(x, means, covs, weights, log=True, lse=True):
    """GMM density on R^n with diagonal covariances"""
    d = x.shape[1]
    stds = torch.sqrt(covs)
    residuals = (x[None, :] - means[:, None]) / stds[:, None]
    log_probs = (torch.log(weights)[:, None] - 0.5 * (
        d * math.log(2 * math.pi) +
        2 * torch.log(stds).sum(dim=1)[:, None] +
        (residuals ** 2).sum(dim=-1)
    ))
    if lse:
        log_probs = torch.logsumexp(log_probs, dim=0)
    if log:
        return log_probs
    return torch.exp(log_probs)


def gauss_density_nd(x, mean=None, cov=None, log=True, cov_type='diag'):
    """Gaussian density on R^n"""
    if cov_type == 'full':
        return __full_gauss_density_nd(x, mean, cov, log=log)
    elif cov_type == 'diag':
        return __diag_gauss_density_nd(x, mean, cov, log=log)


def gmm_density_nd(x,
                   means=None,
                   covs=None,
                   weights=None,
                   log=True,
                   cov_type='diag'):
    """GMM density on R^n"""
    if cov_type == 'full':
        return __full_gmm_density_nd(x, means, covs, weights, log=log)
    elif cov_type == 'diag':
        return __diag_gmm_density_nd(x, means, covs, weights, log=log)


def diag_gmm_log_probs(X, weights, means, stds):
    """Computes the log-probabilities for a GMM with diagonal covariances"""
    n_dim = X.shape[1]
    log_probs = (torch.log(weights)[:, None] - 0.5 * (
        n_dim * math.log(2 * math.pi) +
        2 * torch.log(stds).sum(dim=1)[:, None] + (
            (1 / stds ** 2) @ X.T ** 2 +
            torch.sum((means / stds) ** 2, 1)[:, None] -
            2 * ((means / (stds ** 2)) @ X.T)
        )
    ))

    return log_probs


def diag_gauss_log_probs(X, mean, std):
    d = X.shape[1]
    z = (X - mean) / std
    log_prob = - 0.5 * (
        d * math.log(2 * math.pi) +
        2 * torch.log(std).sum() +
        torch.linalg.norm(z, dim=1) ** 2
    )
    return log_prob
