import torch
import numpy as np
from scipy import linalg


def convert_type(features, cuda):
    if cuda and (type(features) == np.ndarray):
        features = torch.from_numpy(features).cuda()
    elif (not cuda) and (type(features) != np.ndarray):
        features = features.cpu().numpy()
    elif cuda and (features.device.type == 'cpu'):
        features = features.cuda()
    return features


def calculate_fid(ref_features, gen_features, cuda=True, eps=1e-6):
    ref_features = convert_type(ref_features, cuda)
    gen_features = convert_type(gen_features, cuda)

    m_ref, s_ref = get_embedding_statistics(ref_features, cuda)
    m_gen, s_gen = get_embedding_statistics(gen_features, cuda)
    fid = calculate_fd(m_ref, s_ref, m_gen, s_gen, cuda=cuda, eps=eps)
    return fid


def get_embedding_statistics(embeddings, cuda=False):
    if cuda:
        embeddings = embeddings.double()  # More precision = more stable
        mu = torch.mean(embeddings, 0)
        sigma = torch_cov(embeddings, rowvar=False)
    else:
        mu = np.mean(embeddings, axis=0)
        sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def calculate_fd(mu1, sigma1, mu2, sigma2, cuda=False, eps=1e-6):
    if cuda:
        fid = torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
        fid = fid.cpu().numpy()
    else:
        fid = numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
    return fid


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                    Each row of `m` represents a variable, and each column a single
                    observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                    variable, with observations in the columns. Otherwise, the
                    relationship is transposed: each column represents a variable,
                    while the rows contain observations.

    Returns:
            The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                         inception net (like returned by the function 'get_predictions')
                         for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                         representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                         representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; ' 'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            print('Imaginary component of {}, may affect results'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


# PyTorch implementation of Frechet distance, from Andrew Brock (modified slightly)
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Pytorch implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
             representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
             representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    # Using double precision instead of float seems to make the GPU FD more stable
    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()

    # Add a tiny offset to the covariance matrices to make covmean estimate more stable
    # Will change the output by a couple decimal places compared to not doing this
    offset = torch.eye(sigma1.size(0)).cuda().double() * eps

    sigma1, sigma2 = sigma1 + offset, sigma2 + offset

    diff = mu1 - mu2

    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    tr_covmean = torch.trace(covmean)

    out = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return out
