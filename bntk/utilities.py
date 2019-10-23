import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor


def sigmoid_plus(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_minus(z):
    expz = np.exp(z)
    return expz / (1 + expz)


def sigmoid(z):
    # Either of the two sigmoids might overflow but the right one
    # will be selected.
    err_state = np.seterr(all='ignore')
    sigma = np.where(z >= 0, sigmoid_plus(z), sigmoid_minus(z))
    np.seterr(**err_state)
    if np.isinf(sigma).any() or np.isnan(sigma).any():
        print('Sigmoid resulting in inf or nan')
    return sigma


def compute_log_loss_gradient(y, X, w, delta=None):
    """Compute gradient of log loss"""
    n = len(y)
    disc = sigmoid(X.dot(w)) - y
    ll_grad = X.T.dot(disc)
    return ll_grad / n if delta is None else ll_grad / n + delta * w / n


def compute_log_loss_hessian(X, w, delta=None):
    pred = sigmoid(X.dot(w))
    n = len(X)
    try:
        d = len(w)
        s = (1 / n * pred * (1 - pred))[:, np.newaxis]
    except ValueError:
        d = 1
        s = (1 / n * pred * (1 - pred))
    H = X.T.dot(s * X)
    return H if delta is None else H + delta * np.eye(d) / n


def compute_log_loss(y, X, w, delta=None):
    """compute the cost by negative log likelihood/X-entropy
    with regularizer optionally"""
    n = len(y)
    pred = X.dot(w)
    ll = (np.sum(y * (np.log(1 + np.exp(pred)) - pred))  # Positive frac.
          + np.sum((1 - y) * np.log(1 + np.exp(pred))))  # Negative frac.
    return (ll / n if delta is None
            else ll / n + 0.5 * delta * w.dot(w)) / n


def create_sine_data(n=100, n_test=0, offset=0, noise_sigma=0.25):
    scale = 7
    x = np.sort((np.random.rand(n) - 0.5) * scale)
    y = np.sin(x + offset) + np.random.randn(n) * noise_sigma
    if offset != 0:
        x = np.vstack([x, np.ones_like(x)]).T
    else:
        x = x.reshape(n, 1)
    if n_test:
        x_test = np.linspace(-scale*1.1, scale*1.1, n_test)
        y_test = np.sin(x_test + offset) + np.random.randn(n_test) * noise_sigma
        if offset != 0:
            x_test = np.vstack([x_test, np.ones_like(x_test)]).T
        else:
            x_test = x_test.reshape(n_test, 1)
        return x, y, x_test, y_test
    return x, y


def is_psd(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def get_samples(mu, sigma, seed=None, n_samples=1000):
    np.random.seed(seed)
    return mu[:, np.newaxis] + sigma @ np.random.randn(len(mu), n_samples)


def get_sigmoid_samples(mu, sigma, seed=None, n_samples=1000):
    np.random.seed(seed)
    return sigmoid(mu[:, np.newaxis] + sigma @ np.random.randn(len(mu), n_samples))


def get_scale(Covariance):
    try:
        return np.linalg.cholesky(Covariance)
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.diag(Covariance)))


def get_mnist_data(hw):
    transform = Compose([Resize((hw, hw)), ToTensor()])
    D = MNIST(root='data', transform=transform)
    data, targets = (torch.cat([D[i][0] for i in range(len(D))]).numpy().reshape(-1, hw ** 2).astype(np.float),
                     np.array([D[i][1] for i in range(len(D))]).astype(np.float))
    return data, targets


def identity(x):
    return x


def gp_kernel_divergence(K0, K1):
    Kinv1 = np.linalg.pinv(K1)
    _, lndet_1 = np.linalg.slogdet(K1)
    _, lndet_0 = np.linalg.slogdet(K0)
    return np.trace(Kinv1 @ K0) + lndet_1 - lndet_0


def rlbl(y_prob):
    return (y_prob > 0.5).astype(int)
