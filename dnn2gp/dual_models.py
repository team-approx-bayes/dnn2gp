import numpy as np
import scipy as sp


class DualModel:
    def __init__(self, X_hat, y_hat, s_noise, m_0, S_0=None,
                 P_0=None, comp_post=True):
        self.X_hat = X_hat
        self.y_hat = y_hat
        self.n = self.y_hat.shape[0]
        assert not (S_0 is None and P_0 is None)
        self.m_0 = m_0
        self._S_0 = S_0
        self._P_0 = P_0
        self.s_noise = s_noise
        if comp_post:
            self.compute_posterior()

    def compute_posterior(self, Us=None, vs=None, theta_star=None, delta=None):
        raise NotImplementedError

    @property
    def S_0(self):
        if self._S_0 is None:
            self._S_0 = np.linalg.pinv(self.P_0)
        return self._S_0

    @property
    def P_0(self):
        if self._P_0 is None:
            self._P_0 = np.linalg.pinv(self.S_0)
        return self._P_0

    @property
    def beta(self):
        return 1 / (self.s_noise ** 2)

    def posterior_predictive_f(self, X_hat_star, **kwargs):
        # can generate joint function draws for f
        raise NotImplementedError

    def posterior_predictive_y(self, X_hat_star):
        # can generate new data according to model
        m_f_hat_star, s_f_hat_star = self.posterior_predictive_f(X_hat_star)
        return m_f_hat_star, s_f_hat_star + np.eye(len(m_f_hat_star)) / self.beta


class DualLinearRegression(DualModel):

    def __init__(self, X_hat, y_hat, s_noise, m_0, S_0=None,
                 P_0=None, comp_post=True):
        self.m_post = None
        self.m_post_fix = None
        self.S_post = None
        self.P_post = None
        super().__init__(X_hat, y_hat, s_noise, m_0, S_0, P_0, comp_post)

    def compute_posterior(self, Us=None, vs=None, Ss=None, theta_star=None, delta=None):
        self.P_post = self.P_0 + self.beta[np.newaxis, :] * self.X_hat.T @ self.X_hat
        self.S_post = np.linalg.pinv(self.P_post)
        self.m_post = self.S_post @ (self.P_0 @ self.m_0 + self.X_hat.T @ (self.y_hat * self.beta))
        if Us is not None and vs is not None:
            beta = 1 / Ss
            hess_term = Us.T @ (Us * beta[:, np.newaxis]) @ theta_star
            grad_term = delta * theta_star  # = - Us.T @ vs due to full gradient equal zero
            self.m_post_fix = self.S_post @ (self.P_0 @ self.m_0 + hess_term + grad_term)

    @property
    def posterior(self):
        return self.m_post, self.S_post, self.P_post

    def posterior_predictive_f(self, X_hat_star, use_fix=False, diag_only=False):
        m_f_hat_star = X_hat_star @ (self.m_post_fix if use_fix else self.m_post)
        if diag_only:
            s_f_hat_star = np.einsum('ij,ij->i', X_hat_star @ self.S_post, X_hat_star)
        else:
            s_f_hat_star = X_hat_star @ self.S_post @ X_hat_star.T
        return m_f_hat_star, s_f_hat_star


class DualGPRegression(DualModel):

    def __init__(self, X_hat, y_hat, s_noise, m_0, S_0=None,
                 P_0=None, comp_post=True):
        self.K = None
        self.K_inv = None
        self.m_offset = None
        super().__init__(X_hat, y_hat, s_noise, m_0, S_0, P_0, comp_post)

    def compute_posterior(self, Us=None, vs=None, theta_star=None, delta=None):
        if np.allclose(np.diag(np.diag(self.S_0)), self.S_0):  # diag speed up
            s = np.diag(self.S_0)[:, np.newaxis]
            self.K = self.X_hat @ (s * self.X_hat.T)
        else:
            self.K = self.X_hat @ self.S_0 @ self.X_hat.T
        self.K_inv = np.linalg.pinv(self.K + self.s_noise ** 2 * np.eye(self.n))
        self.m_offset = self.X_hat @ self.m_0

    def posterior_predictive_f(self, X_hat_star, use_fix=False, diag_only=False):
        if np.allclose(np.diag(np.diag(self.S_0)), self.S_0):  # diag speed up
            s = np.diag(self.S_0)[:, np.newaxis]
            K_star_ = X_hat_star @ (s * self.X_hat.T)
            if diag_only:
                K_star_star = np.einsum('ij,ji->i', X_hat_star, (s * X_hat_star.T))
            else:
                K_star_star = X_hat_star @ (s * X_hat_star.T)
        else:
            K_star_ = X_hat_star @ self.S_0 @ self.X_hat.T
            if diag_only:
                K_star_star = np.einsum('ij,ij->i', X_hat_star @ self.S_0, X_hat_star)
            else:
                K_star_star = X_hat_star @ self.S_0 @ X_hat_star.T
        m_f_hat_star = X_hat_star @ self.m_0 + K_star_ @ self.K_inv @ (self.y_hat - self.m_offset)
        if diag_only:
            s_f_hat_star = K_star_star - np.einsum('ij,ij->i', K_star_ @ self.K_inv, K_star_)
        else:
            s_f_hat_star = K_star_star - K_star_ @ self.K_inv @ K_star_.T
        return m_f_hat_star, s_f_hat_star

    def log_marginal_likelihood(self, X_hat=None, y_hat=None):
        if X_hat is None or y_hat is None:
            X_hat, y_hat = self.X_hat, self.y_hat
        if np.allclose(np.diag(np.diag(self.S_0)), self.S_0):  # diag speed up
            s = np.diag(self.S_0)[:, np.newaxis]
            K = X_hat @ (s * X_hat.T) + np.eye(len(X_hat)) * self.s_noise ** 2
        else:
            K = X_hat @ self.S_0 @ X_hat.T + np.eye(len(X_hat)) * self.s_noise ** 2
        try:
            L = sp.linalg.cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return -np.inf
        if y_hat.ndim == 1:
            y_hat = (y_hat - (X_hat @ self.m_0))[:, np.newaxis]
        alpha = sp.linalg.cho_solve((L, True), y_hat)  # Line 3
        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_hat, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        return log_likelihood


if __name__ == '__main__':
    # Test Case that GP and LR align in posterior predictive.
    # Visual inspection test on some sinusoid data.
    # Sampling from functions can also be done and plotted.
    import matplotlib.pyplot as plt
    from dnn2gp.gaussian import Gaussian
    show_samples = False
    n = 100
    noise_sigma = 0.25
    scale = 7
    x = np.sort((np.random.rand(n) - 0.5) * scale)
    y = np.sin(x) + np.random.randn(n) * noise_sigma
    xs = np.linspace(-0.5 * scale, 0.5 * scale, n)
    x, xs = x.reshape(n, 1), xs.reshape(n, 1)
    linreg = DualLinearRegression(x, y, np.array([noise_sigma]), np.array([0.2]), P_0=0.1 * np.eye(1))
    gpreg = DualGPRegression(x, y, np.array([noise_sigma]), np.array([0.2]), P_0=0.1 * np.eye(1))
    pred_lr = linreg.posterior_predictive_f(xs)
    pred_lr_dist = Gaussian(pred_lr[0], pred_lr[1])
    pred_gp = gpreg.posterior_predictive_f(xs)
    pred_gp_dist = Gaussian(pred_gp[0], pred_gp[1])
    assert np.allclose(pred_gp[0], pred_lr[0])
    assert np.allclose(pred_gp[1], pred_lr[1])
    plt.scatter(x.flatten(), y, label='data')
    plt.plot(xs.flatten(), np.sin(xs), label='$f(x)$', c='k')
    plt.plot(xs.flatten(), pred_lr[0], label='LR')
    plt.plot(xs, pred_gp[0], label='GP')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$ or $f(x)$')
    if show_samples:
        for i in range(10):
            fs = pred_gp_dist.sample()
            plt.plot(xs.flatten(), fs)
    plt.show()
