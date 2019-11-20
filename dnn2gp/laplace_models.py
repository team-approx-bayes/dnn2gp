import numpy as np
import torch
from torch.distributions import Normal, Bernoulli
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dnn2gp.utilities import compute_log_loss_gradient, compute_log_loss_hessian, sigmoid, is_psd, identity
from dnn2gp.neural_networks import SimpleMLP, WilliamsNN, SimpleConvNet

torch.set_default_dtype(torch.double)


class LaplaceModel:
    """
    Model of type l(theta) = sum li(theta) + 1/2 delta theta^T theta
    i.e. empirical risk minimization. Subclasses determine the li.
    """
    def __init__(self, X, y, delta, compute_posterior=True):
        self.X = X
        self.y = y
        self.delta = delta
        self.n = self.X.shape[0]
        self.theta_star = None
        self.laplace_params = dict()
        if compute_posterior:
            self.compute_theta_star()

    def compute_theta_star(self):
        # compute theta_star and the posterior parameters (m, S, P)
        raise NotImplementedError

    @property
    def exact_posterior(self):
        raise ValueError('This class does not provide exact posterior')

    def UsSs(self, hessian_approx, X=None, y=None):
        arg_check(hessian_approx)
        raise NotImplementedError

    def vs(self, hessian_approx):
        arg_check(hessian_approx)
        raise NotImplementedError

    def q_laplace(self, hessian_approx, skip_inverse=False):
        """Use the theta_star unchanged and compute chosen approximate Hessian
        -> Precision matrix of q_laplace"""
        if self.laplace_params.get(hessian_approx, None) is not None:
            return self.laplace_params[hessian_approx]
        Us, Ss = self.UsSs(hessian_approx)
        P = self.delta * np.eye(self.d)
        P += Us.T @ (Us * Ss[:, np.newaxis])
        if skip_inverse:
            S = None
        else:
            S = np.linalg.pinv(P)
            S = 0.5 * (S + S.T)
        self.laplace_params[hessian_approx] = (self.theta_star, S, P)
        return self.theta_star, S, P

    def posterior_predictive_f(self, X_star, hessian_approx):
        raise NotImplementedError

    def predictive_map(self, X_star):
        raise NotImplementedError


class LinearRegression(LaplaceModel):
    # noise variance equal to one

    def __init__(self, X, y, delta, sigma_noise=1, compute_posterior=True):
        self.S_post_exact = None
        self.P_post_exact = None
        self.m_post_exact = None
        self.sigma_noise = sigma_noise
        self.sn = self.sigma_noise ** 2
        self.bn = 1 / self.sn
        self.d = X.shape[1]
        super().__init__(X, y, delta, compute_posterior)

    def compute_theta_star(self):
        self.P_post_exact = self.delta * np.eye(self.d) + self.bn * self.X.T @ self.X
        self.S_post_exact = np.linalg.pinv(self.P_post_exact)
        self.m_post_exact = self.S_post_exact @ (self.bn * self.X.T) @ self.y
        self.theta_star = self.m_post_exact

    @property
    def exact_posterior(self):
        return self.m_post_exact, self.S_post_exact, self.P_post_exact

    def UsSs(self, hessian_approx, X=None, y=None):
        arg_check(hessian_approx)
        X, y = X if X is not None else self.X, y if y is not None else self.y
        SIs = np.ones_like(y)
        if hessian_approx == 'g':
            residual = X @ self.theta_star - y
            return self.bn * X * residual[:, np.newaxis], SIs
        elif hessian_approx == 'J':
            if len(X.shape) > 2:
                return X.transpose(0, 2, 1), self.bn * SIs
            return X, self.bn * SIs
        elif hessian_approx == 'H':
            if len(X.shape) > 2:
                return X.transpose(0, 2, 1).sum(axis=1), self.bn * SIs
            return X, self.bn * SIs

    def vs(self, hessian_approx):
        arg_check(hessian_approx)
        if hessian_approx == 'g':
            return np.ones(self.n)
        elif hessian_approx == 'J':
            return self.bn * (self.X @ self.theta_star - self.y)  # residual
        elif hessian_approx == 'H':
            return self.bn * (self.X @ self.theta_star - self.y)  # residual

    def posterior_predictive_f(self, X_star, hessian_approx):
        m, S, P = self.q_laplace(hessian_approx)
        m_f_hat_star = X_star @ m
        s_f_hat_star = X_star @ S @ X_star.T
        return m_f_hat_star, s_f_hat_star

    def posterior_predictive_empirical_f(self, X_star, hessian_approx, diagonal=False):
        m, S, P = self.q_laplace(hessian_approx)
        if diagonal:
            Scale = np.sqrt(np.diag(1 / np.diag(P)))
        else:
            l, Q = np.linalg.eigh(S)
            Scale = Q @ np.diag(np.sqrt(l))
        means = list()
        for randn_sample in np.random.randn(1000, len(self.theta_star)):
            means.append(X_star @ (m + Scale @ randn_sample))
        pred_samples = np.array(means)
        return np.mean(pred_samples, axis=0), np.cov(pred_samples.T)

    def predictive_map(self, X_star):
        return X_star @ self.theta_star


class LogisticRegression(LaplaceModel):

    def __init__(self, X, y, delta, compute_posterior=True, n_iter=100,
                 newton_step_size=0.9, jacobian_bound=True):
        self.n_iter = n_iter
        self.alpha = newton_step_size
        self.jacobian_bound = jacobian_bound
        self.d = X.shape[1]
        super().__init__(X, y, delta, compute_posterior)

    def compute_theta_star(self):
        theta = np.zeros(self.d)
        for i in range(self.n_iter):
            g = compute_log_loss_gradient(self.y, self.X, theta, delta=self.delta)
            H = compute_log_loss_hessian(self.X, theta, delta=self.delta)
            p_k, _, _, _ = np.linalg.lstsq(H, -g, rcond=-1)
            theta = theta + self.alpha * p_k
        self.theta_star = theta

    def UsSs(self, hessian_approx, X=None, y=None):
        arg_check(hessian_approx)
        X, y = X if X is not None else self.X, y if y is not None else self.y
        SIs = np.ones_like(y)
        if hessian_approx == 'g':
            residual = sigmoid(X @ self.theta_star) - y
            return X * residual[:, np.newaxis], SIs
        elif hessian_approx == 'J':
            pred = sigmoid(X @ self.theta_star)
            s = pred * (1 - pred)
            if len(X.shape) > 2:
                return X.transpose(0, 2, 1) * c
            return X, s
        elif hessian_approx == 'H':
            if len(X.shape) > 2:
                raise NotImplementedError
            pred = sigmoid(X @ self.theta_star)
            s = pred * (1 - pred)
            return X, s

    def vs(self, hessian_approx):
        arg_check(hessian_approx)
        if hessian_approx == 'g':
            return np.ones(self.n)
        elif hessian_approx == 'J':
            return sigmoid(self.X @ self.theta_star) - self.y  # residual
        elif hessian_approx == 'H':
            return sigmoid(self.X @ self.theta_star) - self.y  # residual

    def posterior_predictive_f(self, X_star, hessian_approx):
        m, S, P = self.q_laplace(hessian_approx)
        m_f_hat_star = X_star @ m
        s_f_hat_star = X_star @ S @ X_star.T
        return m_f_hat_star, s_f_hat_star

    def posterior_predictive_empirical_f(self, X_star, hessian_approx, diagonal=False):
        m, S, P = self.q_laplace(hessian_approx)
        if diagonal:
            Scale = np.diag(np.sqrt(1 / np.diag(P)))
        else:
            l, Q = np.linalg.eigh(S)
            Scale = Q @ np.diag(np.sqrt(l))
        means = list()
        for randn_sample in np.random.randn(1000, len(self.theta_star)):
            means.append(X_star @ (m + Scale @ randn_sample))
        pred_samples = np.array(means)
        return np.mean(pred_samples, axis=0), np.cov(pred_samples.T)

    def predictive_map(self, X_star):
        return X_star @ self.theta_star


class NeuralNetworkRegression(LaplaceModel):

    def __init__(self, X, y, delta, sigma_noise=1, compute_posterior=True, n_epochs=10000, activation='tanh',
                 step_size=1e-3, hidden_size=10, n_layers=3, n_samples_pred=100, diagonal=True, seed=77, lr_factor=0.99):
        torch.manual_seed(seed)
        self.n_epochs = n_epochs
        self.alpha = step_size
        self.model = self._init_model(X.shape[1], hidden_size, n_layers, activation)
        self.losses = list()
        self.d = len(self.model.weights)
        self.n_samples_pred = n_samples_pred
        self.diagonal = diagonal
        self.sigma_noise = sigma_noise
        self.sn = self.sigma_noise ** 2
        self.bn = 1 / self.sn
        self.lr_factor = lr_factor
        super().__init__(X, y, delta, compute_posterior)

    def _init_model(self, in_size, hidden_size, n_layers, activation='tanh'):
        return SimpleMLP(in_size, hidden_size, n_layers, activation)

    @property
    def Xt(self):
        return torch.from_numpy(self.X)

    @property
    def yt(self):
        return torch.from_numpy(self.y)

    def compute_theta_star(self):
        opt = Adam(self.model.parameters(), lr=self.alpha, weight_decay=0)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=self.lr_factor, min_lr=1e-10)
        for i in range(self.n_epochs):
            opt.zero_grad()
            output = self.model.forward(self.Xt)
            likelihood = Normal(output.flatten(), self.sigma_noise)
            prior = Normal(0, 1 / np.sqrt(self.delta))
            nll = - torch.sum(likelihood.log_prob(self.yt))
            loss = nll - torch.sum(prior.log_prob(self.model.weights))
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
            if (i+1) % 500 == 0:
                print(i + 1, nll.item())
            self.losses.append(loss.item())
        self.theta_star = self.model.weights.detach().numpy()

    def UsSs(self, hessian_approx, X=None, y=None):
        arg_check(hessian_approx)
        X, y = (torch.from_numpy(X) if X is not None else self.Xt,
                torch.from_numpy(y) if y is not None else self.y)
        Us = list()
        Ss = np.ones(len(y))
        for xi, yi in zip(X, y):
            self.model.zero_grad()
            output = self.model.forward(xi)
            if hessian_approx == 'g':
                likelihood = Normal(output.flatten(), self.sigma_noise)
                loss = - likelihood.log_prob(yi)
                loss.backward()
                Us.append(self.model.gradient)
            elif hessian_approx == 'J':
                output.backward()
                Us.append(self.model.gradient)
            elif hessian_approx == 'H':
                raise NotImplementedError
        return np.stack(Us), Ss if hessian_approx == 'g' else self.bn * Ss

    def vs(self, hessian_approx):
        arg_check(hessian_approx)
        if hessian_approx == 'g':
            return np.ones(self.n)
        elif hessian_approx == 'J':
            with torch.no_grad():
                pred = self.model(self.Xt).flatten().numpy()
            return self.bn * (pred - self.y)  # residual
        elif hessian_approx == 'H':
            raise NotImplementedError

    def posterior_predictive_f(self, X_star, hessian_approx, compute_cov=True, diag_only=False, n_samples=None):
        m, S, P = self.q_laplace(hessian_approx)
        if self.diagonal:
            Scale = np.diag(np.sqrt(1 / np.diag(P)))
        else:
            l, Q = np.linalg.eigh(P)
            Scale = Q @ np.diag(1 / np.sqrt(l))
        theta_star = self.model.weights
        self.model.eval()
        means = list()
        for randn_sample in np.random.randn(n_samples or self.n_samples_pred, len(theta_star)):
            weight_sample = m + Scale @ randn_sample
            self.model.adjust_weights(weight_sample)
            fs = self.model(torch.from_numpy(X_star)).flatten().detach().numpy()
            means.append(fs)
        self.model.adjust_weights(theta_star)
        self.model.train()
        pred_samples = np.array(means)
        if compute_cov:
            if diag_only:
                return np.mean(pred_samples, axis=0), np.var(pred_samples, axis=0)
            else:
                return np.mean(pred_samples, axis=0), np.cov(pred_samples.T)
        else:
            return np.mean(pred_samples, axis=0)

    def predictive_map(self, X_star):
        return self.model(torch.from_numpy(X_star)).detach().flatten().numpy()


class NeuralNetworkClassification(LaplaceModel):

    def __init__(self, X, y, delta, compute_posterior=True, n_epochs=10000,
                 step_size=1e-2, hidden_size=10, n_layers=3, n_samples_pred=100, diagonal=True,
                 jacobian_bound=True, activation='tanh'):
        self.n_epochs = n_epochs
        self.alpha = step_size
        self.model = self._init_model(X.shape[1], hidden_size, n_layers, activation)
        self.losses = list()
        self.d = len(self.model.weights)
        self.n_samples_pred = n_samples_pred
        self.diagonal = diagonal
        self.jacobian_bound = jacobian_bound
        super().__init__(X, y, delta, compute_posterior)

    def _init_model(self, in_size, hidden_size, n_layers, activation='tanh'):
        return SimpleMLP(in_size, hidden_size, n_layers, activation)

    @property
    def Xt(self):
        return torch.from_numpy(self.X)

    @property
    def yt(self):
        return torch.from_numpy(self.y)

    def compute_theta_star(self):
        opt = Adam(self.model.parameters(), lr=self.alpha, weight_decay=0)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=0.99, min_lr=1e-10)
        for i in range(self.n_epochs):
            opt.zero_grad()
            output = self.model.forward(self.Xt)
            likelihood = Bernoulli(logits=output.flatten())
            prior = Normal(0, 1 / np.sqrt(self.delta))
            loss = - torch.sum(likelihood.log_prob(self.yt)) - torch.sum(prior.log_prob(self.model.weights))
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
            self.losses.append(loss.item())
        self.theta_star = self.model.weights.detach().numpy()

    def UsSs(self, hessian_approx, X=None, y=None):
        arg_check(hessian_approx)
        X, y = (torch.from_numpy(X) if X is not None else self.Xt,
                torch.from_numpy(y) if y is not None else self.y)
        Us = list()
        sfacs = list()
        SIs = np.ones(len(y))
        for xi, yi in zip(X, y):
            self.model.zero_grad()
            output = self.model.forward(xi)
            if hessian_approx == 'g':
                likelihood = Bernoulli(logits=output.flatten())
                loss = - likelihood.log_prob(yi)
                loss.backward()
                Us.append(self.model.gradient)
            elif hessian_approx == 'J':
                output.backward()
                lout = torch.sigmoid(output.detach()).item()
                sfacs.append(lout * (1 - lout))
                Us.append(self.model.gradient)
            elif hessian_approx == 'H':
                raise NotImplementedError
        if hessian_approx == 'J':
            sfac = np.stack(sfacs)
            sfac = np.where(sfac < 1e-9, 1e-9, sfac)
        return np.stack(Us), SIs if hessian_approx == 'g' else sfac

    def vs(self, hessian_approx):
        arg_check(hessian_approx)
        if hessian_approx == 'g':
            return np.ones(self.n)
        elif hessian_approx == 'J':
            self.model.zero_grad()
            with torch.no_grad():
                output = self.model.forward(self.Xt).flatten()
                pred = torch.sigmoid(output).numpy()
            return pred - self.y  # residual
        elif hessian_approx == 'H':
            raise NotImplementedError

    def posterior_predictive_f(self, X_star, hessian_approx, link=identity, compute_cov=True, n_samples=None):
        m, _, P = self.q_laplace(hessian_approx, skip_inverse=True)
        if self.diagonal:
            Scale = np.diag(np.sqrt(1 / np.diag(P)))
        else:
            l, Q = np.linalg.eigh(P)
            Scale = Q @ np.diag(1 / np.sqrt(l))
        theta_star = self.model.weights
        self.model.eval()
        means = list()
        for randn_sample in np.random.randn(n_samples or self.n_samples_pred, len(theta_star)):
            weight_sample = m + Scale @ randn_sample
            self.model.adjust_weights(weight_sample)
            fs = link(self.model(torch.from_numpy(X_star)).flatten().detach().numpy())
            means.append(fs)
        self.model.adjust_weights(theta_star)
        self.model.train()
        pred_samples = np.array(means)
        if compute_cov:
            return np.mean(pred_samples, axis=0), np.cov(pred_samples.T)
        else:
            return np.mean(pred_samples, axis=0)

    def predictive_map(self, X_star):
        return self.model(torch.from_numpy(X_star)).detach().flatten().numpy()


class WilliamsNetwork(NeuralNetworkRegression):

    def __init__(self, X, y, sigma_u, sigma_v, sigma_b, compute_posterior=True, n_epochs=10000, sigma_noise=1,
                 step_size=1e-3, hidden_size=10, n_samples_pred=100, diagonal=False):
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.sigma_b = sigma_b
        super().__init__(X, y, None, sigma_noise, compute_posterior, n_epochs, step_size,
                         hidden_size, None, n_samples_pred, diagonal)

    def _init_model(self, in_size, hidden_size, n_layers, activation):
        return WilliamsNN(in_size, hidden_size)

    def compute_theta_star(self):
        opt = Adam(self.model.parameters(), lr=self.alpha, weight_decay=0)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=0.99, min_lr=1e-10)
        for i in range(self.n_epochs):
            opt.zero_grad()
            output = self.model.forward(self.Xt)
            likelihood = Normal(output.flatten(), 1.0)
            prior_u = Normal(0, self.sigma_u)
            prior_v = Normal(0, self.sigma_v)
            prior_b = Normal(0, self.sigma_b)
            loss = - (torch.sum(likelihood.log_prob(self.yt)) + torch.sum(prior_u.log_prob(self.model.U))
                      + torch.sum(prior_v.log_prob(self.model.V)) + torch.sum(prior_b.log_prob(self.model.b)))
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
            self.losses.append(loss.item())
        self.theta_star = self.model.weights.detach().numpy()


class LogisticWilliamsNetwork(NeuralNetworkClassification):

    def __init__(self, X, y, sigma_u, sigma_v, sigma_b, compute_posterior=True, n_epochs=10000,
                 step_size=1e-3, hidden_size=10, n_samples_pred=100, diagonal=False):
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.sigma_b = sigma_b
        super().__init__(X, y, None, compute_posterior, n_epochs, step_size,
                         hidden_size, None, n_samples_pred, diagonal, jacobian_bound=False)

    def _init_model(self, in_size, hidden_size, n_layers, activation):
        return WilliamsNN(in_size, hidden_size)

    def compute_theta_star(self):
        opt = Adam(self.model.parameters(), lr=self.alpha, weight_decay=0)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=0.99, min_lr=1e-10)
        for i in range(self.n_epochs):
            opt.zero_grad()
            output = self.model.forward(self.Xt)
            likelihood = Bernoulli(logits=output.flatten())
            prior_u = Normal(0, self.sigma_u)
            prior_v = Normal(0, self.sigma_v)
            prior_b = Normal(0, self.sigma_b)
            loss = - (torch.sum(likelihood.log_prob(self.yt)) + torch.sum(prior_u.log_prob(self.model.U))
                      + torch.sum(prior_v.log_prob(self.model.V)) + torch.sum(prior_b.log_prob(self.model.b)))
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
            self.losses.append(loss.item())
        self.theta_star = self.model.weights.detach().numpy()


class CNNClassification(NeuralNetworkClassification):

    def __init__(self, X, y, delta, compute_posterior=True, n_epochs=10000,
                 step_size=1e-2, hidden_size=10, n_samples_pred=100, diagonal=True,
                 jacobian_bound=True, hw=None):
        assert hw is not None
        self.hw = hw
        super().__init__(X, y, delta, compute_posterior, n_epochs, step_size, hidden_size, None,
                         n_samples_pred, diagonal, jacobian_bound)

    def _init_model(self, in_size, hidden_size, n_layers, activation):
        return SimpleConvNet(self.hw)


def arg_check(hessian_approx):
    if hessian_approx not in {'g', 'J', 'H'}:
        raise ValueError('Hessian approx parameter needs to be one of g, J, H')
