import numpy as np
import torch
from torch.distributions import Normal

from dnn2gp.neural_networks import SimpleMLP
from dnn2gp.vogn import VOGN
from dnn2gp.dual_models import DualGPRegression


class VariationalNeuralRegression:
    def __init__(self, X, y, delta, n_epochs=5000, alpha=0.9, hidden_size=10, n_layers=3, n_samples_train=25,
                 n_samples_pred=500, beta=None, initial_prec=1, seed=77):
        torch.manual_seed(seed)
        self.n_epochs = n_epochs
        self.alpha = alpha  # lr
        self.model = SimpleMLP(X.shape[1], hidden_size, n_layers)
        self.losses = list()
        self.d = len(self.model.weights)
        self.n_samples_pred = n_samples_pred
        self.X = X
        self.y = y
        self.delta = delta
        self.n = self.X.shape[0]
        beta = 1 - alpha if beta is None else beta
        self.vogn = VOGN(self.model, train_set_size=1, prior_prec=delta, lr=alpha, betas=(0.0, beta),
                         num_samples=n_samples_train, inital_prec=initial_prec)
        self._compute_variational_posterior()

    @property
    def Xt(self):
        return torch.from_numpy(self.X).double()

    @property
    def yt(self):
        return torch.from_numpy(self.y).double()

    def _compute_variational_posterior(self):
        def log_lh(ypred, yt):
            dist = Normal(ypred, 1)
            return - torch.sum(dist.log_prob(yt))

        for i in range(self.n_epochs):
            def closure():
                self.vogn.zero_grad()
                ypred = self.model.forward(self.Xt)
                loss = log_lh(ypred.flatten(), self.yt)
                return loss, ypred, None

            loss, _ = self.vogn.step(closure)
            self.losses.append(loss.item())

    @property
    def KL(self):
        return self.vogn.kl_divergence()

    @property
    def loss(self):
        return self.losses[-1]

    @property
    def ELBO(self):
        return self.loss + self.KL

    def posterior_predictive_f(self, X, compute_std=False):
        Xt = torch.from_numpy(X).double()
        preds = np.array(self.vogn.get_mc_predictions(lambda x: self.model(x), Xt, mc_samples=500, ret_numpy=True))
        mean_pred = np.mean(preds, axis=0).flatten()
        if not compute_std:
            return mean_pred
        std_pred = np.std(preds, axis=0).flatten()
        return mean_pred, std_pred

    def compute_log_mlh(self, sample=True):
        Us, vs, mt, st, beta = self.vogn.dual_gp_params(self.model, self.Xt, self.yt, sample=sample)
        X_hat, y_hat = Us, Us @ mt - vs
        dual_gp = DualGPRegression(X_hat, y_hat, s_noise=np.sqrt(1 / beta), m_0=mt, S_0=np.diag(st))
        return dual_gp.log_marginal_likelihood()

    def compute_log_mlh_converged(self, sample=True):
        Us, vs, m_t, m_0, s_0 = self.vogn.dual_gp_params_star(self.model, self.Xt, self.yt, sample=sample)
        X_hat, y_hat = Us, Us @ m_t - vs
        dual_gp = DualGPRegression(X_hat, y_hat, s_noise=1., m_0=m_0, S_0=np.diag(s_0))
        return dual_gp.log_marginal_likelihood()
