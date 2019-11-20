import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import numpy as np
import seaborn as sns
import brewer2mpl

from dnn2gp.vogn import VOGN
from dnn2gp.laplace_models import NeuralNetworkRegression
from dnn2gp.dual_models import DualGPRegression, DualLinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import matplotlib.pyplot as plt

import torch
from dnn2gp.neural_networks import SimpleMLP
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import vector_to_parameters

plt.style.use('seaborn-white')
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 16
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.family'] = 'sans-serif'
plt.rc('text', usetex=True)
bmap = brewer2mpl.get_map('Set1', 'qualitative', 4)
colors = bmap.mpl_colors

class ToyData1D:
    def __init__(self, train_x, train_y, test_x=None, x_min=None, x_max=None,
                 n_test=None, normalize=False, dtype=np.float64):
        self.train_x = np.array(train_x, dtype=dtype)[:, None]
        self.train_y = np.array(train_y, dtype=dtype)[:, None]
        self.n_train = self.train_x.shape[0]
        if test_x is not None:
            self.test_x = np.array(test_x, dtype=dtype)[:, None]
            self.n_test = self.test_x.shape[0]
        else:
            self.n_test = n_test
            self.test_x = np.linspace(x_min, x_max, num=n_test, dtype=dtype)[:, None]


def load_snelson_data(n=200, dtype=np.float64):
    if n > 200:
        raise ValueError('Only 200 data points on snelson.')

    def _load_snelson(filename):
        with open('data/snelson/{fn}'.format(fn=filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")],
                            dtype=dtype)

    train_x = _load_snelson("train_inputs")
    train_y = _load_snelson("train_outputs")
    test_x = _load_snelson("test_inputs")
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return ToyData1D(train_x, train_y, test_x=test_x)

def plot_uncertainties(means, variances, labels, colors, skip=[], sigma_noise=None, plot_on_one_figure=True, name=''):
    extra_var = sigma_noise**2 if sigma_noise is not None else 0

    if plot_on_one_figure:
        plt.figure(figsize=(7, 4.0))
    for i, (l_i, m_i, v_i, c_i) in enumerate(zip(labels, means, variances, colors)):
        if i not in skip:
            if not plot_on_one_figure:
                plt.figure(figsize=(7, 4.0))

            plt.scatter(X[:,0], y, s=2, color='black')
            plt.plot (X_test[:,0], m_i, label=l_i, color=c_i)
            plt.fill_between(X_test[:,0], m_i - np.sqrt(v_i + extra_var), m_i + np.sqrt(v_i + extra_var),
                             alpha=0.2, color=c_i)
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
        if not plot_on_one_figure or i == len(labels) - 1:
            plt.ylim([-2.7, 2])
            plt.legend(loc=(0.5, -0.02))
            plt.xlim([-4, 10])
            plt.tight_layout()
    if plot_on_one_figure:
        plt.savefig('figures/regression_uncertainty_{}.pdf'.format(name))


def compute_dual_quantities(model, sigma_noise, X, y=None):
    grads = []
    lam = np.zeros((X.shape[0],))
    residual = np.zeros((X.shape[0],))
    if y is None:
        y = np.zeros((X.shape[0],))
    for i, (xi, yi) in enumerate(zip(X, y)):
        model.zero_grad()
        output = model(xi)
        output.backward()
        grad = torch.cat([p.grad.data.flatten() for p in model.parameters()]).detach()
        grads.append(grad)
        output = output.detach().numpy()
        residual[i] = sigma_noise ** (-2) * (output - yi)
        lam[i] = sigma_noise ** (-2)

    jacobian = torch.stack(grads).detach().numpy()
    return jacobian, residual, lam


def compute_dual_quantities_mc(opt, model, sigma_noise, Xs, ys, mc_samples=1):
    parameters = opt.param_groups[0]['params']
    jacobians = [[[], [], []] for _ in range(len(Xs))]
    precision = opt.state['precision']
    mu = opt.state['mu']
    for _ in range(mc_samples):
        raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
        p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
        vector_to_parameters(p, parameters)
        for i, (X, y) in enumerate(zip(Xs, ys)):
            jacobian, res, lam = compute_dual_quantities(model, sigma_noise, X, y)
            jacobians[i][0].append(jacobian)
            jacobians[i][1].append(res)
            jacobians[i][2].append(lam)
    vector_to_parameters(mu, parameters)
    return [[np.concatenate(jacobians[i][j], 0) for j in range(len(jacobians[i]))]
            for i in range(len(Xs))]

snelson_data = load_snelson_data(n=200)
X_snelson, y_snelson = snelson_data.train_x, snelson_data.train_y.reshape((-1,))
X_test_snelson = np.linspace(-4, 10, 1000).reshape((-1, 1))

mask = ((X_snelson < 1.5) | (X_snelson > 3)).flatten()
X = X_snelson[mask, :]
y = y_snelson[mask]
X_test = X_test_snelson

plt.figure(1)
plt.scatter(X, y, s=2, color='k' , label='data')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Snelson data')

## Snelson

hidden_size = 32
hidden_layers = 1
input_size = 1
activation = 'sigmoid'
delta = 0.1
sigma_noise = 0.286
initial_lr = 0.1

primal_nn = NeuralNetworkRegression(X, y, sigma_noise=sigma_noise, delta=delta, n_epochs=20000,
                                    step_size=initial_lr, hidden_size=hidden_size, n_layers=hidden_layers + 1,
                                    diagonal=True, activation=activation, lr_factor=0.99)

y_test_pred = primal_nn.predictive_map(X_test)
nn_mc_mean, nn_mc_var = primal_nn.posterior_predictive_f(X_test, 'J', n_samples=1000,
                                                                 compute_cov=True, diag_only=True)

m_0, S_0 = np.zeros(primal_nn.d), 1 / delta * np.eye(primal_nn.d)
(Us, Ss), vs = primal_nn.UsSs('J'), primal_nn.vs('J')
X_hat, y_hat, s_noise = Us, Us @ primal_nn.theta_star - vs / Ss, 1 / np.sqrt(Ss)

X_hat_test, Ss_test = primal_nn.UsSs('J', X=X_test, y=np.ones((X_test.shape[0],)))
m_0, S_0 = np.zeros(primal_nn.d), 1 / delta * np.eye(primal_nn.d)
(Us, Ss), vs = primal_nn.UsSs('J'), primal_nn.vs('J')
X_hat, y_hat, s_noise = Us, Us @ primal_nn.theta_star - vs / Ss, 1 / np.sqrt(Ss)
dual_gp = DualGPRegression(X_hat, y_hat, s_noise, m_0, S_0=S_0)
gp_mean, gp_var_primal = dual_gp.posterior_predictive_f(X_hat_test, diag_only=True)

dual_blr = DualLinearRegression(X_hat, y_hat, s_noise, m_0, S_0=S_0)
dual_blr.P_post = np.diag(np.diag(dual_blr.P_post))
dual_blr.S_post = np.diag(1/np.diag(dual_blr.P_post))
blr_mean, blr_var = dual_blr.posterior_predictive_f(X_hat_test, diag_only=True)

gp_rbf = GaussianProcessRegressor(kernel=ConstantKernel()*RBF(), n_restarts_optimizer=20,
                                 random_state=0, alpha=sigma_noise**2).fit(X, y[:, np.newaxis])

gp_rbf_pred_mean, gp_rbf_pred_std = gp_rbf.predict(X_test, return_std=True)
gp_rbf_pred_mean = gp_rbf_pred_mean.flatten()
gp_rbf_pred_var = gp_rbf_pred_std**2

labels = ['DNN-Laplace', 'DNN2GP-Laplace', 'GP-RBF', 'DNN2GP-LaplaceDiag']
means = [nn_mc_mean, y_test_pred, gp_rbf_pred_mean, y_test_pred]
variances = [nn_mc_var, gp_var_primal, gp_rbf_pred_var, blr_var]
colors = ['blue', 'red', '#555555', 'black']

plot_uncertainties(means, variances, labels, colors, skip=[3], sigma_noise=sigma_noise, name='laplace')

torch.manual_seed(100)
np.random.seed(100)

delta = 1

objective = lambda pred, target: -torch.distributions.Normal(target, sigma_noise).log_prob(
    pred).sum()
model_vi = SimpleMLP(input_size, hidden_size, hidden_layers + 1, activation)

optimizer_vi = VOGN(model_vi, train_set_size=X.shape[0], prior_prec=delta, lr=1, betas=(0.9, 0.995),
                    num_samples=10, inital_prec=50.0)

Xt = torch.from_numpy(X)
yt = torch.from_numpy(y)
X_test_t = torch.from_numpy(X_test)

epochs = 16000
for t in range(epochs):
    for xb, yb in DataLoader(TensorDataset(Xt, yt), batch_size=20):
        def closure():
            optimizer_vi.zero_grad()
            output = model_vi(xb).flatten()
            loss = objective(output, yb)
            return loss, output, None


        loss, _ = optimizer_vi.step(closure)

    if (t + 1) % 500 == 0:
        logits = model_vi(Xt).flatten()
        loss = objective(logits, yt).detach().item()
        print(f'Epoch {t + 1}, Log-loss: {loss}')


vi_pred_mean = torch.stack([model_vi(X_test_t)]).mean(0).detach().numpy()

pred_mc_samples = 1000
vi_pred_mc = torch.stack(optimizer_vi.get_mc_predictions(model_vi.forward, X_test_t, mc_samples=pred_mc_samples), 0)

vi_pred_mc_mean = vi_pred_mc.mean(0).detach().numpy().flatten()
vi_pred_mc_var = vi_pred_mc.var(0).detach().numpy().flatten()

use_mean_as_sample = False
if use_mean_as_sample:
    mc_samples = 1
    X_vi_hat, residual, lam = compute_dual_quantities(model_vi, sigma_noise, Xt, y)
    X_test_vi_hat, _, _ = compute_dual_quantities(model_vi, sigma_noise, X_test_t, None)
else:
    mc_samples = 1
    torch.manual_seed(100)
    [X_vi_hat, residual, lam], [X_test_vi_hat, _, _] = compute_dual_quantities_mc(optimizer_vi,
                                                                                  model_vi,
                                                                                  sigma_noise,
                                                                                  [Xt, X_test_t],
                                                                                  [y, None],
                                                                                  mc_samples=mc_samples)

s_noise_vi = np.sqrt(mc_samples) * np.power(lam, -0.5)
y_vi_hat = X_vi_hat @ model_vi.weights.detach().numpy() - residual / lam
dual_gp_vi = DualGPRegression(X_vi_hat, y_vi_hat, s_noise_vi, m_0, S_0=S_0)
gp_mean_vi, gp_var_vi = dual_gp_vi.posterior_predictive_f(X_test_vi_hat, diag_only=True)

dual_blr_vi = DualLinearRegression(X_vi_hat, y_vi_hat, s_noise_vi, m_0, S_0=S_0)
dual_blr_vi.m_post = optimizer_vi.state['mu'].detach().numpy()
dual_blr_vi.P_post = np.diag(optimizer_vi.state['precision'].detach().numpy())
dual_blr_vi.S_post = np.diag(1 / np.diag(dual_blr_vi.P_post))
blr_mean_vi, blr_var_vi = dual_blr_vi.posterior_predictive_f(X_test_vi_hat, diag_only=True)

labels = ['DNN-VI', 'DNN2GP-VI', 'GP-RBF', 'DNN2GP-VI-diag']
means = [vi_pred_mc_mean, vi_pred_mc_mean, gp_rbf_pred_mean, vi_pred_mc_mean]
variances = [vi_pred_mc_var, gp_var_vi, gp_rbf_pred_var, blr_var_vi]
colors = ['blue', 'red', '#555555', 'black']

plot_uncertainties(means, variances, labels, colors, skip=[3], sigma_noise=sigma_noise, name='vi')