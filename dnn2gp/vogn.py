import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F


required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


class VOGN(Optimizer):

    def __init__(self, model, train_set_size, lr=1e-3, betas=(0.9, 0.999), prior_mu=None, prior_prec=1.0,
                 inital_prec=1.0, num_samples=1, momentum=None):

        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if prior_mu is not None and not torch.is_tensor(prior_mu):
            raise ValueError("Invalid prior mu value (from previous task): {}".format(prior_mu))
        if torch.is_tensor(prior_prec):
            if (prior_prec < 0.0).all():
                raise ValueError("Invalid prior precision tensor: {}".format(prior_prec))
        else:
            if prior_prec < 0.0:
                raise ValueError("Invalid prior precision value: {}".format(prior_prec))
        if torch.is_tensor(inital_prec):
            if (inital_prec < 0.0).all():
                raise ValueError("Invalid initial precision tensor: {}".format(inital_prec))
        else:
            if inital_prec < 0.0:
                raise ValueError("Invalid initial precision value: {}".format(inital_prec))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if num_samples < 1:
            raise ValueError("Invalid num_samples parameter: {}".format(num_samples))
        if train_set_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_set_size))
        if momentum is not None and not torch.is_tensor(momentum):
            raise ValueError("Invalid momentum initialisation: {}".format(momentum))

        defaults = dict(lr=lr, betas=betas, prior_mu=prior_mu, prior_prec=prior_prec, inital_prec=inital_prec,
                        num_samples=num_samples, train_set_size=train_set_size, momentum=momentum)
        super().__init__(model.parameters(), defaults)

        self.train_modules = []
        self.set_train_modules(model)
        for module in self.train_modules:
            module.register_forward_hook(update_input)

        defaults = self.defaults
        parameters = self.param_groups[0]['params']
        device = parameters[0].device

        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()

        if torch.is_tensor(defaults['prior_mu']):
            self.state['prior_mu'] = defaults['prior_mu'].to(device)
        else:
            self.state['prior_mu'] = torch.zeros_like(p, device=device)

        if torch.is_tensor(defaults['inital_prec']):
            self.state['precision'] = defaults['inital_prec'].to(device)
        else:
            self.state['precision'] = torch.ones_like(p, device=device) * defaults['inital_prec']

        if torch.is_tensor(defaults['prior_prec']):
            self.state['prior_prec'] = defaults['prior_prec'].to(device)
        else:
            self.state['prior_prec'] = torch.ones_like(p, device=device) * defaults['prior_prec']

        if torch.is_tensor(defaults['momentum']):
            self.state['momentum'] = defaults['momentum'].to(device)
        else:
            self.state['momentum'] = torch.zeros_like(p, device=device)

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def step(self, closure):
        if closure is None:
            raise RuntimeError(
                'For now, VOGN only supports that the model/loss can be reevaluated inside the step function')

        defaults = self.defaults
        parameters = self.param_groups[0]['params']
        lr = self.param_groups[0]['lr']
        momentum_beta = defaults['betas'][0]
        beta = defaults['betas'][1]
        momentum = self.state['momentum']

        mu = self.state['mu']
        precision = self.state['precision']
        prior_mu = self.state['prior_mu']
        prior_prec = self.state['prior_prec']

        grad_hat = torch.zeros_like(mu)
        ggn_hat = torch.zeros_like(mu)

        loss_list = []
        pred_list = []
        for _ in range(defaults['num_samples']):
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)

            #loss, preds = closure()
            # Get loss and predictions
            loss, preds, residuals = closure()
            pred_list.append(preds)

            lc = []
            for module in self.train_modules:
                lc.append(module.output)

            linear_grad = torch.autograd.grad(loss, lc)
            loss_list.append(loss.detach())

            grad = []
            ggn = []
            for i, module in enumerate(self.train_modules):
                G = linear_grad[i]
                A = module.input.clone().detach()
                M = A.shape[0]
                G *= M
                G2 = torch.mul(G, G)
                # R = residuals.view(-1, 1)
                # G2 = torch.mul(G / R, G / R)

                if isinstance(module, nn.Linear):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij,ik->jk', G, A))
                    ggn.append(torch.einsum('ij, ik->jk', G2, A2))
                    if module.bias is not None:
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

                if isinstance(module, nn.Conv2d):
                    A = F.unfold(A, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding,
                                 stride=module.stride)
                    A2 = torch.mul(A, A)
                    _, k, hw = A.shape
                    _, c, _, _ = G.shape
                    G = G.view(M, c, -1)
                    G2 = G2.view(M, c, -1)
                    grad.append(torch.einsum('ijl,ikl->jk', G, A))
                    ggn.append(torch.einsum('ijl,ikl->jk', G2, A2))
                    if module.bias is not None:
                        A = torch.ones((M, 1, hw), device=A.device)
                        grad.append(torch.einsum('ijl->j', G))
                        ggn.append(torch.einsum('ijl->j', G2))

                if isinstance(module, nn.BatchNorm1d):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij->j', torch.mul(G, A)))
                    ggn.append(torch.einsum('ij->j', torch.mul(G2, A2)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

                if isinstance(module, nn.BatchNorm2d):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ijkl->j', torch.mul(G, A)))
                    ggn.append(torch.einsum('ijkl->j', torch.mul(G2, A2)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ijkl->j', G))
                        ggn.append(torch.einsum('ijkl->j', G2))

            grad = parameters_to_vector(grad).div(M).detach()
            ggn = parameters_to_vector(ggn).div(M).detach()

            grad_hat.add_(grad)
            ggn_hat.add_(ggn)

        grad_hat = grad_hat.mul(defaults['train_set_size'] / defaults['num_samples'])
        ggn_hat.mul_(defaults['train_set_size'] / defaults['num_samples'])
        momentum.mul_(momentum_beta).add_((1 - momentum_beta), grad_hat)
        loss = torch.mean(torch.stack(loss_list))
        precision.mul_(beta).add_((1 - beta), ggn_hat + prior_prec)
        mu.addcdiv_(-lr, momentum + torch.mul(mu - prior_mu, prior_prec), precision)
        vector_to_parameters(self.state['mu'], self.param_groups[0]['params'])

        return loss, pred_list

    @staticmethod
    def _kl_gaussian(p_mu, p_sigma, q_mu, q_sigma):
        log_std_diff = torch.sum(torch.log(p_sigma ** 2) - torch.log(q_sigma ** 2))
        mu_diff_term = torch.sum((q_sigma ** 2 + (p_mu - q_mu) ** 2) / p_sigma ** 2)
        const = list(q_mu.size())[0]
        return 0.5 * (mu_diff_term - const + log_std_diff)

    def kl_divergence(self):
        prec0 = self.state['prior_prec']
        prec = self.state['precision']
        mu = self.state['mu']
        sigma = 1. / torch.sqrt(prec)
        mu0 = self.state['prior_mu']
        if torch.is_tensor(prec0):
            sigma0 = 1. / torch.sqrt(prec0)
        else:
            sigma0 = 1. / np.sqrt(prec0)
        kl = self._kl_gaussian(p_mu=mu0, p_sigma=sigma0, q_mu=mu, q_sigma=sigma)
        return kl

    def get_mc_predictions(self, forward_function, inputs, mc_samples=1, ret_numpy=False, *args, **kwargs):
        parameters = self.param_groups[0]['params']
        predictions = []
        precision = self.state['precision']
        mu = self.state['mu']
        for _ in range(mc_samples):
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)
            outputs = forward_function(inputs, *args, **kwargs).detach()
            if ret_numpy:
                outputs = outputs.cpu().numpy()
            predictions.append(outputs)
        vector_to_parameters(mu, parameters)

        return predictions

    def compute_linprior(self):
        beta, alpha = self.defaults['betas'][1], self.param_groups[0]['lr']
        mu_t = self.state['mu']
        mu_0 = self.state['prior_mu']
        p_t = self.state['precision']
        p_0 = self.state['prior_prec']
        s_t = 1 / (alpha * p_t + beta * p_0)
        m_t = s_t * (alpha * p_t * mu_t + beta * p_0 * mu_0)
        return m_t, s_t

    def dual_gp_params(self, model, X, y, sample=True):
        """Returns, Us, vs, mt, st, beta
        """
        mu_t = self.state['mu']
        p_t = self.state['precision']
        m_t, s_t = self.compute_linprior()
        beta, alpha = self.defaults['betas'][1], self.param_groups[0]['lr']

        model.eval()
        parameters = self.param_groups[0]['params']
        if sample:
            raw_noise = torch.normal(mean=torch.zeros_like(mu_t), std=1.0)
            p = torch.addcdiv(mu_t, 1., raw_noise, torch.sqrt(p_t))
            vector_to_parameters(p, parameters)
        else:
            vector_to_parameters(mu_t, parameters)

        Us = list()
        output = model.forward(X).flatten()
        n = len(X)
        for i in range(n):
            model.zero_grad()
            output[i].backward(retain_graph=(i < (n-1)))
            Us.append(model.gradient)
        vs = output.detach().numpy() - y.detach().numpy()

        # Reset model parameters to mean
        vector_to_parameters(mu_t, parameters)

        return np.stack(Us), np.array(vs), m_t.detach().numpy(), s_t.detach().numpy(), beta

    def dual_gp_params_star(self, model, X, y, sample=False):
        Us, vs, m_t, _, _ = self.dual_gp_params(model, X, y, sample=sample)
        s_0 = 1 / self.state['prior_prec'].detach().numpy()
        if sample:
            precision = self.state['precision']
            mu = self.state['mu']
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            m_t = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            m_t = m_t.detach().numpy()
        return Us, vs, m_t, np.zeros_like(m_t), s_0

    def get_dual_predictions(self, jac_closure, mc_samples=10, ret_jac=False):
        mu = self.state['mu']
        precision = self.state['precision']
        parameters = self.param_groups[0]['params']
        J_list = []
        fxs = []
        Jv_list = []
        for _ in range(mc_samples):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)

            # Get loss and predictions
            preds, J = jac_closure()
            fxs.append(preds)
            J_list.append(J)  # each J in n x p
            Jv_list.append(J @ p)
        vector_to_parameters(mu, parameters)
        fx_hat = torch.mean(torch.stack(fxs), 0).flatten()
        J_hat = torch.mean(torch.stack(J_list), 0)
        Jv_hat = torch.mean(torch.stack(Jv_list), 0)
        mu_pred = fx_hat + J_hat @ mu - Jv_hat
        std_pred = torch.sqrt(torch.diag(J_hat @ torch.diag(1. / precision) @ J_hat.t()))
        if ret_jac:
            return (fx_hat.detach().numpy(), (J_hat @ mu).detach().numpy(),
                    Jv_hat.detach().numpy(), std_pred.detach().numpy())
        return mu_pred.detach().numpy(), std_pred.detach().numpy()

    def get_dual_iterative_predictions(self, mu_prev, prec_prev, jac_closure,
                                       beta=0.9, mc_samples=10, ret_jac=False):
        mu = self.state['mu']
        precision = self.state['precision']
        parameters = self.param_groups[0]['params']

        J_list = []
        fxs = []
        Jv_list = []
        for _ in range(mc_samples):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu_prev, 1., raw_noise, torch.sqrt(prec_prev))
            vector_to_parameters(p, parameters)

            # Get loss and predictions
            preds, J = jac_closure()
            fxs.append(preds)
            J_list.append(J)  # each J in n x p
            Jv_list.append(J @ p)
        vector_to_parameters(mu, parameters)
        fx_hat = torch.mean(torch.stack(fxs), 0).flatten()
        J_hat = torch.mean(torch.stack(J_list), 0)
        Jv_hat = torch.mean(torch.stack(Jv_list), 0)
        mu_pred = fx_hat + J_hat @ mu - Jv_hat
        std_pred = torch.sqrt(torch.diag(J_hat @ torch.diag(1. / precision) @ J_hat.t()))
        if ret_jac:
            return (fx_hat.detach().numpy(), (J_hat @ mu).detach().numpy(),
                    Jv_hat.detach().numpy(), std_pred.detach().numpy())
        return mu_pred.detach().numpy(), std_pred.detach().numpy()

    def get_dual_laplace_predictions(self, jac_closure, ret_jac=False):
        mu = self.state['mu']
        precision = self.state['precision']
        parameters = self.param_groups[0]['params']
        vector_to_parameters(mu, parameters)

        # Get loss and predictions
        fx, J = jac_closure()
        # flin = f + Jmu - Jmu = f and variance simple
        std = torch.sqrt(torch.diag(J @ torch.diag(1. / precision) @ J.t()))
        if ret_jac:
            return fx.flatten().detach().numpy(), (J @ mu).detach().numpy(), std.detach().numpy()
        return fx.flatten().detach().numpy(), std.detach().numpy()


class VOGGN(VOGN):

    def __init__(self, model, train_set_size, lr=1e-3, betas=(0.9, 0.999), prior_mu=None, prior_prec=1.0,
                 inital_prec=1.0, num_samples=1, momentum=None):
        super(type(self), self).__init__(model, train_set_size, lr, betas, prior_mu, prior_prec,
                                         inital_prec, num_samples, momentum)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss without doing the backward pass
        """

        if closure is None:
            raise RuntimeError(
                'For now, VOGN only supports that the model/loss can be reevaluated inside the step function')

        defaults = self.defaults
        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        lr = self.param_groups[0]['lr']
        momentum_beta = defaults['betas'][0]
        beta = defaults['betas'][1]
        momentum = self.state['momentum']

        mu = self.state['mu']
        precision = self.state['precision']
        prior_mu = self.state['prior_mu']
        prior_prec = self.state['prior_prec']

        grad_hat = torch.zeros_like(mu)
        ggn_hat = torch.zeros_like(mu)
        jac_hat = torch.zeros((defaults['train_set_size'], len(mu)))

        loss_list = []
        pred_list = []
        for _ in range(defaults['num_samples']):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(precision))
            vector_to_parameters(p, parameters)

            # Get loss and predictions
            loss, preds, residuals = closure()
            pred_list.append(preds)

            lc = []
            # Store the linear combinations
            for module in self.train_modules:
                lc.append(module.output)

            linear_grad = torch.autograd.grad(loss, lc)
            loss_list.append(loss.detach())

            grad = []
            ggn = []
            for i, module in enumerate(self.train_modules):
                G = linear_grad[i]
                A = module.input.clone().detach()
                M = A.shape[0]
                G *= M
                R = residuals.view(-1, 1)
                J = G / R
                G2 = torch.mul(J, J)

                if isinstance(module, nn.Linear):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij, ik->jk', G, A))
                    ggn.append(torch.einsum('ij, ik->jk', G2, A2))
                    if module.bias is not None:
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

            grad = parameters_to_vector(grad).div(M).detach()
            ggn = parameters_to_vector(ggn).div(M).detach()

            grad_hat.add_(grad)
            ggn_hat.add_(ggn)

        # Convert the parameter gradient to a single vector.
        grad_hat = grad_hat.mul(defaults['train_set_size'] / defaults['num_samples'])
        jac_hat = jac_hat.mul(defaults['train_set_size'] / defaults['num_samples'])
        ggn_hat.mul_(defaults['train_set_size'] / defaults['num_samples'])

        # Add momentum
        momentum.mul_(momentum_beta).add_((1 - momentum_beta), grad_hat)

        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))

        # Update precision matrix
        precision.mul_(beta).add_((1 - beta), ggn_hat + prior_prec)
        # Update mean vector
        mu.addcdiv_(-lr, momentum + torch.mul(mu - prior_mu, prior_prec), precision)
        # Update model parameters
        vector_to_parameters(self.state['mu'], self.param_groups[0]['params'])

        return loss, pred_list
