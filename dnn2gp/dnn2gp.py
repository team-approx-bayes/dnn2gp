import numpy as np
import torch
from tqdm import tqdm


def gradient(model):
    grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
    return grad.detach()


def weights(model):
    wts = torch.cat([p.flatten() for p in model.parameters()])
    return wts


def compute_kernel(Jacobians, agg_type='diag'):
    """Compute kernel by various aggregation types based on Jacobians"""
    if agg_type == 'diag':
        K = np.einsum('ikp,jkp->ij', Jacobians, Jacobians)  # one gp per class and then sum
    elif agg_type == 'sum':
        K = np.einsum('ikp,jlp->ij', Jacobians, Jacobians)  # sum kernel up
    elif agg_type == 'full':
        K = np.einsum('ikp,jlp->ijkl', Jacobians, Jacobians)  # full kernel NxNxKxK
    else:
        raise ValueError('agg_type not available')
    return K


def compute_dnn2gp_quantities(model, data_loader, device, limit=-1, post_prec=None):
    """ Compute reparameterized nn2gp quantities for softmax regression (multiclassification)
    :param model: pytorch function subclass with differentiable output
    :param data_loader: data iterator yielding tuples of features and labels
    :param device: device to do heavy compute on (saving and aggregation on CPU)
    :param limit: maximum number of data to iterate over
    :param post_prec: posterior precision (diagonal)
    """
    Jacobians = list()
    predictive_mean_GP = list()
    labels = list()
    predictive_var_f = list()
    predictive_noise = list()
    predictive_mean = list()
    theta_star = weights(model)
    n_points = 0
    for data, label in tqdm(data_loader):
        data, label = data.to(device).double(), label.to(device).double()
        prediction = model.forward(data)
        p = torch.softmax(prediction, -1).detach()
        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        y_uct = p - (p ** 2)
        for i in range(prediction.shape[0]):
            Jacs = list()
            kpreds = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                Jacs.append(Jij)
                with torch.no_grad():
                    kpreds.append(Jij @ theta_star)
                model.zero_grad()
            Jacs = torch.stack(Jacs)
            Jacobians.append(Jacs.to('cpu'))
            jtheta_star = torch.stack(kpreds).flatten()
            predictive_mean_GP.append(jtheta_star.to('cpu'))
            predictive_mean.append(p[i].to('cpu'))
            if post_prec is not None:
                f_uct = torch.diag(Lams[i] @ torch.einsum('kp,p,mp->km', Jacs, 1/post_prec, Jacs)
                                   @ Lams[i])
                predictive_var_f.append(f_uct.to('cpu'))
                predictive_noise.append(y_uct[i].to('cpu'))
        labels.append(label.to('cpu'))
        n_points += data_loader.batch_size
        if n_points >= limit > 0:
            break
    if post_prec is not None:
        return (torch.stack(Jacobians), torch.stack(predictive_mean_GP), torch.stack(labels).flatten(),
                torch.stack(predictive_var_f), torch.stack(predictive_noise), torch.stack(predictive_mean))
    return torch.stack(Jacobians), torch.stack(predictive_mean_GP), torch.stack(labels).flatten()


def compute_laplace(model, train_loader, prior_prec, device):
    """ Compute diagonal posterior precision due to Laplace approximation
    :param model: pytorch neural network
    :param train_loader: data iterator of training set with features and labels
    :param prior_prec: prior precision scalar
    :param device: device to compute/backpropagate on (ideally GPU)
    """
    theta_star = weights(model)
    post_prec = (torch.ones_like(theta_star) * prior_prec)
    for data, label in tqdm(train_loader):
        data, label = data.to(device).double(), label.to(device).double()
        prediction = model.forward(data)
        p = torch.softmax(prediction, -1).detach()
        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        Jacs = list()
        for i in range(prediction.shape[0]):
            Jac = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                Jac.append(Jij)
                model.zero_grad()
            Jac = torch.stack(Jac).t()
            Jacs.append(Jac)
        Jacs = torch.stack(Jacs).detach()
        post_prec += torch.einsum('npj,nij,npi->p', Jacs, Lams, Jacs)
    return post_prec
