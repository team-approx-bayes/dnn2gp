import torch
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader

from dnn2gp.neural_networks import LeNet5, LeNet5CIFAR
from dnn2gp.datasets import Dataset
from dnn2gp import compute_laplace, compute_dnn2gp_quantities, compute_kernel

torch.set_default_dtype(torch.double)
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(77)
np.random.seed(77)


def compute_kernel_and_predictive_laplace(model, loader, delta, fname):
    post_prec = compute_laplace(model, loader, delta, device)
    Jacobians, predictive_mean_GP, labels, predictive_var_f, predictive_noise, predictive_mean = \
        compute_dnn2gp_quantities(model, loader, device, limit=1000, post_prec=post_prec)
    labels = labels.numpy()
    strat_labels = list()
    for i in range(10):
        ixs = np.where(labels == i)[0]
        strat_labels.append(ixs[:30])
    strat_labels = np.hstack(strat_labels)
    Jacobians = Jacobians.numpy()[strat_labels]
    predictive_var_f = predictive_var_f.numpy()[strat_labels]
    predictive_mean_GP = predictive_mean_GP.numpy()[strat_labels]
    predictive_noise = predictive_noise.numpy()[strat_labels]
    predictive_mean = predictive_mean.numpy()[strat_labels]
    K = compute_kernel(Jacobians, agg_type='diag')  # one gp per class and sum variances

    np.save('results/{fname}_Laplace_gp_predictive_mean'.format(fname=fname), predictive_mean_GP)
    np.save('results/{fname}_Laplace_predictive_mean'.format(fname=fname), predictive_mean)
    np.save('results/{fname}_Laplace_predictive_var_f'.format(fname=fname), predictive_var_f)
    np.save('results/{fname}_Laplace_predictive_noise'.format(fname=fname), predictive_noise)
    np.save('results/{fname}_Laplace_kernel'.format(fname=fname), K)


def compute_kernel_predictive_VI(model, loader, fname):
    Jvm, Jthetavm, lbsvm = compute_dnn2gp_quantities(model, loader, device, limit=1000)
    lbsvm = lbsvm.numpy()
    strat_labels = list()
    for i in range(10):
        ixs = np.where(lbsvm == i)[0]
        strat_labels.append(ixs[:30])
    strat_labels = np.hstack(strat_labels)
    Jvm = Jvm.numpy()[strat_labels]
    mpred = Jthetavm.numpy()[strat_labels]
    K = np.einsum('ikp,jkp->ij', Jvm, Jvm)  # one gp per class and then sum

    np.save('results/{fname}_VI_gp_predictive_mean'.format(fname=fname), mpred)
    np.save('results/{fname}_VI_kernel'.format(fname=fname), K)


if __name__ == '__main__':
    ### MNIST
    transformations = transforms.Compose([transforms.ToTensor(), lambda x: x.double()])
    trainset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transformations)
    loader = DataLoader(trainset, batch_size=128, shuffle=True)

    model = LeNet5(num_classes=2).to(device)
    model.load_state_dict(torch.load('models/2_class_mnist_zone_lenet_vogn.tk', map_location=device))
    compute_kernel_predictive_VI(model, loader, 'BIN_MNIST')

    model = LeNet5(num_classes=10).to(device)
    # Adam with weight-decay = 1e-4
    model.load_state_dict(torch.load('models/full_mnist_lenet_adaml2.tk', map_location=device))
    compute_kernel_and_predictive_laplace(model, loader, 1e-4, 'MNIST')
    # for VOGN model, prior precision = 1e-4
    model.load_state_dict(torch.load('models/full_mnist_lenet_vogn.tk', map_location=device))
    compute_kernel_predictive_VI(model, loader, 'MNIST')

    ### CIFAR-10
    data = Dataset('cifar10')
    loader = data.get_train_loader(128)

    model = LeNet5CIFAR().to(device)
    # delta = weight_decay = 1e-2 for Adam
    model.load_state_dict(torch.load('models/cifar_lenet_adam.tk', map_location=device))
    compute_kernel_and_predictive_laplace(model, loader, 1e-2, 'CIFAR')
    # prior_precision = 1 for VOGN
    model.load_state_dict(torch.load('models/cifar_lenet_vogn.tk', map_location=device))
    compute_kernel_predictive_VI(model, loader, 'CIFAR')
