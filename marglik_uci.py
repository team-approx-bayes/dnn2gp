import numpy as np
import pickle
from tqdm import tqdm
from itertools import repeat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from multiprocessing import Pool

from dnn2gp.laplace_models import NeuralNetworkRegression
from dnn2gp.dual_models import DualGPRegression

import os
import vadam
from vadam.datasets import Dataset


def get_uci_dataset(dataset_name):
    data_path = os.path.join(vadam.__path__[0], 'data')
    ds = []
    for i in range(20):
        dataset = Dataset(data_set=dataset_name + str(i), data_folder=data_path)
        x_train, y_train = dataset.load_full_train_set(use_cuda=False)
        x_test, y_test = dataset.load_full_test_set(use_cuda=False)
        y_train = y_train.unsqueeze(1).double().numpy()
        y_test = y_test.unsqueeze(1).double().numpy()
        x_train = x_train.double().numpy()
        x_test = x_test.double().numpy()
        scl = StandardScaler(copy=False)
        scl.fit_transform(x_train)
        scl.transform(x_test)
        y_scl = StandardScaler(copy=False)
        y_scl.fit_transform(y_train)
        y_scl.transform(y_test)
        ds.append((None, None, x_train, y_train.squeeze(), x_test, y_test.squeeze(), y_scl.scale_[0]))
    return ds


def compute_marglik(delta, Ds, hidden_size, n_layers, sn, n_epochs=1000):
    dres = {'fits': list(), 'mlh': list(), 'vimlh': list(), 'convimlh': list(),
            'test_loss_map': list(), 'test_loss_lap': list(), 'test_loss_vi': list(),
            'train_loss_map': list(), 'train_loss_lap': list(), 'train_loss_vi': list()}
    for j, (x, y, x_train, y_train, x_test, y_test, y_scale) in enumerate(Ds):
        print((delta, hidden_size, sn), '{}/{}'.format(j+1, len(Ds)))
        primal_nn = NeuralNetworkRegression(x_train, y_train, delta, sigma_noise=sn, n_epochs=n_epochs,
                                            hidden_size=hidden_size,
                                            n_layers=n_layers, diagonal=True, n_samples_pred=1, step_size=0.05, lr_factor=0.8)
        m_0, S_0 = np.zeros(primal_nn.d), 1 / delta * np.eye(primal_nn.d)
        (Us, Ss), vs = primal_nn.UsSs('J'), primal_nn.vs('J')
        x_hat, y_hat, s_noise = Us, Us @ primal_nn.theta_star - vs / Ss, 1 / np.sqrt(Ss)
        dual_gp = DualGPRegression(x_hat, y_hat, s_noise, m_0, S_0=S_0, comp_post=True)
        map_pred_train = primal_nn.predictive_map(x_train)
        map_pred_test = primal_nn.predictive_map(x_test)
        dres['mlh'].append(dual_gp.log_marginal_likelihood())
        dres['test_loss_map'].append(y_scale**2 * mean_squared_error(y_test, map_pred_test))
        dres['train_loss_map'].append(y_scale**2 * mean_squared_error(y_train, map_pred_train))
    return dres


def reg_ms_delta(hidden_size=20, n_layers=3, n_params=50, fname=''):
    sn = 0.64
    n_epochs = 1000
    Ds = get_uci_dataset('wine')
    deltas = list(np.logspace(-1, 3.6, n_params))
    res = {'datasets': Ds, 'params': deltas}
    parameters = [(delta, datasets, hidden_size, n_layers, sn, n_epochs) for delta, datasets in
                  zip(deltas, repeat(Ds))]
    with Pool(processes=mp.cpu_count()-1) as p:
        metric_results = tqdm(p.starmap(compute_marglik, parameters), total=len(deltas))

    res['results'] = list(metric_results)

    with open('results/reg_ms_delta_{fname}.pkl'.format(fname=fname), 'wb') as f:
        pickle.dump(res, f)
    return res


def reg_ms_sigma(hidden_size=20, n_layers=3, n_params=50, fname=''):
    n_epochs = 1000
    delta = 30
    Ds = get_uci_dataset('wine')
    sns = list(np.logspace(-1, 1, n_params))
    res = {'datasets': Ds, 'params': sns}
    parameters = [(delta, datasets, hidden_size, n_layers, sn, n_epochs) for sn, datasets in
                  zip(sns, repeat(Ds))]
    with Pool(processes=mp.cpu_count()-1) as p:
        metric_results = tqdm(p.starmap(compute_marglik, parameters), total=len(sns))

    res['results'] = list(metric_results)

    with open('results/reg_ms_sigma_{fname}.pkl'.format(fname=fname), 'wb') as f:
        pickle.dump(res, f)
    return res

def reg_ms_width(n_layers=2, n_params=50, fname=''):
    sn = 0.64
    delta = 3
    n_epochs = 1500
    Ds = get_uci_dataset('wine')
    hidden_sizes = np.unique(np.logspace(0, 2.177, n_params).astype(int))
    res = {'datasets': Ds, 'params': hidden_sizes}
    parameters = [(delta, datasets, hidden_size, n_layers, sn, n_epochs) for hidden_size, datasets in
                  zip(hidden_sizes, repeat(Ds))]
    with Pool(processes=mp.cpu_count()-1) as p:
        metric_results = tqdm(p.starmap(compute_marglik, parameters), total=len(hidden_sizes))

    res['results'] = list(metric_results)

    with open('results/reg_ms_width_{fname}.pkl'.format(fname=fname), 'wb') as f:
        pickle.dump(res, f)
    return res


if __name__ == '__main__':
    import argparse

    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Model selection experiment with result saving and MP.')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--n_params', type=int, default=30)
    parser.add_argument('--tune_param', type=str, default='delta')
    args = parser.parse_args()
    if args.tune_param == 'delta':
        reg_ms_delta(fname=args.name, n_params=args.n_params)
    elif args.tune_param == 'width':
        reg_ms_width(fname=args.name, n_params=args.n_params)
    elif args.tune_param == 'sigma':
        reg_ms_sigma(fname=args.name, n_params=args.n_params)
    elif args.tune_param == 'all':
        reg_ms_width(fname=args.name, n_params=args.n_params)
        reg_ms_delta(fname=args.name, n_params=args.n_params)
        reg_ms_sigma(fname=args.name, n_params=args.n_params)
    else:
        raise ValueError(args.tune_param)
