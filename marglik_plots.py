import numpy as np
import pickle
import matplotlib.pyplot as plt

import seaborn as sns
import brewer2mpl


plt.style.use('seaborn-white')
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 16
plt.rcParams['font.style'] = 'normal'
plt.rcParams['font.family'] = 'sans-serif'
plt.rc('text', usetex=True)
bmap = brewer2mpl.get_map('Set1', 'qualitative', 6)
colors = bmap.mpl_colors

vlap = colors[1]
clap = colors[0]
test = 'black'
train = 'darkgray'


def produce_paper_toy_plots(name):
    with open('results/reg_ms_delta_{name}.pkl'.format(name=name), 'rb') as f:
        res = pickle.load(f)


    mlhs = np.array([res['results'][i]['mlh'] for i in range(len(res['params']))])
    vimlhs = np.array([res['results'][i]['convimlh'] for i in range(len(res['params']))])
    testvi = np.array([res['results'][i]['test_loss_vi'] for i in range(len(res['params']))])
    testmap = np.array([res['results'][i]['test_loss_map'] for i in range(len(res['params']))])
    trainvi = np.array([res['results'][i]['train_loss_vi'] for i in range(len(res['params']))])
    trainmap = np.array([res['results'][i]['train_loss_map'] for i in range(len(res['params']))])
    n = len(res['datasets'])
    div = np.sqrt(n)
    deltas = np.array(res['params'])

    '''Marginal likelihood according to Laplace Theorem (1)'''
    fig, ax1 = plt.subplots(figsize=(4.9, 4))
    ax1.set_xscale('log')
    ylim = [0.05, 0.4]
    xlim = [1e-2, 1e2]

    ax1.plot(deltas, trainmap.mean(axis=1), label='train loss', linestyle='--', c=train, zorder=1)
    ax1.plot(deltas, testmap.mean(axis=1), label='test loss', linestyle='-', c=test, zorder=1)
    testlosses = testmap.mean(axis=1)

    m, s = trainmap.mean(axis=1), trainmap.std(axis=1)/div
    ax1.fill_between(deltas, m-s, m+s, color='gray', alpha=0.15)
    m, s = testmap.mean(axis=1), testmap.std(axis=1)/div
    ax1.fill_between(deltas, m-s, m+s, color='gray', alpha=0.15)
    ax1.legend(loc=(0.10, 0.78))
    ax1.set_ylabel('MSE')
    ax1.set_ylim(ylim)
    ax1.set_xlabel('hyperparameter $\delta$')
    ax1.set_xlim(xlim)
    ax1.set_yticks([0.1, 0.2, 0.3])

    ax2 = ax1.twinx()
    ylim = [107, 143]

    ax2.plot(deltas, -mlhs.mean(axis=1), color=clap, label='Train MargLik', linewidth=3, zorder=1)
    m, s = -mlhs.mean(axis=1), -mlhs.std(axis=1)/div
    ax2.fill_between(deltas, m-s, m+s, color=clap, alpha=0.15)
    lhss = -mlhs.mean(axis=1)
    ax2.grid(False)

    ax1.scatter([deltas[np.argmin(testlosses)]], [np.min(testlosses)], c='black', marker='*', linewidth=4, zorder=2)
    ax2.scatter([deltas[np.argmin(lhss)]], [np.min(lhss)], c='black', marker='*', linewidth=4, zorder=2)

    ax2.legend(loc=(0.10, 0.67))
    ax2.set_ylabel('-log marginal likelihood')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xscale('log')
    ax2.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig('figures/marglik_delta_toy_laplace.pdf')

    '''Marginal likelihood according to VI Theorem (2)'''
    fig, ax1 = plt.subplots(figsize=(4.9, 4))
    ax1.set_xscale('log')
    ylim = [0.05, 0.45]
    xlim = [1e-2, 1e2]
    ax1.plot(deltas, trainvi.mean(axis=1), label='train loss', linestyle='--', c=train, zorder=1)
    ax1.plot(deltas, testvi.mean(axis=1), label='test loss', linestyle='-', c=test, zorder=1)
    testlosses = testvi.mean(axis=1)
    m, s = trainvi.mean(axis=1), trainvi.std(axis=1)/div
    ax1.fill_between(deltas, m-s, m+s, color='gray', alpha=0.15)
    m, s = testvi.mean(axis=1), testmap.std(axis=1)/div
    ax1.fill_between(deltas, m-s, m+s, color='gray', alpha=0.15)
    ax1.legend(loc=(0.09, 0.78))
    ax1.set_ylabel('MSE')
    ax1.set_ylim(ylim)
    ax1.set_xlabel('hyperparameter $\delta$')
    ax1.set_xlim(xlim)
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax2 = ax1.twinx()
    ylim = [107, 143]

    ax2.plot(deltas, -vimlhs.mean(axis=1), color=vlap, label='Train MargLik', linewidth=3, zorder=1)
    m, s = -vimlhs.mean(axis=1), -mlhs.std(axis=1)/div
    ax2.fill_between(deltas, m-s, m+s, color=vlap, alpha=0.15)
    lhss = -vimlhs.mean(axis=1)
    ax2.grid(False)

    ax1.scatter([deltas[np.argmin(testlosses)]], [np.min(testlosses)], c='black', marker='*', linewidth=4, zorder=2)
    ax2.scatter([deltas[np.argmin(lhss)]], [np.min(lhss)], c='black', marker='*', linewidth=4, zorder=2)

    ax2.legend(loc=(0.09, 0.67))
    ax2.set_ylabel('-log marginal likelihood')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xscale('log')
    ax2.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig('figures/marglik_delta_toy_VI.pdf')


    '''Fits to the data with various deltas for insight on the type of fit.'''
    ixs = [0, 9, -4]
    x, y = res['datasets'][0]['x'], res['datasets'][0]['y']
    xt, yt = res['datasets'][0]['x_train'], res['datasets'][0]['y_train']
    fig, axs = plt.subplots(3, 1, figsize=(4, 4), sharex=True)

    ax, i = axs[0], ixs[0]
    ax.scatter(xt[:, 0], yt, alpha=0.2, color='black')
    fm, fstd = res['results'][i]['fits'][0]['map'], res['results'][i]['fits'][0]['gpcov']
    ax.plot(x[:, 0], fm, label='Laplace-GP', color=clap)
    ax.fill_between(x[:, 0], fm+fstd, fm-fstd, alpha=0.3, color=clap)
    ax.annotate('$\delta={delta:.2f}$'.format(delta=deltas[i]), xy=(-2.5, 1))

    ax, i = axs[1], ixs[1]
    ax.set_ylabel('y')
    ax.scatter(xt[:, 0], yt, alpha=0.2, color='black')
    fm, fstd = res['results'][i]['fits'][0]['map'], res['results'][i]['fits'][0]['gpcov']
    ax.plot(x[:, 0], fm, label='MAP', color=clap)
    ax.fill_between(x[:, 0], fm+fstd, fm-fstd, alpha=0.3, color=clap)
    ax.annotate('$\delta={delta:.2f}$'.format(delta=deltas[i]), xy=(-2.5, 1))


    ax, i = axs[2], ixs[2]
    ax.scatter(xt[:, 0], yt, alpha=0.2, color='black')
    fm, fstd = res['results'][i]['fits'][0]['map'], res['results'][i]['fits'][0]['gpcov']
    ax.plot(x[:, 0], fm, label='MAP', color=clap)
    ax.fill_between(x[:, 0], fm+fstd, fm-fstd, alpha=0.3, color=clap)
    ax.annotate('$\delta={delta:.2f}$'.format(delta=deltas[i]), xy=(-2.5, 1))
    ax.set_xlabel('x')

    plt.tight_layout()
    plt.savefig('figures/marglik_delta_toy_fits.pdf')


def produce_additional_toy_plots(name):
    with open('results/reg_ms_width_{name}.pkl'.format(name=name), 'rb') as f:
        res = pickle.load(f)

    mlhs = np.array([res['results'][i]['mlh'] for i in range(len(res['params']))])
    testmap = np.array([res['results'][i]['test_loss_map'] for i in range(len(res['params']))])
    trainmap = np.array([res['results'][i]['train_loss_map'] for i in range(len(res['params']))])

    n = len(res['datasets'])
    div = np.sqrt(n)
    len(res['params'])
    widths = np.array(res['params'])

    '''Marginal likelihood vs Width according to Laplace Theorem (1)'''
    fig, ax1 = plt.subplots(figsize=(4.9, 4))  # paper size
    ylim = [0.0, 0.4]
    xlim = [1, 1001]
    ax1.set_xscale('log')
    ax1.plot(widths, trainmap.mean(axis=1), label='train loss', linestyle='--', c=train, zorder=1)
    ax1.plot(widths, testmap.mean(axis=1), label='test loss', linestyle='-', c=test, zorder=1)
    testlosses = testmap.mean(axis=1)

    m, s = trainmap.mean(axis=1), trainmap.std(axis=1) / div
    ax1.fill_between(widths, m - s, m + s, color='gray', alpha=0.15)
    m, s = testmap.mean(axis=1), testmap.std(axis=1) / div
    ax1.fill_between(widths, m - s, m + s, color='gray', alpha=0.15)
    ax1.legend(loc=(0.10, 0.75))
    ax1.set_ylabel('MSE')
    ax1.set_ylim(ylim)
    ax1.set_xlabel('width')
    ax1.set_xlim(xlim)
    ax1.set_yticks([0.1, 0.2, 0.3])

    ax2 = ax1.twinx()
    ylim = [107, 143]

    ax2.set_xscale('log')
    ax2.plot(widths, -mlhs.mean(axis=1), color=clap, label='Train MargLik', linewidth=3, zorder=1)
    m, s = -mlhs.mean(axis=1), -mlhs.std(axis=1) / div
    ax2.fill_between(widths, m - s, m + s, color=clap, alpha=0.15)
    lhss = -mlhs.mean(axis=1)
    ax2.grid(False)

    ax1.scatter([widths[np.argmin(testlosses)]], [np.min(testlosses)], c='black', marker='*',
                linewidth=4, zorder=2)
    ax2.scatter([widths[np.argmin(lhss)]], [np.min(lhss)], c='black', marker='*', linewidth=4,
                zorder=2)
    gen_loss = np.abs(testmap.mean(axis=1) - trainmap.mean(axis=1))
    ix = np.argmin(gen_loss)
    ax2.scatter([widths[ix]], [testmap.mean(axis=1)[ix]], c='black', marker='*', linewidth=4,
                zorder=2)

    ax2.legend(loc=(0.10, 0.64))
    ax2.set_ylabel('-log marginal likelihood')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig('figures/marglik_width_toy_laplace.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Model selection experiment with result saving and MP.')
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    name = args.name
    # delta on the toy data set with Laplace and VI (used in paper)
    produce_paper_toy_plots(name)
    # width and depth on the toy data set with Laplace and VI (not in paper)
    produce_additional_toy_plots(name)
    # TODO: add UCI result plotting
