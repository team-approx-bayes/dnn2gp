import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import iplot
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab10')
tab10 = lambda c: rgb2hex(cmap(c))


def plot_kernel(kernel, Y, Y_colors, fname, use_gl=False):
    N = len(Y)
    zmax = max(-kernel.min(), kernel.max())
    heatmap = go.Heatmapgl if use_gl else go.Heatmap
    data = [
        heatmap(
            x=np.arange(1, N + 1),
            y=np.arange(1, N + 1),
            z=kernel,
            zmin=-zmax,
            zmax=zmax,
            colorscale=[[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']],
            colorbar=dict(thickness=15, tickformat='e', exponentformat='e', showexponent='none')
        )
    ]
    shapes = []
    if Y_colors is not None:
        jump = 1 / N
        for idx, c in enumerate(Y):
            pos = idx / N
            line_h = {
                'type': 'line',
                'xref': 'paper',
                'yref': 'paper',
                'x0': pos,
                'y0': 1.01,
                'x1': pos + jump,
                'y1': 1.01,
                'line': {
                    'color': Y_colors(c),
                    'width': 10,
                },
            }
            line_v = {
                'type': 'line',
                'xref': 'paper',
                'yref': 'paper',
                'yanchor': -1,
                'x0': -0.01,
                'y0': 1 - pos,
                'x1': -0.01,
                'y1': 1 - (pos + jump),
                'line': {
                    'color': Y_colors(c),
                    'width': 10,
                },
            }
            shapes.append(line_h)
            shapes.append(line_v)
            pos += jump

    layout = go.Layout(
        font=dict(size=24.5),  # 19
        autosize=False,
        showlegend=False,
        width=655,
        height=595,
        xaxis=dict(
            title='data examples',
            autorange=True,
            zeroline=False,
            linecolor='black',
            mirror=True,
        ),
        yaxis=dict(
            title='data examples',
            autorange='reversed',
            ticklen=14,
            zeroline=False,
            mirror=True,
            linecolor='black',
        ),
        margin=go.layout.Margin(
            l=98,  # 115
            r=5,
            b=70,  # 95
            t=10,
            pad=0
        ),
        shapes=shapes
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)
    pio.write_image(fig, 'figures/{}_ker.pdf'.format(fname))


def plot_observations(jtheta, Y, classes, Y_colors, fname, use_gl=False, width=300, plot_var=False):
    N, K = jtheta.shape
    if not plot_var:
        zmin = 0
        zmax = 1
    else:
        zmin = 0
        zmax = jtheta.max() #0.25
    heatmap = go.Heatmapgl if use_gl else go.Heatmap
    data = [
        heatmap(
            x=classes,
            y=np.arange(1, N + 1),
            z=jtheta,
            zmin=zmin,
            zmax=zmax,
            colorscale=[[0.0, 'white'], [1.0, 'red']],
            colorbar=dict(thickness=15, exponentformat='e', showexponent='none')
        )
    ]
    shapes = []
    if Y_colors is not None:
        jump = 1 / N
        for idx, c in enumerate(Y):
            pos = idx / N
            line_v = {
                'type': 'line',
                'xref': 'paper',
                'yref': 'paper',
                'yanchor': -1,
                'x0': -0.02,
                'y0': 1 - pos,
                'x1': -0.02,
                'y1': 1 - (pos + jump),
                'line': {
                    'color': Y_colors(c),
                    'width': 10,
                },
            }
            shapes.append(line_v)
            pos += jump

    layout = go.Layout(
        font=dict(size=21.5),  # dict(size=24.5),
        autosize=False,
        showlegend=False,
        width=width,
        height=595,
        xaxis=dict(
            title='class',
            type='category',
            autorange=True,
            zeroline=False,
            linecolor='black',
            nticks=K,
            mirror=True
        ),
        yaxis=dict(
            title='data examples',
            autorange='reversed',
            zeroline=False,
            mirror=True,
            ticklen=12,
            linecolor='black',
        ),
        margin=go.layout.Margin(
            l=95,
            r=5,
            b=70,
            t=10,
            pad=0
        ),
        shapes=shapes
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)
    pio.write_image(fig, 'figures/{}_obs.pdf'.format(fname))


if __name__ == '__main__':
    K = np.load('results/CIFAR_Laplace_kernel.npy')
    Jtheta = np.load('results/CIFAR_Laplace_gp_predictive_mean.npy')
    Var_y = np.load('results/CIFAR_Laplace_gp_output_var.npy')
    Var_f = np.load('results/CIFAR_Laplace_gp_functional_var.npy')
    ps = np.load('results/CIFAR_Laplace_nn_predictive.npy')
    Ys = np.concatenate([np.repeat([i], 30) for i in range(10)])
    plot_kernel(K, Ys, tab10, 'cifar_laplace_kernel')
    plot_observations(ps, Ys, np.arange(10), tab10, 'cifar_laplace_pred_mean_ste', width=550)
    plot_observations(Var_f, Ys, np.arange(10), tab10, 'cifar_laplace_var_f', width=450,
                      plot_var=True)
    plot_observations(Var_y, Ys, np.arange(10), tab10, 'cifar_laplace_var_y', width=450,
                      plot_var=True)
