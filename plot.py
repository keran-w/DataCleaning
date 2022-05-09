import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

from IPython import display

def plot_config(chinese=False):
    backend_inline.set_matplotlib_formats('svg')
    sns.set(style="whitegrid")
    sns.set_color_codes("pastel")
    plt.rcParams['axes.unicode_minus'] = False
    if chinese:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        sns.set(font='SimHei')
        sns.set_style({'font.sans-serif':['SimHei', 'Arial']})


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot_correlation(df, chinese=False, fillna=-1, decimal=4):
    plot_config(chinese)
    sns.heatmap(df.corr().fillna(fillna), annot=True,
                cmap='Blues', fmt=f'.{decimal}g')
    plt.show()
