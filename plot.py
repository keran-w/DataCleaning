import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from itertools import cycle
from sklearn.metrics import roc_curve, auc

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


def plot_correlation(df, chinese=False, fillna=0, decimal=4):
    plot_config(chinese)
    sns.heatmap(df.corr().fillna(fillna), annot=True, cmap='Blues', fmt=f'.{decimal}g')
    plt.show()
    
def exploratory_data_analysis(df, title='', install=False, display=None, output_filename=None):
    import os
    if install:
        os.system('pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip')
    from pandas_profiling import ProfileReport
    try:
        profile = ProfileReport(df, title=title, html={'style':{'full_width':True}})
        if display == 'html':
            profile.to_file(f'{"" if not output_filename else output_filename}_EDA.html')
        elif display == 'colab':
            profile.to_notebook_iframe()
        else:
            profile
    except:
        raise np.ModuleDeprecationWarning('pandas_profiling error, try install=True')
        
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(y_true, y_score, plot_class=None):
    y_true = np.array(y_true).reshape(-1, 1)
    n_classes = len(np.unique(y_true))
    y_score = np.array(y_score)
    assert n_classes == y_score.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    lw = 2
    if plot_class is not None:
        plt.figure()
        plt.plot(
            fpr[plot_class],
            tpr[plot_class],
            color='darkorange',
            lw=lw,
            label='ROC curve (area = %0.2f)' % roc_auc[plot_class],
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve of class {plot_class}')
        plt.legend(loc='lower right')
        plt.show()
    else:
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve of class ")
        plt.legend(loc="lower right")
        plt.show()