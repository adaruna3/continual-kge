import os
from copy import copy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy import integrate
from math import isnan
from argparse import ArgumentParser

# for stats tests
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import statsmodels.stats.multicomp as multi

# for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, RegularPolygon, Ellipse
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgba

# for terminal logging
from logger.terminal_utils import logout


import pdb


class AbstractProcessorViz:
    def __init__(self, args):
        log_name = str(args.tag) + "__"
        log_name += str(args.dataset) + "_"
        log_name += "mt" + str(args.model) + "_"
        log_name += "clm" + str(args.cl_method)
        log_dir = os.path.abspath(os.path.dirname(__file__)) + "/logs/"
        self.log_fp = log_dir + log_name


class ProcessorViz(AbstractProcessorViz):
    def __init__(self, args):
        super(ProcessorViz, self).__init__(args)
        if os.path.isdir(self.log_fp):  # overwrites existing events log
            files = os.listdir(self.log_fp)
            for filename in files:
                if "events" in filename:
                    os.remove(self.log_fp+"/"+filename)
                # rmtree(self.log_fp)
        self._writer = SummaryWriter(self.log_fp)
        self.timestamp = 0
        self.gruvae_timestamp = 0

    def add_tr_sample(self, sess, sample):
        loss = sample
        self._writer.add_scalar("Loss/TrainSess_"+str(sess), loss, self.timestamp)
        self.timestamp += 1

    def add_de_sample(self, sample):
        hits_avg = 0.0
        mrr_avg = 0.0
        for sess in range(sample.shape[0]):
            hits, mrr = sample[sess,:]
            self._writer.add_scalar("HITS/DevSess_"+str(sess), hits, self.timestamp)
            self._writer.add_scalar("MRR/DevSess_"+str(sess), mrr, self.timestamp)
            hits_avg += hits
            mrr_avg += mrr
        hits_avg = hits_avg / float(sample.shape[0])
        mrr_avg = mrr_avg / float(sample.shape[0])
        self._writer.add_scalar("HITS/DevAvg", hits_avg, self.timestamp)
        self._writer.add_scalar("MRR/DevAvg", mrr_avg, self.timestamp)

    def add_gruvae_tr_sample(self, sample):
        total_loss, rc_loss, kl_loss, kl_weight = sample
        self._writer.add_scalar("GRUVAE/Loss", total_loss, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/RCLoss", rc_loss, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/KLWeight", kl_weight, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/KLLoss", kl_loss, self.gruvae_timestamp)
        self.gruvae_timestamp += 1

    def add_gruvae_de_sample(self, sample):
        precision, u_precision, coverage = sample[0]
        self._writer.add_scalar("GRUVAE/Precision", precision, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/UPrecision", u_precision, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/Coverage", coverage, self.gruvae_timestamp)


def plot_bar(values, names, colors=None, ylabel=None, title=None, ylim=None, yerr=None):
    fig, ax = plt.subplots(1, 1)
    bar = ax.bar(x=range(len(values)), height=values, color=colors, yerr=yerr)
    ax.get_xaxis().set_visible(False)
    ax.legend(bar, names,
              loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, fancybox=True, shadow=True)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig


def plot_mbar(values, names, colors, hatches, ylabel=None, titles=None,
              top_title=None, ylim=None, yerr=None):
    """

    :param values: num groups x num methods data
    :param names:
    :param colors:
    :param hatches:
    :param ylabel:
    :param titles:
    :param top_title:
    :param ylim:
    :param yerr:
    :return:
    """
    fig, ax = plt.subplots(1, values.shape[0])
    for i in range(values.shape[0]):
        bars = ax[i].bar(x=range(len(values[i])), height=values[i],
                        color=colors[i] if type(colors[0]) == list else colors,
                        alpha=.99,
                        yerr=yerr[i] if yerr is not None else None)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax[i].get_xaxis().set_visible(False)
        if i == round(float(len(values)) / 2.0):
            ax[i].legend(bars, names[i] if type(names[0]) == list else names,
                         loc='lower center', bbox_to_anchor=(0.5, -0.17),
                         ncol=4, fancybox=True, shadow=True)

        if ylim is not None:
            ax[i].set_ylim(ylim)
        if i == 0 and ylabel is not None:
            ax[i].set_ylabel(ylabel)
        if i != 0:
            ax[i].get_yaxis().set_visible(False)
        if titles is not None:
            ax[i].set_title(titles[i])

    if top_title is not None:
        fig.suptitle(top_title)

    return fig


def plot_mbar_stacked(values1, values2, names, colors, hatches, ylabel=None, titles=None,
              top_title=None, ylim=None, yerr1=None, yerr2=None):
    """

    :param values: num groups x num methods data
    :param names:
    :param colors:
    :param hatches:
    :param ylabel:
    :param titles:
    :param top_title:
    :param ylim:
    :param yerr:
    :return:
    """
    fig, ax = plt.subplots(1, values1.shape[0])
    for i in range(values1.shape[0]):
        bars = ax[i].bar(x=range(len(values1[i])), height=values1[i],
                         color=colors[i] if type(colors[0]) == list else colors,
                         alpha=.99,
                         yerr=yerr1[i] if yerr1 is not None else None)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax[i].get_xaxis().set_visible(False)
        if i == round(float(len(values1)) / 2.0):
            ax[i].legend(bars, names[i] if type(names[0]) == list else names,
                         loc='lower center', bbox_to_anchor=(0.5, -0.17),
                         ncol=4, fancybox=True, shadow=True)
        # stacked bars
        bars = ax[i].bar(x=range(len(values1[i])), height=values2[i]-values1[i],
                         bottom=values1[i],
                         color=colors[i] if type(colors[0]) == list else colors,
                         alpha=.30,
                         yerr=yerr2[i] if yerr2 is not None else None)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        if ylim is not None:
            ax[i].set_ylim(ylim)
        if i == 0 and ylabel is not None:
            ax[i].set_ylabel(ylabel)
        if i != 0:
            ax[i].get_yaxis().set_visible(False)
        if titles is not None:
            ax[i].set_title(titles[i])

    if top_title is not None:
        fig.suptitle(top_title)

    return fig


def plot_line(xvalues, yvalues, names, colors, linestyles,
              ylabel=None, titles=None, ylim=None, yerr=None,
              xticks=None, top_title=None):

    num_lines = yvalues.shape[0]

    fig = plt.figure(figsize=(4.25, 4))

    ax = fig.add_subplot(1, 1, 1)
    lines = []
    for j in range(num_lines):
        line, = ax.plot(xvalues, yvalues[j], color=colors[j], linestyle=linestyles[j])
        if yerr is not None:
            ax.fill_between(xvalues, yvalues[j] - yerr[j], yvalues[j] + yerr[j],
                            color=colors[j], alpha=0.2)
        lines.append(line)

    ax.legend(lines, names,
              loc='upper left',
              ncol=1, fancybox=True, shadow=True)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xlim([xticks[0][0], xticks[0][-1]])
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1])

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if top_title is not None:
        fig.suptitle(top_title, x=0.5, y=0.99)

    return fig


def plot_mline(xvalues, yvalues, names, colors, linestyles,
               ylabel=None, titles=None, ylim=None, yerr=None,
               xticks=None, top_title=None):
    num_plots = xvalues.shape[0]
    num_lines = []
    for i in range(yvalues.shape[0]):
        num_lines.append(yvalues[i].shape[0])

    fig = plt.figure(figsize=(10, 6))

    if ylabel is not None:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax = []
    for i in range(num_plots):
        ax.append(fig.add_subplot(num_plots, 1, i+1))
        lines = []
        for j in range(num_lines[i]):
            line, = ax[i].plot(xvalues[i], yvalues[i,j], color=colors[j], linestyle=linestyles[j])
            if yerr is not None:
                ax[i].fill_between(xvalues[i], yvalues[i, j] - yerr[i, j], yvalues[i, j] + yerr[i, j],
                                   color=colors[j], alpha=0.2)
            lines.append(line)


        if i == 0:
            ax[i].legend(lines, names,
                      loc='upper center', bbox_to_anchor=(0.5, 1.64),
                      ncol=4, fancybox=True)

        if titles is not None:
            ax[i].set_ylabel(titles[i])
            ax[i].yaxis.set_label_position("right")

        if i == num_plots-1:
            ax[i].get_xaxis().set_visible(True)
        else:
            ax[i].get_xaxis().set_visible(False)

        if ylim is not None:
            ax[i].set_ylim(ylim)

        if xticks is not None:
            ax[i].set_xlim([xticks[0][0], xticks[0][-1]])
            ax[i].set_xticks(xticks[0])
            ax[i].set_xticklabels(xticks[1])

    if top_title is not None:
        fig.suptitle(top_title, x=0.5, y=0.99)

    fig.subplots_adjust(hspace=0.07)

    return fig


def plot_table(stats, row_labels, col_labels, title=None):
    fig = plt.figure(figsize=(10, 6))
    axs = fig.add_subplot(1, 1, 1)
    fig.patch.set_visible(False)
    axs.axis('off')
    axs.axis('tight')
    plt.grid('off')

    format_stats = copy(stats).astype(str)
    for i in range(format_stats.shape[0]):
        for j in range(format_stats.shape[1]):
            format_stats[i,j] = "{:.4f}".format(stats[i,j])

    the_table = axs.table(cellText=format_stats, rowLabels=row_labels, colLabels=col_labels, loc='center')
    fig.tight_layout()
    if title is not None:
        axs.set_title(title, weight='bold', size='medium',
                      horizontalalignment='center', verticalalignment='center')
    return fig


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_radar(stats, colors, linestyles, metric_labels, method_labels, title):
    N = len(metric_labels)
    theta = radar_factory(N, frame='circle')

    spoke_labels = metric_labels

    fig, ax = plt.subplots(figsize=(4, 4), nrows=1, ncols=1,
                           subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=95)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.2),
                 horizontalalignment='center', verticalalignment='center')

    for idx in range(stats.shape[0]):
        ax.plot(theta, stats[idx, :], color=colors[idx], linestyle=linestyles[idx])
        ax.fill(theta, stats[idx, :], facecolor=colors[idx], alpha=0.25)
    ax.set_varlabels(spoke_labels)

    legend = ax.legend(method_labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small',
                       fancybox=True, shadow=True)
    return fig


def plot_scatter(xvalues, yvalues, names, colors, linestyles,
                 xlabel=None, ylabel=None,
                 xerr=None, yerr=None, top_title=None):
    ells = [Ellipse((xvalues[i], yvalues[i]),
                    width=xerr[0, i] if xerr is not None else 0.03,
                    height=yerr[0, i] if yerr is not None else 0.03,
                    angle=0) for i in range(len(xvalues))]

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for i in range(len(ells)):
        ells[i].set_clip_box(ax.bbox)
        ells[i].set_facecolor(to_rgba(colors[i], 0.3))
        ells[i].set_edgecolor(to_rgba(colors[i], 1.0))
        ells[i].set_linestyle(linestyles[i])
        ells[i].set_linewidth(1.5)
        ax.add_artist(ells[i])
        ax.scatter(xvalues[i], yvalues[i], c=to_rgba(colors[i], 1.0), s=1.0)

    ax.legend(ells, names,
              loc='center right', bbox_to_anchor=(1.27, 0.5),
              ncol=1, fancybox=True, shadow=True)


    ax.set_xlim([0.0, np.max(xvalues)+0.05])
    ax.set_ylim([0.0, np.max(yvalues)+0.05])

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if top_title is not None:
        ax.set_title(top_title)

    return fig


def figs2pdf(figs, filepath):
    pdf = PdfPages(filepath)
    for fig in figs:
        pdf.savefig(fig, bbox_inches="tight")
    pdf.close()


def csvlogs2plots_format_inference(filepath):
    logout("Loading data for :" + str(filepath))
    num_sessions = 5
    stage2idx = {"t": 1}
    metrics = np.zeros(shape=(2, 2, num_sessions, num_sessions))
    with open(filepath, "r") as f:
        for line in f:
            parsed_line = line.strip().split(",")

            if parsed_line[1] == "t":
                inference_values = np.asarray([float(value) for value in parsed_line[5:-1]]).reshape((5, 2))
                for i_row in range(inference_values.shape[0]):
                    for i_col in range(inference_values.shape[1]):
                        metrics[stage2idx[parsed_line[1]], i_col, i_row, int(parsed_line[0])] = \
                            inference_values[i_row, i_col]
    return metrics


def csvlogs2plots_format_noninference(filepath):
    logout("Loading data for :" + str(filepath))
    num_sessions = 5
    stage2idx = {"i": 0, "f": 1}
    LCs = []
    LC = np.ndarray(shape=(0, 2))
    conv = np.ndarray(shape=(1, 0))
    model_sizes = np.ndarray(shape=(1, 0))
    sample_sizes = np.ndarray(shape=(1, 0))
    gruvae_conv = np.ndarray(shape=(1, 0))
    gruvae_model_sizes = np.ndarray(shape=(1, 0))
    metrics = np.zeros(shape=(2, 2, num_sessions, num_sessions))
    with open(filepath, "r") as f:
        for line in f:
            parsed_line = line.strip().split(",")

            if parsed_line[1] == "g":
                gruvae_conv = np.append(gruvae_conv, [[float(parsed_line[2])]], axis=1)
                gruvae_model_sizes = np.append(gruvae_model_sizes, [[max(0.0, float(parsed_line[4]))]], axis=1)

            if parsed_line[1] == "f":
                conv = np.append(conv, [[float(parsed_line[2])]], axis=1)
                sample_sizes = np.append(sample_sizes, [[float(parsed_line[3])]], axis=1)
                model_sizes = np.append(model_sizes, [[max(0.0, float(parsed_line[4]))]], axis=1)

            if parsed_line[1] == "f" or parsed_line[1] == "i":
                inference_values = np.asarray([float(value) for value in parsed_line[5:-1]]).reshape((5, 2))
                for i_row in range(inference_values.shape[0]):
                    for i_col in range(inference_values.shape[1]):
                        metrics[stage2idx[parsed_line[1]], i_col, i_row, int(parsed_line[0])] = \
                            inference_values[i_row, i_col]

            if parsed_line[1] == "f" or parsed_line[1] == "i" or parsed_line[1] == "s":
                sess = int(parsed_line[0])
                epoch = int(parsed_line[2])
                value = float(parsed_line[6 + sess * 2])
                LC = np.append(LC, [[epoch, value]], axis=0)
                if parsed_line[1] == "f":
                    if "DGR" in filepath:  # accounts for epochs and memory taken by generative model
                        LC[:, 0] += gruvae_conv[0, len(LCs)]
                        init_value = copy(LC[0, 1])
                        LC = np.insert(LC, 0, [[0, init_value]], axis=0)
                    LCs.append(copy(LC))
                    LC = np.ndarray(shape=(0, 2))

    if "DGR" in filepath: # accounts for epochs and memory taken by generative model
        conv = conv + gruvae_conv
        model_sizes[0, 1:] = model_sizes[0, 1:] + gruvae_model_sizes[0, 1:]

    return metrics, conv, LCs, model_sizes, sample_sizes


def format_method_names(methods):
    method_names = []
    method2name = {
        "offline": "Batch",
        "finetune": "Finetune",
        "SI": "SI",
        "L2": "L2",
        "PNN": "PNN",
        "CWR": "CWR",
        "DGR": "DGR"
    }
    for method in methods:
        method_names.append(method2name[method])
    return method_names


def format_method_colors(methods):
    method_colors = []
    method2color = {
        "offline": "m",
        "finetune": "m",
        "SI": "b",
        "L2": "b",
        "PNN": "g",
        "CWR": "g",
        "DGR": "y",
    }
    for method in methods:
        method_colors.append(method2color[method])
    return method_colors


def format_method_linestyles(methods):
    method_markers = []
    method2marker = {
        "offline": ":",
        "finetune": "--",
        "SI": ":",
        "L2": "--",
        "PNN": ":",
        "CWR": "--",
        "DGR": ":",
    }
    for method in methods:
        method_markers.append(method2marker[method])
    return method_markers


def format_method_hatches(methods):
    method_markers = []
    method2marker = {
        "offline": "//",
        "finetune": None,
        "SI": None,
        "L2": "//",
        "PNN": "//",
        "CWR": None,
        "DGR": "//",
    }
    for method in methods:
        method_markers.append(method2marker[method])
    return method_markers


def extract_runs_avg_std(datasets, models, methods, num_of_exp=5, num_sess=5):
    summary_num_metrics = 11
    num_metrics = 7
    # avgs
    avg_conv__ = np.ndarray(shape=(0, num_sess, len(methods)))
    avg_mrr_i__ = np.ndarray(shape=(0, num_sess, len(methods)))
    avg_mrr_f__ = np.ndarray(shape=(0, num_sess, len(methods)))
    avg_hit_i__ = np.ndarray(shape=(0, num_sess, len(methods)))
    avg_hit_f__ = np.ndarray(shape=(0, num_sess, len(methods)))
    avg_stats__ = np.ndarray(shape=(0, summary_num_metrics, len(methods)))
    avg_mrr_stats__ = np.ndarray(shape=(0, len(methods), num_metrics))
    avg_hit_stats__ = np.ndarray(shape=(0, len(methods), num_metrics))
    # errs
    std_conv__ = np.ndarray(shape=(0, num_sess, len(methods)))
    std_mrr_i__ = np.ndarray(shape=(0, num_sess, len(methods)))
    std_mrr_f__ = np.ndarray(shape=(0, num_sess, len(methods)))
    std_hit_i__ = np.ndarray(shape=(0, num_sess, len(methods)))
    std_hit_f__ = np.ndarray(shape=(0, num_sess, len(methods)))
    std_stats__ = np.ndarray(shape=(0, summary_num_metrics, len(methods)))
    std_mrr_stats__ = np.ndarray(shape=(0, len(methods), num_metrics))
    std_hit_stats__ = np.ndarray(shape=(0, len(methods), num_metrics))

    for dataset in datasets:
        for model in models:
            if dataset == "WN18RR":
                num_triples = 86835
            elif dataset == "FB15K237":
                num_triples = 272115
            elif dataset == "THOR_U":
                num_triples = 1580
            else:
                logout("Dataset not recognized for result generation", "f")
                exit()

            # accumulates the metrics
            conv_ = np.ndarray(shape=(0, num_sess, len(methods)))
            mrr_i_ = np.ndarray(shape=(0, num_sess, len(methods)))
            mrr_f_ = np.ndarray(shape=(0, num_sess, len(methods)))
            hit_i_ = np.ndarray(shape=(0, num_sess, len(methods)))
            hit_f_ = np.ndarray(shape=(0, num_sess, len(methods)))
            stats_ = np.ndarray(shape=(0, summary_num_metrics, len(methods)))
            mrr_stats_ = np.ndarray(shape=(0, len(methods), num_metrics))
            hit_stats_ = np.ndarray(shape=(0, len(methods), num_metrics))

            for exp_num in range(1, num_of_exp+1):
                conv = np.ndarray(shape=(0, num_sess))
                avg_mrr_f = np.ndarray(shape=(0, num_sess))
                avg_mrr_i = np.ndarray(shape=(0, num_sess))
                avg_hit_f = np.ndarray(shape=(0, num_sess))
                avg_hit_i = np.ndarray(shape=(0, num_sess))
                mrr_acc = []
                hits_acc = []
                mrr_fwt = []
                hits_fwt = []
                mrr_rem = []
                hits_rem = []
                mrr_pbwt = []
                hits_pbwt = []
                ms = []
                sss = []
                lca = []

                # must be accounted for bc SI allocates variables before initial learning session so not in memory sizes
                l2_initial_size = 0.0

                # gather logged data for the plot
                filepath_root = os.path.abspath(os.path.dirname(__file__)) + "/logs/continual_setting__" + dataset + "_mt" + model + "_"
                for method in methods:
                    method_str = "clm" + method
                    filepath = filepath_root + method_str + "/test_" + str(exp_num) + ".csv"
                    inf_f = csvlogs2plots_format_inference(filepath)
                    filepath = filepath_root + method_str + "/performances_" + str(exp_num) + ".csv"
                    inf, run_conv, lcs, model_sizes, sample_sizes = csvlogs2plots_format_noninference(filepath)
                    inf[1, 1, :, :] = inf_f[1, 1, :, :]
                    inf[1, 0, :, :] = inf_f[1, 0, :, :]
                    avg_mrr_i = np.append(avg_mrr_i, [np.average(np.triu(inf[0, 1, :, :]), axis=0)], axis=0)
                    avg_mrr_f = np.append(avg_mrr_f, [np.average(np.triu(inf[1, 1, :, :]), axis=0)], axis=0)
                    avg_hit_i = np.append(avg_hit_i, [np.average(np.triu(inf[0, 0, :, :]), axis=0)], axis=0)
                    avg_hit_f = np.append(avg_hit_f, [np.average(np.triu(inf[1, 0, :, :]), axis=0)], axis=0)
                    conv = np.append(conv, run_conv, axis=0)
                    # ACC & FWT
                    mrr_f_T = inf[1, 1, :, :].T
                    hit_f_T = inf[1, 0, :, :].T
                    mrr_acc.append("{:.4f}".format(np.sum(np.tril(mrr_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
                    hits_acc.append("{:.4f}".format(np.sum(np.tril(hit_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
                    mrr_fwt.append("{:.4f}".format(np.sum(np.triu(mrr_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
                    hits_fwt.append("{:.4f}".format(np.sum(np.triu(hit_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
                    # BWT+ & REM
                    mrr_bwt = 0.0
                    hit_bwt = 0.0
                    for i in range(1, mrr_f_T.shape[0]):
                        for j in range(i):
                            mrr_bwt = mrr_f_T[i, j] - mrr_f_T[j, j]
                    for i in range(1, hit_f_T.shape[0]):
                        for j in range(i):
                            hit_bwt = hit_f_T[i, j] - hit_f_T[j, j]
                    mrr_bwt = mrr_bwt / ((num_sess * (num_sess - 1)) / 2.0)
                    hit_bwt = hit_bwt / ((num_sess * (num_sess - 1)) / 2.0)
                    mrr_rem.append("{:.4f}".format(1.0 - np.absolute(np.min([0, mrr_bwt]))))
                    mrr_pbwt.append("{:.4f}".format(np.max([0, mrr_bwt])))
                    hits_rem.append("{:.4f}".format(1.0 - np.absolute(np.min([0, hit_bwt]))))
                    hits_pbwt.append("{:.4f}".format(np.max([0, hit_bwt])))
                    # MS & SSS
                    if "L2" in filepath:
                        l2_initial_size = copy(float(model_sizes[0, 0]))
                    if "SI" in filepath:
                        if l2_initial_size == 0.0:
                            logout("L2 inital size is wrong.", "w")
                        model_sizes[0, 0] = l2_initial_size
                    ms.append("{:.4f}".format(np.min([1.0, np.average(model_sizes[0, 0] / model_sizes)])))
                    sss.append("{:.4f}".format(1.0 - np.min([1.0, np.average(sample_sizes / num_triples)])))
                    # LCA
                    LCA_fracs = []
                    for lc in lcs:
                        best_value = lc[-1, 1]
                        best_value_idx = int(np.argwhere(lc[:, 1] == best_value)[0])
                        to_best_value_curve = lc[:best_value_idx+1, :]
                        x = to_best_value_curve[:, 0]
                        y = to_best_value_curve[:, 1]
                        normalize_y = np.ones_like(y) * best_value
                        frac = integrate.trapz(x=x, y=y) / integrate.trapz(x=x, y=normalize_y)
                        if isnan(frac):
                            frac = 1.0
                        LCA_fracs.append(frac)
                    lca.append("{:.4f}".format(np.average(LCA_fracs)))

                # perform final data transformations
                conv = np.transpose(conv)
                avg_mrr_i = np.transpose(avg_mrr_i) * 100.0
                avg_mrr_f = np.transpose(avg_mrr_f) * 100.0
                avg_hit_i = np.transpose(avg_hit_i) * 100.0
                avg_hit_f = np.transpose(avg_hit_f) * 100.0
                stats = copy(np.stack((mrr_acc, hits_acc, mrr_fwt, hits_fwt, mrr_pbwt, hits_pbwt, mrr_rem, hits_rem, ms, sss, lca)))
                mrr_stats = copy(np.stack((mrr_acc, mrr_fwt, mrr_pbwt, mrr_rem, ms, sss, lca))).astype(float).T
                hit_stats = copy(np.stack((hits_acc, hits_fwt, hits_pbwt, hits_rem, ms, sss, lca))).astype(float).T

                # append to the averaging arrays
                conv_ = np.append(conv_, [conv], axis=0)
                mrr_i_ = np.append(mrr_i_, [avg_mrr_i], axis=0)
                mrr_f_ = np.append(mrr_f_, [avg_mrr_f], axis=0)
                hit_i_ = np.append(hit_i_, [avg_hit_i], axis=0)
                hit_f_ = np.append(hit_f_, [avg_hit_f], axis=0)
                stats_ = np.append(stats_, [stats.astype(float)], axis=0)
                mrr_stats_ = np.append(mrr_stats_, [mrr_stats], axis=0)
                hit_stats_ = np.append(hit_stats_, [hit_stats], axis=0)

            avg_conv__ = np.append(avg_conv__, [np.average(conv_, axis=0)], axis=0)
            avg_mrr_i__ = np.append(avg_mrr_i__, [np.average(mrr_i_, axis=0)], axis=0)
            avg_mrr_f__ = np.append(avg_mrr_f__, [np.average(mrr_f_, axis=0)], axis=0)
            avg_hit_i__ = np.append(avg_hit_i__, [np.average(hit_i_, axis=0)], axis=0)
            avg_hit_f__ = np.append(avg_hit_f__, [np.average(hit_f_, axis=0)], axis=0)
            avg_stats__ = np.append(avg_stats__, [np.average(stats_, axis=0)], axis=0)
            avg_mrr_stats__ = np.append(avg_mrr_stats__, [np.average(mrr_stats_, axis=0)], axis=0)
            avg_hit_stats__ = np.append(avg_hit_stats__, [np.average(hit_stats_, axis=0)], axis=0)
            std_conv__ = np.append(std_conv__, [np.std(conv_, axis=0)], axis=0)
            std_mrr_i__ = np.append(std_mrr_i__, [np.std(mrr_i_, axis=0)], axis=0)
            std_mrr_f__ = np.append(std_mrr_f__, [np.std(mrr_f_, axis=0)], axis=0)
            std_hit_i__ = np.append(std_hit_i__, [np.std(hit_i_, axis=0)], axis=0)
            std_hit_f__ = np.append(std_hit_f__, [np.std(hit_f_, axis=0)], axis=0)
            std_stats__ = np.append(std_stats__, [np.std(stats_, axis=0)], axis=0)
            std_mrr_stats__ = np.append(std_mrr_stats__, [np.std(mrr_stats_, axis=0)], axis=0)
            std_hit_stats__ = np.append(std_hit_stats__, [np.std(hit_stats_, axis=0)], axis=0)

    return (avg_conv__, std_conv__,
            avg_mrr_i__, avg_mrr_f__, std_mrr_i__, std_mrr_f__,
            avg_hit_i__, avg_hit_f__, std_hit_i__, std_hit_f__,
            avg_stats__, std_stats__,
            avg_mrr_stats__, std_mrr_stats__,
            avg_hit_stats__, std_hit_stats__)


def get_experiment_stats(dataset, model, methods, log_file, num_of_exp=5, num_sess=5):
    summary_num_metrics = 11
    num_metrics = 7

    if dataset == "WN18RR":
        num_triples = 86835
    elif dataset == "FB15K237":
        num_triples = 272115
    elif dataset == "THOR_U":
        num_triples = 1580
    else:
        logout("Dataset not recognized for result generation", "f")
        exit()

    # accumulates the metrics
    conv_ = np.ndarray(shape=(0, num_sess, len(methods)))
    mrr_i_ = np.ndarray(shape=(0, num_sess, len(methods)))
    mrr_f_ = np.ndarray(shape=(0, num_sess, len(methods)))
    hit_i_ = np.ndarray(shape=(0, num_sess, len(methods)))
    hit_f_ = np.ndarray(shape=(0, num_sess, len(methods)))
    stats_ = np.ndarray(shape=(0, summary_num_metrics, len(methods)))
    mrr_stats_ = np.ndarray(shape=(0, len(methods), num_metrics))
    hit_stats_ = np.ndarray(shape=(0, len(methods), num_metrics))

    for exp_num in range(1, num_of_exp+1):
        conv = np.ndarray(shape=(0, num_sess))
        avg_mrr_f = np.ndarray(shape=(0, num_sess))
        avg_mrr_i = np.ndarray(shape=(0, num_sess))
        avg_hit_f = np.ndarray(shape=(0, num_sess))
        avg_hit_i = np.ndarray(shape=(0, num_sess))
        mrr_acc = []
        hits_acc = []
        mrr_fwt = []
        hits_fwt = []
        mrr_rem = []
        hits_rem = []
        mrr_pbwt = []
        hits_pbwt = []
        ms = []
        sss = []
        lca = []

        # must be accounted for bc SI allocates variables before initial learning session so not in memory sizes
        l2_initial_size = 0.0

        # gather logged data for the plot
        filepath_root = os.path.abspath(os.path.dirname(__file__)) + "/logs/continual_setting__" + dataset + "_mt" + model + "_"
        for method in methods:
            method_str = "clm" + method
            filepath = filepath_root + method_str + "/test_" + str(exp_num) + ".csv"
            inf_f = csvlogs2plots_format_inference(filepath)
            filepath = filepath_root + method_str + "/performances_" + str(exp_num) + ".csv"
            inf, run_conv, lcs, model_sizes, sample_sizes = csvlogs2plots_format_noninference(filepath)
            inf[1, 1, :, :] = inf_f[1, 1, :, :]
            inf[1, 0, :, :] = inf_f[1, 0, :, :]
            avg_mrr_i = np.append(avg_mrr_i, [np.average(np.triu(inf[0, 1, :, :]), axis=0)], axis=0)
            avg_mrr_f = np.append(avg_mrr_f, [np.average(np.triu(inf[1, 1, :, :]), axis=0)], axis=0)
            avg_hit_i = np.append(avg_hit_i, [np.average(np.triu(inf[0, 0, :, :]), axis=0)], axis=0)
            avg_hit_f = np.append(avg_hit_f, [np.average(np.triu(inf[1, 0, :, :]), axis=0)], axis=0)
            conv = np.append(conv, run_conv, axis=0)
            # ACC & FWT
            mrr_f_T = inf[1, 1, :, :].T
            hit_f_T = inf[1, 0, :, :].T
            mrr_acc.append("{:.4f}".format(np.sum(np.tril(mrr_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
            hits_acc.append("{:.4f}".format(np.sum(np.tril(hit_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
            mrr_fwt.append("{:.4f}".format(np.sum(np.triu(mrr_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
            hits_fwt.append("{:.4f}".format(np.sum(np.triu(hit_f_T)) / ((num_sess * (num_sess + 1)) / 2.0)))
            # BWT+ & REM
            mrr_bwt = 0.0
            hit_bwt = 0.0
            for i in range(1, mrr_f_T.shape[0]):
                for j in range(i):
                    mrr_bwt = mrr_f_T[i, j] - mrr_f_T[j, j]
            for i in range(1, hit_f_T.shape[0]):
                for j in range(i):
                    hit_bwt = hit_f_T[i, j] - hit_f_T[j, j]
            mrr_bwt = mrr_bwt / ((num_sess * (num_sess - 1)) / 2.0)
            hit_bwt = hit_bwt / ((num_sess * (num_sess - 1)) / 2.0)
            mrr_rem.append("{:.4f}".format(1.0 - np.absolute(np.min([0, mrr_bwt]))))
            mrr_pbwt.append("{:.4f}".format(np.max([0, mrr_bwt])))
            hits_rem.append("{:.4f}".format(1.0 - np.absolute(np.min([0, hit_bwt]))))
            hits_pbwt.append("{:.4f}".format(np.max([0, hit_bwt])))
            # MS & SSS
            if "L2" in filepath:
                l2_initial_size = copy(float(model_sizes[0, 0]))
            if "SI" in filepath:
                if l2_initial_size == 0.0:
                    logout("L2 inital size is wrong.", "w")
                model_sizes[0, 0] = l2_initial_size
            ms.append("{:.4f}".format(np.min([1.0, np.average(model_sizes[0, 0] / model_sizes)])))
            sss.append("{:.4f}".format(1.0 - np.min([1.0, np.average(sample_sizes / num_triples)])))
            # LCA
            LCA_fracs = []
            for lc in lcs:
                best_value = lc[-1, 1]
                best_value_idx = int(np.argwhere(lc[:, 1] == best_value)[0])
                to_best_value_curve = lc[:best_value_idx+1, :]
                x = to_best_value_curve[:, 0]
                y = to_best_value_curve[:, 1]
                normalize_y = np.ones_like(y) * best_value
                frac = integrate.trapz(x=x, y=y) / integrate.trapz(x=x, y=normalize_y)
                if isnan(frac):
                    frac = 1.0
                LCA_fracs.append(frac)
            lca.append("{:.4f}".format(np.average(LCA_fracs)))

        # perform final data transformations
        conv = np.transpose(conv)
        avg_mrr_i = np.transpose(avg_mrr_i) * 100.0
        avg_mrr_f = np.transpose(avg_mrr_f) * 100.0
        avg_hit_i = np.transpose(avg_hit_i) * 100.0
        avg_hit_f = np.transpose(avg_hit_f) * 100.0
        stats = copy(np.stack((mrr_acc, hits_acc, mrr_fwt, hits_fwt, mrr_pbwt, hits_pbwt, mrr_rem, hits_rem, ms, sss, lca)))
        mrr_stats = copy(np.stack((mrr_acc, mrr_fwt, mrr_pbwt, mrr_rem, ms, sss, lca))).astype(float).T
        hit_stats = copy(np.stack((hits_acc, hits_fwt, hits_pbwt, hits_rem, ms, sss, lca))).astype(float).T

        # append to the averaging arrays
        conv_ = np.append(conv_, [conv], axis=0)
        mrr_i_ = np.append(mrr_i_, [avg_mrr_i], axis=0)
        mrr_f_ = np.append(mrr_f_, [avg_mrr_f], axis=0)
        hit_i_ = np.append(hit_i_, [avg_hit_i], axis=0)
        hit_f_ = np.append(hit_f_, [avg_hit_f], axis=0)
        stats_ = np.append(stats_, [stats.astype(float)], axis=0)
        mrr_stats_ = np.append(mrr_stats_, [mrr_stats], axis=0)
        hit_stats_ = np.append(hit_stats_, [hit_stats], axis=0)

    run_stats_test(mrr_stats_[:, :, 0], methods, num_of_exp, "MRR ACC Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(mrr_stats_[:, :, 1], methods, num_of_exp, "MRR FWT Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(mrr_stats_[:, :, 2], methods, num_of_exp, "MRR +BWT Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(mrr_stats_[:, :, 3], methods, num_of_exp, "MRR REM Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 0], methods, num_of_exp, "HIT ACC Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 1], methods, num_of_exp, "HIT FWT Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 2], methods, num_of_exp, "HIT +BWT Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 3], methods, num_of_exp, "HIT REM Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 4], methods, num_of_exp, "MS Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 5], methods, num_of_exp, "SSS Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test(hit_stats_[:, :, 6], methods, num_of_exp, "LCA Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test_all_sessions(conv_, methods, num_of_exp, num_sess, "Convergence Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test_all_sessions(mrr_i_, methods, num_of_exp, num_sess, "MRR Initial Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test_all_sessions(mrr_f_, methods, num_of_exp, num_sess, "MRR Final Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test_all_sessions(hit_i_, methods, num_of_exp, num_sess, "Hits@10 Initial Stats for " + str(dataset) + " using " + str(model).upper(), log_file)
    run_stats_test_all_sessions(hit_f_, methods, num_of_exp, num_sess, "Hits@10 Final Stats for " + str(dataset) + " using " + str(model).upper(), log_file)


def run_stats_test_all_sessions(data, methods, num_exp, num_sess, test_label, log_file):
    for i in range(num_sess):
        run_stats_test(data[:, i, :], methods, num_exp, test_label + " in session " + str(i), log_file)


def run_stats_test(data, methods, num_exp, test_label, log_file):
    df = pd.DataFrame(columns=["exp", "method", "value"])
    for exp_num in range(num_exp):
        for method_num in range(len(methods)):
            df = df.append(pd.DataFrame([[exp_num, methods[method_num], data[exp_num, method_num]]],
                                        columns=["exp", "method", "value"]), ignore_index=True)
    aovrm = AnovaRM(df, 'value', 'exp', within=['method'])
    res = aovrm.fit()
    mcDate = multi.MultiComparison(df["value"], df["method"])
    res2 = mcDate.tukeyhsd()
    with open(log_file, "a") as f:
        f.write(test_label + "\n" + str(res) + "\n" + str(res2))


def get_plots(dataset, model, methods, num_exp=5, num_sess=5):
    avg_conv, std_conv, \
    avg_mrr_i, avg_mrr_f, std_mrr_i, std_mrr_f, \
    avg_hit_i, avg_hit_f, std_hit_i, std_hit_f, \
    avg_stats, std_stats, \
    avg_mrr_stats, std_mrr_stats, \
    avg_hit_stats, std_hit_stats = extract_runs_avg_std([dataset], [model], methods, num_exp, num_sess)

    avg_conv = np.average(avg_conv, axis=0)
    std_conv = np.average(std_conv, axis=0)
    avg_mrr_i = np.average(avg_mrr_i, axis=0)
    avg_mrr_f = np.average(avg_mrr_f, axis=0)
    std_mrr_i = np.average(std_mrr_i, axis=0)
    std_mrr_f = np.average(std_mrr_f, axis=0)
    avg_hit_i = np.average(avg_hit_i, axis=0)
    avg_hit_f = np.average(avg_hit_f, axis=0)
    std_hit_i = np.average(std_hit_i, axis=0)
    std_hit_f = np.average(std_hit_f, axis=0)
    avg_stats = np.average(avg_stats, axis=0)
    std_stats = np.average(std_stats, axis=0)
    avg_mrr_stats = np.average(avg_mrr_stats, axis=0)
    avg_hit_stats = np.average(avg_hit_stats, axis=0)

    # format method names/colors
    names = format_method_names(methods)
    colors = format_method_colors(methods)
    linestyles = format_method_linestyles(methods)
    hatches = format_method_hatches(methods)

    # generate each plot
    conv_f_plot = plot_mbar(avg_conv, names, colors, hatches,
                            ylabel="Epochs",
                            titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                            top_title="Convergence Speed for " + dataset + " across Sessions using " + str(model.upper()),
                            ylim=[0.0, np.max(avg_conv)],
                            yerr=std_conv)
    avg_mrr_i_bplot = plot_mbar(avg_mrr_i, names, colors, hatches,
                               ylabel="MRR %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Initial MRR for " + dataset + " across Sessions using " + str(model.upper()),
                               ylim=[0.0, np.max(avg_mrr_f)],
                               yerr=std_mrr_i)
    avg_mrr_f_bplot = plot_mbar(avg_mrr_f, names, colors, hatches,
                               ylabel="MRR %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Final MRR for " + dataset + " across Sessions using " + str(model.upper()),
                               ylim=[0.0, np.max(avg_mrr_f)],
                               yerr=std_mrr_f)
    avg_mrr_bplot = plot_mbar_stacked(avg_mrr_i, avg_mrr_f, names, colors, hatches,
                                      ylabel="MRR %",
                                      titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                                      top_title="MRR for " + dataset + " across Sessions using " + str(model.upper()),
                                      ylim=[0.0, np.max(avg_mrr_f)],
                                      yerr1=std_mrr_i, yerr2=std_mrr_f)
    avg_hit_i_bplot = plot_mbar(avg_hit_i, names, colors, hatches,
                               ylabel="Hits@10 %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Initial Hits@10 for " + dataset + " across Sessions using " + str(model.upper()),
                               ylim=[0.0, np.max(avg_hit_f)],
                               yerr=std_hit_i)
    avg_hit_f_bplot = plot_mbar(avg_hit_f, names, colors, hatches,
                               ylabel="Hits@10 %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Final Hits@10 for " + dataset + " across Sessions using " + str(model.upper()),
                               ylim=[0.0, np.max(avg_hit_f)],
                               yerr=std_hit_f)
    avg_hit_bplot = plot_mbar_stacked(avg_hit_i, avg_hit_f, names, colors, hatches,
                                      ylabel="Hits@10 %",
                                      titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                                      top_title="Hits@10 for " + dataset + " across Sessions using " + str(model.upper()),
                                      ylim=[0.0, np.max(avg_hit_f)],
                                      yerr1=std_hit_i, yerr2=std_hit_f)
    avg_mrr_i_lplot = plot_line(np.arange(num_sess), avg_mrr_i.T, names, colors, linestyles,
                                ylabel="MRR %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Initial MRR for " + dataset + " across Sessions using " + str(model.upper()),
                                ylim=[0.0, np.max(avg_mrr_f)],
                                yerr=std_mrr_i.T)
    avg_mrr_f_lplot = plot_line(np.arange(num_sess), avg_mrr_f.T, names, colors, linestyles,
                                ylabel="MRR %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Final MRR for " + dataset + " across Sessions using " + str(model.upper()),
                                ylim=[0.0, np.max(avg_mrr_f)],
                                yerr=std_mrr_f.T)
    avg_hit_i_lplot = plot_line(np.arange(num_sess), avg_hit_i.T, names, colors, linestyles,
                                ylabel="Hits@10 %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Initial Hits@10 for " + dataset + " across Sessions using " + str(model.upper()),
                                ylim=[0.0, np.max(avg_hit_f)],
                                yerr=std_hit_i.T)
    avg_hit_f_lplot = plot_line(np.arange(num_sess), avg_hit_f.T, names, colors, linestyles,
                                ylabel="Hits@10 %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Final Hits@10 for " + dataset + " across Sessions using " + str(model.upper()),
                                ylim=[0.0, np.max(avg_hit_f)],
                                yerr=std_hit_f.T)
    avg_summary_table = plot_table(avg_stats,
                               row_labels=["AVG MRR ACC", "AVG Hits@10 ACC", "AVG MRR FWT", "AVG Hits@10 FWT",
                                           "AVG MRR +BWT", "AVG Hits@10 +BWT", "AVG MRR REM", "AVG Hits@10 REM",
                                           "AVG MS", "AVG SSS", "AVG LCA"],
                               col_labels=names,
                               title="AVG Summary Table for " + dataset + " using " + str(model.upper()))
    std_summary_table = plot_table(std_stats,
                               row_labels=["STD MRR ACC", "STD Hits@10 ACC", "STD MRR FWT", "STD Hits@10 FWT",
                                           "STD MRR +BWT", "STD Hits@10 +BWT", "STD MRR REM", "STD Hits@10 REM",
                                           "STD MS", "STD SSS", "STD LCA"],
                               col_labels=names,
                               title="STD Summary Table for " + dataset + " using " + str(model.upper()))

    mrr_radar = plot_radar(avg_mrr_stats, colors, linestyles,
                           metric_labels=["ACC", "FWT", "+BWT", "REM", "MS", "SSS", "LCA"],
                           method_labels=names,
                           title="MRR CL Metrics Radar for " + dataset + " using " + str(model.upper()))
    hit_radar = plot_radar(avg_hit_stats, colors, linestyles,
                           metric_labels=["ACC", "FWT", "+BWT", "REM", "MS", "SSS", "LCA"],
                           method_labels=names,
                           title="Hits@10 CL Metrics Radar for " + dataset + " using " + str(model.upper()))
    mrr_acclca_scatter = plot_scatter(avg_mrr_stats[:, -1], avg_mrr_stats[:, 0], names, colors, linestyles,
                                      xlabel="LCA", ylabel="ACC MRR",
                                      top_title="Comparison for " + dataset + " using " + str(model.upper()))
                                      # xerr=std_mrr_stats[:, -1], yerr=std_mrr_stats[:, 0])
    hit_acclca_scatter = plot_scatter(avg_hit_stats[:, -1], avg_hit_stats[:, 0], names, colors, linestyles,
                                      xlabel="LCA", ylabel="ACC Hits@10",
                                      top_title="Comparison for " + dataset + " using " + str(model.upper()))
                                      # xerr=std_hit_stats[:, -1], yerr=std_hit_stats[:, 0])
    mrr_accms_scatter = plot_scatter(avg_mrr_stats[:, 4], avg_mrr_stats[:, 0], names, colors, linestyles,
                                      xlabel="MS", ylabel="ACC MRR",
                                      top_title="Comparison for " + dataset + " using " + str(model.upper()))
                                      # xerr=std_mrr_stats[:, 4], yerr=std_mrr_stats[:, 0])
    hit_accms_scatter = plot_scatter(avg_hit_stats[:, 4], avg_hit_stats[:, 0], names, colors, linestyles,
                                      xlabel="MS", ylabel="ACC Hits@10",
                                      top_title="Comparison for " + dataset + " using " + str(model.upper()))
                                      # xerr=std_hit_stats[:, 4], yerr=std_hit_stats[:, 0])

    # output to PDF
    return [avg_summary_table, std_summary_table,
            mrr_radar, hit_radar,
            conv_f_plot,
            avg_mrr_i_bplot, avg_mrr_f_bplot, avg_mrr_bplot,
            avg_hit_i_bplot, avg_hit_f_bplot, avg_hit_bplot,
            avg_mrr_i_lplot, avg_mrr_f_lplot, avg_hit_i_lplot, avg_hit_f_lplot,
            mrr_acclca_scatter, hit_acclca_scatter, mrr_accms_scatter, hit_accms_scatter]


def get_avg_plots(datasets, models, methods, avg_name="", num_exp=5, num_sess=5):
    avg_conv, std_conv, \
    avg_mrr_i, avg_mrr_f, std_mrr_i, std_mrr_f, \
    avg_hit_i, avg_hit_f, std_hit_i, std_hit_f, \
    avg_stats, std_stats, \
    avg_mrr_stats, std_mrr_stats, \
    avg_hit_stats, std_hit_stats = extract_runs_avg_std(datasets, models, methods, num_exp, num_sess)

    avg_conv = np.average(avg_conv, axis=0)
    std_conv = np.average(std_conv, axis=0)
    avg_mrr_i = np.average(avg_mrr_i, axis=0)
    avg_mrr_f = np.average(avg_mrr_f, axis=0)
    std_mrr_i = np.average(std_mrr_i, axis=0)
    std_mrr_f = np.average(std_mrr_f, axis=0)
    avg_hit_i = np.average(avg_hit_i, axis=0)
    avg_hit_f = np.average(avg_hit_f, axis=0)
    std_hit_i = np.average(std_hit_i, axis=0)
    std_hit_f = np.average(std_hit_f, axis=0)
    avg_stats = np.average(avg_stats, axis=0)
    std_stats = np.average(std_stats, axis=0)
    avg_mrr_stats = np.average(avg_mrr_stats, axis=0)
    avg_hit_stats = np.average(avg_hit_stats, axis=0)

    # format method names/colors
    names = format_method_names(methods)
    colors = format_method_colors(methods)
    linestyles = format_method_linestyles(methods)
    hatches = format_method_hatches(methods)

    # generate each plot
    conv_f_plot = plot_mbar(avg_conv, names, colors, hatches,
                            ylabel="Epochs",
                            titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                            top_title="Convergence Speed" + avg_name,
                            ylim=[0.0, np.max(avg_conv)],
                            yerr=std_conv)
    avg_mrr_i_bplot = plot_mbar(avg_mrr_i, names, colors, hatches,
                               ylabel="MRR %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Initial MRR" + avg_name,
                               ylim=[0.0, np.max(avg_mrr_f)],
                               yerr=std_mrr_i)
    avg_mrr_f_bplot = plot_mbar(avg_mrr_f, names, colors, hatches,
                               ylabel="MRR %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Final MRR" + avg_name,
                               ylim=[0.0, np.max(avg_mrr_f)],
                               yerr=std_mrr_f)
    avg_mrr_bplot = plot_mbar_stacked(avg_mrr_i, avg_mrr_f, names, colors, hatches,
                                      ylabel="MRR %",
                                      titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                                      top_title="MRR" + avg_name,
                                      ylim=[0.0, np.max(avg_mrr_f)],
                                      yerr1=std_mrr_i, yerr2=std_mrr_f)
    avg_hit_i_bplot = plot_mbar(avg_hit_i, names, colors, hatches,
                               ylabel="Hits@10 %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Initial Hits@10" + avg_name,
                               ylim=[0.0, np.max(avg_hit_f)],
                               yerr=std_hit_i)
    avg_hit_f_bplot = plot_mbar(avg_hit_f, names, colors, hatches,
                               ylabel="Hits@10 %",
                               titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                               top_title="Final Hits@10" + avg_name,
                               ylim=[0.0, np.max(avg_hit_f)],
                               yerr=std_hit_f)
    avg_hit_bplot = plot_mbar_stacked(avg_hit_i, avg_hit_f, names, colors, hatches,
                                      ylabel="Hits@10 %",
                                      titles=["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"],
                                      top_title="Hits@10" + avg_name,
                                      ylim=[0.0, np.max(avg_hit_f)],
                                      yerr1=std_hit_i, yerr2=std_hit_f)
    avg_mrr_i_lplot = plot_line(np.arange(num_sess), avg_mrr_i.T, names, colors, linestyles,
                                ylabel="MRR %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Initial MRR" + avg_name,
                                ylim=[0.0, np.max(avg_mrr_f)],
                                yerr=std_mrr_i.T)
    avg_mrr_f_lplot = plot_line(np.arange(num_sess), avg_mrr_f.T, names, colors, linestyles,
                                ylabel="MRR %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Final MRR" + avg_name,
                                ylim=[0.0, np.max(avg_mrr_f)],
                                yerr=std_mrr_f.T)
    avg_hit_i_lplot = plot_line(np.arange(num_sess), avg_hit_i.T, names, colors, linestyles,
                                ylabel="Hits@10 %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Initial Hits@10" + avg_name,
                                ylim=[0.0, np.max(avg_hit_f)],
                                yerr=std_hit_i.T)
    avg_hit_f_lplot = plot_line(np.arange(num_sess), avg_hit_f.T, names, colors, linestyles,
                                ylabel="Hits@10 %",
                                xticks=[[0, 1, 2, 3, 4], ["LS-1", "LS-2", "LS-3", "LS-4", "LS-5"]],
                                top_title="Final Hits@10" + avg_name,
                                ylim=[0.0, np.max(avg_hit_f)],
                                yerr=std_hit_f.T)
    avg_summary_table = plot_table(avg_stats,
                                   row_labels=["AVG MRR ACC", "AVG Hits@10 ACC", "AVG MRR FWT", "AVG Hits@10 FWT",
                                               "AVG MRR +BWT", "AVG Hits@10 +BWT", "AVG MRR REM", "AVG Hits@10 REM",
                                               "AVG MS", "AVG SSS", "AVG LCA"],
                                   col_labels=names,
                                   title="AVG Summary Table" + avg_name)
    std_summary_table = plot_table(std_stats,
                                   row_labels=["STD MRR ACC", "STD Hits@10 ACC", "STD MRR FWT", "STD Hits@10 FWT",
                                               "STD MRR +BWT", "STD Hits@10 +BWT", "STD MRR REM", "STD Hits@10 REM",
                                               "STD MS", "STD SSS", "STD LCA"],
                                   col_labels=names,
                                   title="STD Summary Table" + avg_name)
    mrr_radar = plot_radar(avg_mrr_stats, colors, linestyles,
                           metric_labels=["ACC", "FWT", "+BWT", "REM", "MS", "SSS", "LCA"],
                           method_labels=names,
                           title="MRR" + avg_name)
    hit_radar = plot_radar(avg_hit_stats, colors, linestyles,
                           metric_labels=["ACC", "FWT", "+BWT", "REM", "MS", "SSS", "LCA"],
                           method_labels=names,
                           title="Hits@10" + avg_name)
    mrr_acclca_scatter = plot_scatter(avg_mrr_stats[:, -1], avg_mrr_stats[:, 0], names, colors, linestyles,
                                      xlabel="LCA", ylabel="ACC MRR",
                                      top_title="ACC to Learning Speed Comparsion" + avg_name)
                                      # xerr=std_mrr_stats[:, -1], yerr=std_mrr_stats[:, 0])
    hit_acclca_scatter = plot_scatter(avg_hit_stats[:, -1], avg_hit_stats[:, 0], names, colors, linestyles,
                                      xlabel="LCA", ylabel="ACC Hits@10",
                                      top_title="ACC to Learning Speed Comparsion" + avg_name)
                                      # xerr=std_hit_stats[:, -1], yerr=std_hit_stats[:, 0])
    mrr_accms_scatter = plot_scatter(avg_mrr_stats[:, 4], avg_mrr_stats[:, 0], names, colors, linestyles,
                                     xlabel="MS", ylabel="ACC MRR",
                                     top_title="ACC to Model Size Comparsion" + avg_name)
                                     # xerr=std_mrr_stats[:, 4], yerr=std_mrr_stats[:, 0])
    hit_accms_scatter = plot_scatter(avg_hit_stats[:, 4], avg_hit_stats[:, 0], names, colors, linestyles,
                                     xlabel="MS", ylabel="ACC Hits@10",
                                     top_title="ACC to Model Size Comparsion" + avg_name)
                                     # xerr=std_hit_stats[:, 4], yerr=std_hit_stats[:, 0])

    # output to PDF
    return [avg_summary_table, std_summary_table,
            mrr_radar, hit_radar,
            conv_f_plot,
            avg_mrr_i_bplot, avg_mrr_f_bplot, avg_mrr_bplot,
            avg_hit_i_bplot, avg_hit_f_bplot, avg_hit_bplot,
            avg_mrr_i_lplot, avg_mrr_f_lplot, avg_hit_i_lplot, avg_hit_f_lplot,
            mrr_acclca_scatter, hit_acclca_scatter, mrr_accms_scatter, hit_accms_scatter]


if __name__ == "__main__":
    parser = ArgumentParser(description="Generates output plots and statistical tests for all experiments.")
    parser.add_argument("-es", dest="exp_setting", type=str, help="select Experimental Setting for visualizations")
    parser.add_argument('-mc', dest='methods', type=str, default=["offline", "finetune", "L2", "SI", "PNN", "CWR", "DGR"],
                        nargs='+', help='Methods to compare for outputs')
    parser.add_argument("-t", dest="tag", type=str, help="Tag name for outputs")
    args = parser.parse_args()

    plt.rcParams.update({'font.weight': 'bold'})

    if args.exp_setting == "robot":  # optional plots not in paper commented out
        # analogy = get_plots("THOR_U", "analogy", args.methods)
        # transe = get_plots("THOR_U", "transe", args.methods)
        avg = get_avg_plots(["THOR_U"], ["transe","analogy"], args.methods, avg_name="Robot Evaluation " + args.tag.upper())
        # figs2pdf(analogy + transe + avg, "robot_results_" + args.tag + ".pdf")
        figs2pdf(avg, "robot_results_" + args.tag + ".pdf")
        get_experiment_stats("THOR_U", "transe", args.methods, "robot_transe_" + args.tag + ".txt")
        get_experiment_stats("THOR_U", "analogy", args.methods, "robot_analogy_" + args.tag + ".txt")
    elif args.exp_setting == "bench":
        # wn_analogy = get_plots("WN18RR", "analogy", args.methods)
        # wn_transe = get_plots("WN18RR", "transe", args.methods)
        # fb_analogy = get_plots("FB15K237", "analogy", args.methods)
        # fb_transe = get_plots("FB15K237", "transe", args.methods)
        avg = get_avg_plots(["WN18RR", "FB15K237"], ["transe", "analogy"], args.methods, avg_name="Benchmark Evaluation")
        # figs2pdf(wn_analogy + wn_transe + fb_analogy + fb_transe + avg, "bench_results.pdf")
        figs2pdf(avg, "bench_results.pdf")
        get_experiment_stats("WN18RR", "transe", args.methods, "wn_transe.txt")
        get_experiment_stats("WN18RR", "analogy", args.methods, "wn_analogy.txt")
        get_experiment_stats("FB15K237", "transe", args.methods, "fb_transe.txt")
        get_experiment_stats("FB15K237", "analogy", args.methods, "fb_analogy.txt")
    else:
        logout("Experiment Setting not recognized", "e")
