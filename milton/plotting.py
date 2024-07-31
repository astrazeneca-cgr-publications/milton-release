import numpy as np
import pandas as pd
import scipy.stats as st
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import holoviews as hv
from bokeh.models import HoverTool
from io import StringIO
from bs4 import BeautifulSoup

from .processing import clip_quantiles, trim_quantiles

# needed for holoviews to fully initialize
hv.extension('bokeh')
hv.output(widget_location='top_left')


def cm_scale(series, cmap=None, trim=False, low_q=0.01, high_q=.99):
    """ Converts series to color map by first its clipping quantiles
    to better scale the values to [0, 1].
    """
    proc_func = trim_quantiles if trim else clip_quantiles
    scaled = series.pipe(proc_func, low_q=low_q, high_q=high_q)\
        .pipe(lambda s: pd.Series(
            MinMaxScaler().fit_transform(s.to_numpy().reshape((-1, 1)))[:, 0],
            s.index))
    
    return scaled if cmap is None else cmap(scaled)


def kde_density2d(data, groups=None, n_points=100, q_thresh=.01):
    """ Generates 2D gaussian KDE surface for the data, which can 
    be optionally grouped into a number of groups.
    
    Parameters
    ----------
    data : array n_obs*2
    groups : integer array of the same length as data
    n_points : number of interpolation points in each dimension
    q_thresh : discard points beyond quantile range (q_thresh, 1-q_thresh)
    
    Returns
    -------
    surface data : either a dict keyed by each group with corresponding Z array
        as value or just Z when no groups were specified
    XY : 2*n_points*n_points array with value coordinates 
    """
    x = data[:, 0]
    y = data[:, 1]
    
    if groups is None:
        groups = np.ones(len(data))
    
    xmin = np.percentile(x, 100 * q_thresh)
    xmax = np.percentile(x, 100 * (1 - q_thresh))
    ymin = np.percentile(y, 100 * q_thresh)
    ymax = np.percentile(y, 100 * (1 - q_thresh))
    
    XY = np.mgrid[xmin:xmax:complex(n_points), ymin:ymax:complex(n_points)]
    xy = np.vstack([m.ravel() for m in XY]).T
    
    selection = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    X = data[selection]
    G = groups[selection]
    results = {}
    
    for g in np.unique(G):
        kernel = st.gaussian_kde(X[G == g].T)
        z_gr = kernel(xy.T).reshape((n_points, n_points))
        results[g] = z_gr
        
    res = results if len(results) > 1 else results[g]
    return res, XY


def density_contours_2d(x2d, groups, levels=5, q_thresh=0.001):
    z_map, XY = kde_density2d(np.asarray(x2d), 
                              np.asarray(groups), 
                              q_thresh=q_thresh)
    n, m = XY[0].shape
    min_x = XY[0].min()
    min_y = XY[1].min()
    dx = XY[0].max() - min_x
    dy = XY[1].max() - min_y
    plots = []
    
    for group in np.unique(groups):
        contours = []
        z_min = z_map[group].min()
        z_max = z_map[group].max()
        
        for lvl in np.linspace(z_min, z_max, levels + 2)[1:-1]:
            for cnt in measure.find_contours(z_map[group], lvl):
                contours.append({
                    'x': min_x + dx * cnt[:, 0] / n, 
                    'y': min_y + dy * cnt[:, 1] / m, 
                    'value': lvl})
        plots.append(contours)
        
    return plots


def roc_markers(fpr, tpr, n=10):
    """Finds n indices if the ROC curve that are approximately
    equi-distant.
    """
    points = np.vstack([fpr, tpr])
    point_dists = np.sqrt(np.sum(np.diff(points)**2, axis=0))
    curve_length = np.cumsum(point_dists)
    ix = np.searchsorted(curve_length, np.linspace(0, 2, n + 2))
    n = len(curve_length)
    return np.where(ix < n, ix, n - 1)


def plot_multi_curve(curves, axes, include_avg=True):
    """Generates a set of ROC curve plots for set of scorings.
    It includes by default an extra curve for the average score.
    
    Parameters
    ----------
    curves : ndarray, shape: (n_curves, k_points, 2)
      The curves to plot
    axes : two strings
      names of plot axes
    include_avg : boolean
      When True (default), plots the mean of individual curves
    
    Returns
    -------
    Holoviews Overlay object with all curves.
    """
    plots = []
    for i in range(len(curves)):
        p = hv.Curve(curves[i], tuple(axes), group='experts')
        plots.append(p)

    if include_avg:
        avg = curves.mean(axis=0)
        p0 = hv.Curve(avg, tuple(axes), group='ensemble')
        p1 = hv.Points(avg[::3], group='ensemble')
        hover = HoverTool(tooltips=[
            (axes[0], '@{%s}{0.00}' % axes[0]),
            (axes[1], '@{%s}{0.00}' % axes[1]),
        ])
        plots.append(hv.Overlay([
            p0.opts(hv.opts.Curve(line_color='Blue', alpha=.1)), 
            p1.opts(tools=[hover], size=5)]))
    return hv.Overlay(plots)


def marker_score_dist(df, x, y, group, 
                      n_points=1000, 
                      low_q=0.001, 
                      high_q=.999, 
                      n_bins=25, 
                      title=None, 
                      height=200,
                      width=250,
                      simple=False):
    """Plots a composite of a scatter plot (x, y) with a side axis
    of histogram of y grouped by the values of group.
    Current implementation expects group to contain values [0, 1].
    
    Parameters
    ----------
    df : the data frame with columns named in x, y, group
    x : name of the x dimension of the scatter plot
    y : name of the y dimension of the scatter plot
    group : name of the column with binary groups (only values [0, 1] 
        are read, other are discarded)
    n_points : number of scatter points to sample
    low_q : filter out df[y] values below this quantile
    high_q : filter out df[y] values above this quantile
    title : the plot title
    simple : whether to strip the plot of axes (for visual simplicity)
    """
    q0, q1 = df[y].quantile([low_q, high_q])
    
    data_smp = df[[x, y, group]]\
        .pipe(lambda df: df[(df[y] >= q0) & (df[y] <= q1) & df[group].notnull()])\
        .sample(n_points)

    bins = np.linspace(q0, q1, n_bins)

    points = hv.Points(data_smp, kdims=[x, y])\
        .opts(alpha=.1, size=10, color='score', 
              cmap='coolwarm', line_alpha=0, 
              title=title, fontsize={'title': 9})
    
    if simple:
        points.opts(labelled=[], yaxis=None, xaxis=None)

    histograms = []
    for gr in (0, 1):
        hist = hv.Histogram(
            np.histogram(df.loc[df[group] == gr, y], bins=bins, density=True),
            kdims=y)
        hist.opts(yaxis=None, xaxis=None, alpha=.5, 
                  line_alpha=.1, cmap='coolwarm')
        histograms.append(hist)
        
    hist = hv.Overlay(histograms).opts(width=width//3)
    return points.opts(height=height, width=width) << hist


def plot_ensemble_score_dist(scores, y_true, bins=30):
    plots = []

    for target, label in zip((0, 1), ('Controls', 'Cases')):
        hist, bins = np.histogram(scores[y_true == target], 
                                  density=True, bins=bins)
        plots.append(hv.Histogram((hist, bins), label=label))

    return hv.Overlay(plots)\
        .opts(hv.opts.Histogram(
            xlabel='Probability',
            xticks=5, 
            yticks=5,
            alpha=.5,
            title='Scores on validation set'))


def plot_ukb_score_dist(known_scores, unknown_scores, n_points=150):
    bins = np.linspace(0, 1, n_points)
    plots = []
    scores = [
        known_scores.rename('ICD10-based', copy=False),
        unknown_scores.rename('Rest of UKB', copy=False)
    ]
    for s in scores:
        h, _ = np.histogram(s, bins=bins, density=True)
        p = hv.Curve((bins, h), kdims='score', vdims='Frequency', label=s.name)
        plots.append(p)

    return hv.Overlay(plots).opts(hv.opts.Curve(
        alpha=.7,
        title='Scores on full UKB'))


def plot_feature_importance(coeffs, names, top_n=20):
    def sort_by_func(df, func, **kwargs):
        """Sorts DataFrame by the results of function evaluation.
        """
        order = func(df).sort_values(**kwargs)
        return df.reindex(order.index)
    
    ft_imp = pd.concat(coeffs, axis=1, keys=names)\
        .fillna(0)\
        .rename_axis('Feature')\
        .pipe(sort_by_func, lambda df: df[names[0]].abs(), ascending=True)\
        .reset_index()

    ft_imp_tall = ft_imp.melt(id_vars=['Feature'], 
                value_vars=names, 
                var_name='Model',
                value_name='Importance')

    holo_map = hv.HoloMap(kdims=['Model'], sort=False)
    
    def set_importance_width(plot, element):
        plot.handles['table'].autosize_mode = 'fit_columns'

    for i, name in enumerate(names):
        viz_name = '%s. %s' % (i+1, name)
        imp = ft_imp_tall.query('Model == @name').drop('Model', axis=1)
        bar_chart = hv.Bars(imp.iloc[-top_n:], 'Feature', 'Importance')\
            .opts(width=600, invert_axes=True, labelled=[])
        table = hv.Table(imp.iloc[::-1])\
            .opts(width=350, hooks=[set_importance_width])
        holo_map[viz_name] = table + bar_chart

    return holo_map.opts(title='').collate()


class Proj2dPlots:
    """A set of utilities for various kinds of plots of 2D projections.
    Currenly, only PCA is supported. Alternatives include TSNE.
    """
    
    def __init__(self, 
                 max_scatter_points=1000,
                 cmap='coolwarm', 
                 y_name='Cases/Controls'):
        self.max_scatter_points = max_scatter_points
        self.cmap = cmap
        self.y_name = y_name
        
    def fit(self, X, y):
        pca = PCA().fit(X)
        ix = pd.Index(np.arange(1, X.shape[1] + 1), name='# of components')
        self.explained_variance_ratio_ = pd.Series(
            100 * pca.explained_variance_ratio_,
            index=ix,
            name='Variance Explained [%]')
        
        x2d = pd.DataFrame(PCA(n_components=2).fit_transform(X), 
                           columns=['_X_', '_Y_'], 
                           index=X.index)
        
        self.contours_ = density_contours_2d(x2d[['_X_', '_Y_']], y)
        max_points = min(self.max_scatter_points, X.shape[0])
        features = X.apply(cm_scale).sample(max_points)
        
        self.data_ = pd.concat([
            x2d.loc[features.index],
            y.loc[features.index].to_frame(self.y_name),
            features
        ], axis=1)
        
        return self

    def plot_scree(self):
        ratios = self.explained_variance_ratio_.cumsum()
        return hv.Curve(ratios).opts(title='PCA Explained Variance Ratio')

    def _plot_projection(self,data, color):
        chart = hv.Scatter(data,  
                           kdims=['_X_', '_Y_'], 
                           vdims=[color], 
                           label=color)
        return chart.opts(
            alpha=.25, 
            size=10,
            color=color,
            colorbar=False,
            cmap=self.cmap,
            line_alpha=0,
            tools=[])


    def plot_projections(self, features=None):
        plots = []
        contours = [hv.Contours(c) for c in self.contours_]
        
        if features is None:
            features = [self.y_name]
            
        data = self.data_[['_X_', '_Y_'] + features]

        for feature in features:
            scatter = self._plot_projection(data, feature)
            p = hv.Overlay([scatter] + contours)\
                .opts(hv.opts.Contours(line_width=2, alpha=.6))\
                .opts(xaxis=None, yaxis=None)
            plots.append(p)

        if len(plots) == 1:
            return plots[0]
        else:
            return hv.Layout(plots).opts(hv.opts.Scatter(labelled=[]))

    
def hv_render_html(plot):
    """A bit of a hack to have embeddable Holoviews plots that correctly render
    holomaps. 
    
    Returns
    -------
    js_scripts - javascript/css definitions for the HTML header
    html - HTML div + javascript to be embedded in the HTML body
    """
    with StringIO() as fout:
        hv.save(plot, fout, toolbar=True)
        fout.seek(0)
        page = BeautifulSoup(fout, 'html.parser')
        header = page.find('head').find_all(['style', 'script'])
        body = page.find('body')
        return ''.join(map(str, header)), ''.join(map(str, body.contents))

    
def html_components(dict_of_plots):
    """A simple utility that extracts the repeatable javascript/css header
    definitions from a dictionary of HTML-rendered holoviews objects.
    
    Returns
    -------
    header_stuff : js/css definitions common to all plots (string)
    plot_dic : dictionary of the same structure as the input, but with
        values replaced by their corresponding HTML body representations
    """
    res = {}
    scripts = None
    for k, val in dict_of_plots.items():
        scripts, plot = val
        res[k] = plot

    return scripts, res
