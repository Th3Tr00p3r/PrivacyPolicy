from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gensim.corpora import Dictionary
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from wordcloud import WordCloud


@dataclass
class Plotter:
    """
    A generalized, hierarchical plotting tool, designed to work as a context manager.

    Attributes
    ----------
    parent_ax : Axes, optional
        Parent axes object, by default None.
    parent_figure : Figure, optional
        Parent figure object, by default None.
    figsize : Tuple[float, float], optional
        Size of the figure, by default (5, 4).
    fontsize : int, optional
        Font size for the plot, by default 14.
    subplots : Tuple[int, int], optional
        Subplots configuration, by default (1, 1).
    suptitle : str, optional
        Title for the figure, by default None.
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    x_scale : str, optional
        Scale for the x-axis, by default None.
    y_scale : str, optional
        Scale for the y-axis, by default None.
    xlim : Tuple[float, float], optional
        Limits for the x-axis, by default None.
    ylim : Tuple[float, float], optional
        Limits for the y-axis, by default None.
    should_force_aspect : bool, optional
        Flag for forcing aspect ratio, by default False.
    should_autoscale : bool, optional
        Flag for autoscaling, by default False.
    selection_limits : Tuple[float, float], optional
        Limits for manual selection, by default None.
    should_close_after_selection : bool, optional
        Flag to close the plot after manual selection, by default False.
    subplot_kw : Dict[str, Any], optional
        Additional keyword arguments for subplots, by default field(default_factory=dict).

    Methods
    -------
    __enter__()
        Prepare the 'axes' object to use in context manager.
    __exit__(*exc)
        Set axes and figure attributes, and show the plot if at the top of the hierarchy.
    _quadratic_xscale_backwards(x)
        Helper function for quadratic scaling.
    _set_axis_attributes(ax)
        Set attributes for a given Axes object.

    """

    parent_ax: Axes = None
    # TODO: 'parent_figure' can be used for implementing 'subfigures', later on
    parent_figure: Figure = None
    figsize: Tuple[float, float] = (5, 4)
    fontsize: int = 14
    subplots: Tuple[int, int] = (1, 1)
    suptitle: str = None
    xlabel: str = None
    ylabel: str = None
    x_scale: str = None
    y_scale: str = None
    xlim: Tuple[float, float] = None
    ylim: Tuple[float, float] = None
    should_force_aspect: bool = False
    should_autoscale: bool = False
    selection_limits: Tuple[float, float] = None
    should_close_after_selection: bool = False
    subplot_kw: Dict[str, Any] = field(default_factory=dict)

    def __enter__(self):
        """Prepare the 'axes' object to use in context manager"""

        # dealing with a figure object
        if self.parent_ax is None:
            if self.parent_figure is None:  # creating a new figure
                if self.figsize is None:  # auto-determine size
                    n_rows, n_cols = self.subplots
                    ax_width, ax_height = self.AX_SIZE
                    self.figsize = (n_cols * ax_width, n_rows * ax_height)
                self.fig = plt.figure(figsize=self.figsize, constrained_layout=True)
                self.axes = self.fig.subplots(*self.subplots, subplot_kw=self.subplot_kw)
                if not hasattr(self.axes, "size"):  # if self.axes is not an ndarray
                    self.axes = np.array([self.axes])
            else:  # using given figure
                self.fig = self.parent_figure
                self.axes = np.array(self.parent_figure.get_axes())

        # dealing with a axes object
        else:
            if not hasattr(self.parent_ax, "size"):  # if parent_ax is not an ndarray
                self.axes = np.array([self.parent_ax])
            else:
                self.axes = self.parent_ax
            try:  # 1D array of axes
                self.fig = self.axes[0].figure
            except AttributeError:  # 2D array of axes
                self.fig = self.axes[0][0].figure

        if self.axes.size == 1:
            return self.axes[0]  # return a single Axes object
        else:
            return self.axes  # return a Numpy ndarray of Axes objects

    def __exit__(self, *exc):
        """
        Set axes attributes.
        Set figure attributes and show it if Plotter is at top of hierarchy.
        """

        for ax in self.axes.flatten().tolist():
            # set ax attributes
            self._set_axis_attributes(ax)
            # manual selection
            if self.selection_limits is not None:
                self.fig.suptitle(self.suptitle, fontsize=self.fontsize)
                self.fig.show()
                x_coords = []
                while not x_coords:  # enforce at least one point selection
                    selected_points_list = self.fig.ginput(n=-1, timeout=-1)
                    x_coords = [x for (x, y) in selected_points_list]
                    if x_coords:
                        break
                    print(
                        "Must select at least 1 point! (left button to set, right to erase last, middle or Enter to confirm)"
                    )
                if len(x_coords) == 1:  # select max only
                    self.selection_limits = tuple(self.selection_limits[0], max(x_coords))
                else:  # select min and max
                    self.selection_limits = tuple(min(x_coords), max(x_coords))
                if self.should_close_after_selection:
                    plt.close(self.fig)
                return

        if self.parent_ax is None:  # set figure attributes, and show it (dealing with figure)
            self.fig.suptitle(self.suptitle, fontsize=self.fontsize)
            self.fig.canvas.draw_idle()

    def _quadratic_xscale_backwards(self, x):
        """
        Helper function for quadratic scaling.

        Parameters
        ----------
        x : np.ndarray
            Input array for scaling.

        Returns
        -------
        np.ndarray
            Scaled output array.

        """

        if (x < 0).any():
            new_x = np.empty(x.shape)
            new_x[x < 0] = 0
            new_x[x >= 0] = x[x >= 0] ** (1 / 2)
            return new_x
        else:
            return x ** (1 / 2)

    def _set_axis_attributes(self, ax):
        """
        Set attributes for a given Axes object.

        Parameters
        ----------
        ax : Axes
            Matplotlib Axes object.

        """

        if self.should_force_aspect:
            force_aspect(ax, aspect=1)
        if self.should_autoscale:
            ax.autoscale()
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        if self.x_scale is not None:
            if self.x_scale == "quadratic":
                ax.set_xscale(
                    "function", functions=(lambda x: x**2, self._quadratic_xscale_backwards)
                )
            else:
                ax.set_xscale(self.x_scale)
        if self.y_scale is not None:
            ax.set_yscale(self.y_scale)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        [text.set_fontsize(self.fontsize) for text in [ax.title, ax.xaxis.label, ax.yaxis.label]]


def display_scatter(
    arr2d: np.ndarray, labels: List[str] = None, annots: List[str] = None, title="", **kwargs
):
    """
    Display scatter plot for 2D or 3D data.

    Parameters
    ----------
    arr2d : np.ndarray
        2D or 3D data array (sample rows, feature columns).
    labels : List[str], optional
        List of labels, by default None.
    annots : List[str], optional
        List of annotations, by default None.
    title : str, optional
        Title for the plot, by default "".
    **kwargs
        Additional keyword arguments for the plot.

    Raises
    ------
    ValueError
        If the array is not 2D or 3D.

    Notes
    -----
    - For 3D data, the scatter plot is prepared with a 3D projection.

    """

    if arr2d.ndim != 2:
        raise ValueError("Must supply a 2D array! (sample rows, feature columns)")

    if arr2d.shape[1] not in [2, 3]:
        raise ValueError("Can only display 2D or 3D data!")

    # prepare 3d projection if needed
    if arr2d.shape[1] == 3:
        kwargs["subplot_kw"] = dict(projection="3d")

    with Plotter(**kwargs) as ax:
        dim_slices = [col for col in arr2d.T]

        if labels is not None:
            unique_labels = list(set(labels))
            for col_idx, label in enumerate(unique_labels):
                label_indices = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(
                    *[dim_slice[label_indices] for dim_slice in dim_slices],
                    6,
                    label=label,
                )
            ax.legend()

        else:
            ax.scatter(*dim_slices, c=labels)

        ax.set_title(title)

        if annots is not None:
            for idx, annot in enumerate(annots):
                ax.annotate(annot, (arr2d[idx, 0], arr2d[idx, 1]), fontsize=8)


def display_dim_reduction(arr2d, name: str, **kwargs):
    """
    Display dimensionality-reduced data.

    Parameters
    ----------
    arr2d : np.ndarray
        2D matrix with 2 or 3 components (columns).
    name : str
        Name for the visualization.
    **kwargs
        Additional keyword arguments for the scatter plot.

    Notes
    -----
    - Expects a 2D matrix with 2 or 3 components (columns).
    - Uses 'display_scatter' internally for visualization.

    """

    components = range(1, arr2d.shape[1] + 1)
    axis_labels_dict = {
        f"{ax_str}label": f"Principle Component {component_num}"
        for ax_str, component_num in zip("xyz", components)
    }
    display_scatter(arr2d, title=f"{name} Visualization", **{**kwargs, **axis_labels_dict})


def display_wordcloud(dct: Dictionary | Dict[str, int], per_doc: bool = False):
    """
    Display a word cloud.

    Parameters
    ----------
    dct : Dictionary | Dict[str, int]
        Gensim Dictionary object or a word-frequency dictionary.
    per_doc : bool, optional
        Flag for per-document word frequency, by default False.

    Notes
    -----
    - Creates a word cloud from the provided dictionary.
    - Displays the word cloud using Matplotlib.

    """

    # create a word-count (total or per-document) dictionary from a Gensim Dictionary object
    word_count_dict = {
        dct[id]: count for id, count in getattr(dct, "dfs" if per_doc else "cfs").items()
    }

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white")

    # Generate the word cloud from your word frequency dictionary
    wordcloud.generate_from_frequencies(word_count_dict)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def force_aspect(ax, aspect=1) -> None:
    """
    Set the aspect ratio for a matplotlib plot.

    Parameters
    ----------
    ax : object
        Matplotlib Axes object.
    aspect : float, optional
        Desired aspect ratio, by default 1.

    Notes
    -----
    - Adjusts the aspect ratio of the plot using Matplotlib.
    - See accepted answer here: https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    """

    with suppress(ValueError):
        img, *_ = ax.get_images()
        extent = img.get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
