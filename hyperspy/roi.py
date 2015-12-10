import traits.api as t
import numpy as np
from hyperspy.events import Events, Event
import hyperspy.interactive as hsi
from hyperspy.axes import DataAxis
from hyperspy.drawing import widgets


class BaseROI(t.HasTraits):

    """Base class for all ROIs.

    Provides some basic functionality that is likely to be shared between all
    ROIs, and serve as a common type that can be checked for.
    """

    def __init__(self):
        """Sets up events.roi_changed event, and inits HasTraits.
        """
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()
        self.signal_map = dict()

    def _get_coords(self):
        """_get_coords() is the getter for the coords property, and should be
        implemented by inheritors to return a 2D tuple containing all the
        coordinates that are needed to define the ROI. The tuple should be of
        the structure:
        tuple([tuple( <all 1st axis coordinates> ), \
               tuple( <all 2nd axis coordinates> ), ... ])
        """
        raise NotImplementedError()

    def _set_coords(self, value):
        """_set_coords is the setter for the coords property, and should be
        implemented by inheritors to set its internal represenation of the ROI
        from a tuple of the following structure:
        tuple([tuple( <all 1st axis coordinates> ), \
               tuple( <all 2nd axis coordinates> ), ... ])
        """
        raise NotImplementedError()

    coords = property(lambda s: s._get_coords(), lambda s, v: s._set_coords(v))

    def _get_ndim(self):
        return len(self.coords)

    ndim = property(lambda s: s._get_ndim())

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        The base implementation simply triggers the roi_changed event.
        """
        if t.Undefined not in np.ravel(self.coords):
            self.events.roi_changed.trigger(self)

    def _make_slices(self, axes_collecion, axes, ranges=None):
        """
        Utility function to make a slice structure that will slice all the axes
        in 'axes_collecion'. The axes defined in 'axes[i]' argument will be
        sliced with 'ranges[i]', all other axes with 'slice(None)'. If 'ranges'
        is None, the ranges defined by the ROI will be used.
        """
        if ranges is None:
            ranges = []
            for i in xrange(self.ndim):
                c = self.coords[i]
                if len(c) == 1:
                    ranges.append((c[0],))
                elif len(c) == 2:
                    ranges.append((c[0], c[1]))
        slices = []
        for ax in axes_collecion:
            if ax in axes:
                i = axes.index(ax)
                try:
                    ilow = ax.value2index(ranges[i][0])
                except ValueError:
                    if ranges[i][0] < ax.low_value:
                        ilow = ax.low_index
                    else:
                        raise
                if len(ranges[i]) == 1:
                    ihigh = 1 + ilow
                else:
                    try:
                        ihigh = 1 + ax.value2index(
                            ranges[i][1], rounding=lambda x: round(x - 1))
                    except ValueError:
                        if ranges[i][0] < ax.high_value:
                            ihigh = ax.high_index + 1
                        else:
                            raise
                slices.append(slice(ilow, ihigh))
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __call__(self, signal, out=None, axes=None):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * "navigation" or "signal", in which the first axes of that
                  space's axes will be used.
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        """
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        natax = signal.axes_manager._get_axes_in_natural_order()
        slices = self._make_slices(natax, axes)
        if out is None:
            roi = signal[slices]
            return roi
        else:
            signal.__getitem__(slices, out=out)

    def mean(self, signal, out=None, axes=None):
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        roi = self(signal, out=None, axes=axes)
        ids = []
        for ax in axes:
            ids.append(signal.axes_manager._axes.index(ax))
        # Reverse-sort so indices stay valid while collapsing
        ids.sort(reverse=True)
        for idx in ids:
            roi = roi.mean(axis=idx + 3j)

        if out is None:
            return roi
        else:
            out.data = roi.data
            out.events.data_changed.trigger()

    def _parse_axes(self, axes, axes_manager):
        """Utility function to parse the 'axes' argument to a tuple of
        DataAxis, and find the matplotlib Axes that contains it.

        Arguments
        ---------
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The DataAxis in the collection can be either of the following:
                * "navigation" or "signal", in which the first axes of that
                  space's axes will be used.
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        axes_manager : AxesManager
            The AxesManager to use for parsing axes, if axes is not already a
            tuple of DataAxis.

        Returns
        -------
        (tuple(<DataAxis>), matplotlib Axes)
        """
        nd = self.ndim
        if isinstance(axes, basestring) and (axes.startswith("nav") or
                                             axes.startswith("sig")):
            # Specifies space
            if axes.startswith("nav"):
                x = axes_manager.navigation_axes[0]
                if nd > 1:
                    y = axes_manager.navigation_axes[1]
            elif axes.startswith("sig"):
                x = axes_manager.signal_axes[0]
                if nd > 1:
                    y = axes_manager.signal_axes[1]
        elif isinstance(axes, tuple):
            if isinstance(axes[0], DataAxis):
                x = axes[0]
            else:
                x = axes_manager[axes[0]]
            if nd > 1:
                if isinstance(axes[1], DataAxis):
                    y = axes[1]
                else:
                    y = axes_manager[axes[1]]
        else:
            if axes_manager.navigation_dimension >= nd:
                x = axes_manager.navigation_axes[0]
                if nd > 1:
                    y = axes_manager.navigation_axes[1]
            elif axes_manager.signal_dimension >= nd:
                x = axes_manager.signal_axes[0]
                if nd > 1:
                    y = axes_manager.signal_axes[1]
            elif nd == 2 and axes_manager.navigation_dimensions == 1 and \
                    axes_manager.signal_dimension == 1:
                # We probably have a navigator plot including both nav and sig
                # axes.
                x = axes_manager.signal_axes[0]
                y = axes_manager.navigation_axes[0]
            else:
                raise ValueError("Could not find valid axes configuration.")

        if nd > 1:
            axes = (x, y)
        else:
            axes = (x,)
        return axes

    def _get_mpl_ax(self, plot, axes):
        """Returns MPL Axes that contains the ROI.

        plot : MPL_HyperExplorer
            The space of the first DataAxis in axes will be used to extract the
            matplotlib Axes.
        """
        nd = len(axes)

        if nd > 1 and axes[0].navigate != axes[1].navigate:
            # Here we assume that the navigator plot includes one of
            # the signal dimensions, and that the user wants to have
            # the ROI there.
            ax = plot.navigator_plot.ax
        elif axes[0].navigate:
            ax = plot.navigator_plot.ax
        else:
            ax = plot.signal_plot.ax
        return ax


class BaseInteractiveROI(BaseROI):

    """Base class for interactive ROIs, i.e. ROIs with widget interaction.
    The base class defines a lot of the common code for interacting with
    widgets, but inhertors need to implement the following functions:

    _get_widget_type()
    _apply_roi2widget(widget)

    If the widgets are of the common interface used in drawing.widgets, the
    other functions do not need to be overridden. However, if it deviates from
    this interface, the following function would most likely need to be
    overridden:

    _set_coords_from_widget(widget)

    The reason for _set_coords_from_widget having a default implementation
    while _apply_roi2widget does not, is because the application of the
    geometry to the widget should happen in an optimized manner that prevents
    flickering of the widget during the update.
    """

    def __init__(self):
        super(BaseInteractiveROI, self).__init__()
        self.widgets = set()

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        This implementation triggers the roi_changed event, and updates the
        widgets associated with it.
        """
        if t.Undefined not in np.ravel(self.coords):
            if not self.events.roi_changed.suppress:
                self._update_widgets()
            self.events.roi_changed.trigger(self)

    def _get_widget_type(self, axes, signal):
        """Get the type of a widget that can represent the ROI on the given
        axes and signal.
        """
        raise NotImplementedError()

    def _apply_roi2widget(self, widget):
        """This function is responsible for applying the ROI geometry to the
        widget. When this function is called, the widget's events are already
        suppressed, so this should not be necessary _apply_roi2widget to
        handle.
        """
        raise NotImplementedError()

    def _update_widgets(self, exclude=set()):
        """Internal function for updating the associated widgets to the
        geometry contained in the ROI.

        Arguments
        ---------
        exclude : set()
            A set of widgets to exclude from the update. Useful e.g. if a
            widget has triggered a change in the ROI: Then all widgets,
            excluding the one that was the source for the change, should be
            updated.
        """
        if not isinstance(exclude, set):
            exclude = set(exclude)
        for w in self.widgets - exclude:
            with w.events.changed.suppress_single(self._on_widget_change):
                self._apply_roi2widget(w)

    def interactive(self, signal, navigation_signal="same", out=None,
                    **kwargs):
        """Creates an interactivley sliced Signal (sliced by this ROI) via
        hyperspy.interactive.

        Arguments:
        ----------
        signal : Signal
            The source signal to slice
        navigation_signal : Signal, None or "same" (default)
            If not None, it will automatically create a widget on
            navigation_signal. Passing "same" is identical to pasing the same
            signal to 'signal' and 'navigation_signal', but is less ambigous,
            and allows "same" to be the default value.
        out : Signal
            If not None, it will use 'out' as the output instead of returning
            a new Signal.
        """
        if navigation_signal == "same":
            navigation_signal = signal
        if navigation_signal is not None:
            if navigation_signal not in self.signal_map:
                self.add_widget(navigation_signal)
        if out is None:
            return hsi.interactive(self.__call__,
                                   event=self.events.roi_changed,
                                   signal=signal,
                                   **kwargs)
        else:
            return hsi.interactive(self.__call__,
                                   event=self.events.roi_changed,
                                   signal=signal, out=out, **kwargs)

    def navigate(self, signal):
        """Make a widget for this ROI and use it as a navigator for passed
        signal.
        """
        # Check valid plot and navdim >= roi dim
        if signal._plot is None or \
                signal.axes_manager.navigation_dimension < self.ndim:
            raise ValueError("Cannot navigate this signal with %s" %
                             self.__class__.__name__, signal)

        nav_axes = signal.axes_manager.navigation_axes[0:self.ndim + 1]

        def nav_signal_function(axes_manager=None):
            if axes_manager is None:
                axes_manager = signal.axes_manager
            nav_idx = list()
            for ax in nav_axes:
                nav_idx.append(axes_manager._axes.index(ax))
            nav_idx = tuple(nav_idx)
            slices = self._make_slices(axes_manager._axes, nav_axes)
            data = np.mean(signal.data.__getitem__(slices), nav_idx)
            return np.atleast_1d(data)

        signal.signal_callback = nav_signal_function
        sp = signal._plot.signal_plot
        sp.update()
        w = self.add_widget(signal, axes=nav_axes, color='red')
        if hasattr(w.events, 'resized'):
            w.events.resized.connect(sp.update, 0)
        w.connect_navigate()
        if signal._plot.pointer is not None:
            signal._plot.pointer.close()
        signal._plot.pointer = w
        signal._plot.navigator_plot.update()
        return w

    def _set_coords_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        c = widget.coordinates
        s = widget.get_size_in_axes()
        self.coords = zip(c, c + s)

    def _on_widget_change(self, widget):
        """Callback for widgets' 'changed' event. Updates the internal state
        from the widget, and triggers events (excluding connections to the
        source widget).
        """
        with self.events.suppress:
            self._bounds_check = False
            try:
                self._set_coords_from_widget(widget)
            finally:
                self._bounds_check = True
        self._update_widgets(exclude=(widget,))
        self.events.roi_changed.trigger(self)

    def add_widget(self, signal, axes=None, widget=None, color='green'):
        """Add a widget to visually represent the ROI, and connect it so any
        changes in either are reflected in the other. Note that only one
        widget can be added per signal/axes pair.

        Arguments:
        ----------
        signal : Signal
            The signal to witch the widget is added. This is used to determine
            with plot to add the widget to, and it supplies the axes_manager
            for the widget.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The DataAxis in the collection can be either of the following:
                * "navigation" or "signal", in which the first axes of that
                  space's axes will be used.
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        widget : Widget or None (default)
            If specified, this is the widget that will be added. If None, the
            default widget will be used, as given by _get_widget_type().
        color : Matplotlib color specifier (default: 'green')
            The color for the widget. Any format that matplotlib uses should be
            ok. This will not change the color fo any widget passed with the
            'widget' argument.
        """
        axes = self._parse_axes(axes, signal.axes_manager,)
        if widget is None:
            widget = self._get_widget_type(axes, signal)(signal.axes_manager)
            widget.color = color

        # Remove existing ROI, if it exsists and axes match
        if signal in self.signal_map and \
                self.signal_map[signal][1] == axes:
            self.remove_widget(signal)

        if axes is not None:
            # Set DataAxes
            widget.axes = axes
        with widget.events.changed.suppress_single(self._on_widget_change):
            self._apply_roi2widget(widget)
        if widget.ax is None:
            ax = self._get_mpl_ax(signal._plot, axes)
            widget.set_mpl_ax(ax)

        # Connect widget changes to on_widget_change
        widget.events.changed.connect(self._on_widget_change, 1)
        # When widget closes, remove from internal list
        widget.events.closed.connect(self._remove_widget, 1)
        self.widgets.add(widget)
        self.signal_map[signal] = (widget, axes)
        return widget

    def _remove_widget(self, widget):
        widget.events.closed.disconnect(self._remove_widget)
        widget.events.changed.disconnect(self._on_widget_change)
        widget.close()
        for signal, w in self.signal_map.iteritems():
            if w == widget:
                self.signal_map.pop(signal)
                break

    def remove_widget(self, signal):
        if signal in self.signal_map:
            w = self.signal_map.pop(signal)[0]
            self._remove_widget(w)


class BasePointROI(BaseInteractiveROI):

    """Base ROI class for point ROIs, i.e. ROIs with a unit size in each of its
    dimensions.
    """

    def _set_coords_from_widget(self, widget):
        c = widget.coordinates
        self.coords = zip(c)


class Point1DROI(BasePointROI):

    """Selects a single point in a 1D space. The coordinate of the point in the
    1D space is stored in the 'value' trait.
    """
    value = t.CFloat(t.Undefined)

    def __init__(self, value):
        super(Point1DROI, self).__init__()
        self.value = value

    def _value_changed(self, old, new):
        self.update()

    def _get_coords(self):
        return ((self.value,),)

    def _set_coords(self, value):
        if self.coords != value:
            self.value = value[0][0]

    def _apply_roi2widget(self, widget):
        widget.coordinates = self.value

    def _get_widget_type(self, axes, signal):
        # Figure out whether to use horizontal or veritcal line:
        if axes[0].navigate:
            plotdim = len(signal._plot.navigator_data_function().shape)
            axdim = signal.axes_manager.navigation_dimension
            idx = signal.axes_manager.navigation_axes.index(axes[0])
        else:
            plotdim = len(signal._plot.signal_data_function().shape)
            axdim = signal.axes_manager.signal_dimension
            idx = signal.axes_manager.signal_axes.index(axes[0])

        if plotdim == 2:  # Plot is an image
            # axdim == 1 and plotdim == 2 indicates "spectrum stack"
            if idx == 0 and axdim != 1:    # Axis is horizontal
                return widgets.DraggableVerticalLine
            else:  # Axis is vertical
                return widgets.DraggableHorizontalLine
        elif plotdim == 1:  # It is a spectrum
            return widgets.DraggableVerticalLine
        else:
            raise ValueError("Could not find valid widget type")

    def __repr__(self):
        return "%s(value=%f)" % (
            self.__class__.__name__,
            self.value)


class Point2DROI(BasePointROI):

    """Selects a single point in a 2D space. The coordinates of the point in
    the 2D space are stored in the traits 'x' and 'y'.
    """
    x, y = (t.CFloat(t.Undefined),) * 2

    def __init__(self, x, y):
        super(Point2DROI, self).__init__()
        self.x, self.y = x, y

    def _get_coords(self):
        return ((self.x,), (self.y,))

    def _set_coords(self, value):
        if self.coords != value:
            (self.x,), (self.y,) = value

    def _x_changed(self, old, new):
        self.update()

    def _y_changed(self, old, new):
        self.update()

    def _apply_roi2widget(self, widget):
        widget.coordinates = (self.x, self.y)

    def _get_widget_type(self, axes, signal):
        return widgets.DraggableSquare

    def __repr__(self):
        return "%s(x=%f, y=%f)" % (
            self.__class__.__name__,
            self.x, self.y)


class SpanROI(BaseInteractiveROI):

    """Selects a range in a 1D space. The coordinates of the range in
    the 1D space are stored in the traits 'left' and 'right'.
    """
    left, right = (t.CFloat(t.Undefined),) * 2

    def __init__(self, left, right):
        super(SpanROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.left, self.right = left, right

    def _get_coords(self):
        return ((self.left, self.right),)

    def _set_coords(self, value):
        if self.coords != value:
            self.left, self.right = value[0]

    def _right_changed(self, old, new):
        if self._bounds_check and \
                self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self._bounds_check and \
                self.right is not t.Undefined and new >= self.right:
            self.left = old
        else:
            self.update()

    def _apply_roi2widget(self, widget):
        widget.set_bounds(left=self.left, right=self.right)

    def _get_widget_type(self, axes, signal):
        return widgets.DraggableResizableRange

    def __repr__(self):
        return "%s(left=%f, right=%f)" % (
            self.__class__.__name__,
            self.left,
            self.right)


class RectangularROI(BaseInteractiveROI):

    """Selects a range in a 2D space. The coordinates of the range in
    the 2D space are stored in the traits 'left', 'right', 'top' and 'bottom'.
    """
    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4

    def __init__(self, left, top, right, bottom):
        super(RectangularROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

    def _get_coords(self):
        return (self.left, self.right), (self.top, self.bottom)

    def _set_coords(self, value):
        if self.coords != value:
            (self.left, self.right), (self.top, self.bottom) = value

    def _top_changed(self, old, new):
        if self._bounds_check and \
                self.bottom is not t.Undefined and new >= self.bottom:
            self.top = old
        else:
            self.update()

    def _bottom_changed(self, old, new):
        if self._bounds_check and \
                self.top is not t.Undefined and new <= self.top:
            self.bottom = old
        else:
            self.update()

    def _right_changed(self, old, new):
        if self._bounds_check and \
                self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self._bounds_check and \
                self.right is not t.Undefined and new >= self.right:
            self.left = old
        else:
            self.update()

    def _apply_roi2widget(self, widget):
        widget.set_bounds(left=self.left, bottom=self.bottom,
                          right=self.right, top=self.top)

    def _get_widget_type(self, axes, signal):
        return widgets.ResizableDraggableRectangle

    def __repr__(self):
        return "%s(left=%f, top=%f, right=%f, bottom=%f)" % (
            self.__class__.__name__,
            self.left,
            self.top,
            self.right,
            self.bottom)


class Line2DROI(BaseInteractiveROI):

    x1, y1, x2, y2, linewidth = (t.CFloat(t.Undefined),) * 5

    def __init__(self, x1, y1, x2, y2, linewidth):
        super(Line2DROI, self).__init__()
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.linewidth = linewidth

    def _get_coords(self):
        return (self.x1, self.x2), (self.y1, self.y2)

    def _set_coords(self, value):
        if self.coords != value:
            (self.x1, self.x2), (self.y1, self.y2) = value

    def _set_coords_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        c = widget.coordinates
        s = widget.size[0]
        self.coords = tuple(np.transpose(c).tolist())
        self.linewidth = s

    def _x1_changed(self, old, new):
        self.update()

    def _x2_changed(self, old, new):
        self.update()

    def _y1_changed(self, old, new):
        self.update()

    def _y2_changed(self, old, new):
        self.update()

    def _linewidth_changed(self, old, new):
        self.update()

    def _apply_roi2widget(self, widget):
        widget.coordinates = np.transpose(self.coords)
        widget.size = np.array([self.linewidth])

    def _get_widget_type(self, axes, signal):
        return widgets.DraggableResizable2DLine

    def navigate(self, signal):
        raise NotImplementedError("Line2DROI does not support navigation.")

    @staticmethod
    def _line_profile_coordinates(src, dst, linewidth=1):
        """Return the coordinates of the profile of an image along a scan line.
        Parameters
        ----------
        src : 2-tuple of numeric scalar (float or int)
            The start point of the scan line.
        dst : 2-tuple of numeric scalar (float or int)
            The end point of the scan line.
        linewidth : int, optional
            Width of the scan, perpendicular to the line
        Returns
        -------
        coords : array, shape (2, N, C), float
            The coordinates of the profile along the scan line. The length of
            the profile is the ceil of the computed length of the scan line.
        Notes
        -----
        This is a utility method meant to be used internally by skimage
        functions. The destination point is included in the profile, in
        contrast to standard numpy indexing.
        """
        src_row, src_col = src = np.asarray(src, dtype=float)
        dst_row, dst_col = dst = np.asarray(dst, dtype=float)
        d_row, d_col = dst - src
        theta = np.arctan2(d_row, d_col)

        length = np.ceil(np.hypot(d_row, d_col) + 1)
        # we add one above because we include the last point in the profile
        # (in contrast to standard numpy indexing)
        line_col = np.linspace(src_col, dst_col, length)
        line_row = np.linspace(src_row, dst_row, length)

        data = np.zeros((2, length, linewidth))
        data[0, :, :] = np.tile(line_col, [linewidth, 1]).T
        data[1, :, :] = np.tile(line_row, [linewidth, 1]).T

        if linewidth != 1:
            # we subtract 1 from linewidth to change from pixel-counting
            # (make this line 3 pixels wide) to point distances (the
            # distance between pixel centers)
            col_width = (linewidth - 1) * np.sin(-theta) / 2
            row_width = (linewidth - 1) * np.cos(theta) / 2
            row_off = np.linspace(-row_width, row_width, linewidth)
            col_off = np.linspace(-col_width, col_width, linewidth)
            data[0, :, :] += np.tile(col_off, [length, 1])
            data[1, :, :] += np.tile(row_off, [length, 1])
        return data

    @property
    def length(self):
        p0 = np.array((self.x1, self.y1), dtype=np.float)
        p1 = np.array((self.x2, self.y2), dtype=np.float)
        d_row, d_col = p1 - p0
        return np.hypot(d_row, d_col)

    @staticmethod
    def profile_line(img, src, dst, axes, linewidth=1,
                     order=1, mode='constant', cval=0.0):
        """Return the intensity profile of an image measured along a scan line.
        Parameters
        ----------
        img : numeric array, shape (M, N[, C])
            The image, either grayscale (2D array) or multichannel
            (3D array, where the final axis contains the channel
            information).
        src : 2-tuple of numeric scalar (float or int)
            The start point of the scan line.
        dst : 2-tuple of numeric scalar (float or int)
            The end point of the scan line.
        linewidth : int, optional
            Width of the scan, perpendicular to the line
        order : int in {0, 1, 2, 3, 4, 5}, optional
            The order of the spline interpolation to compute image values at
            non-integer coordinates. 0 means nearest-neighbor interpolation.
        mode : string, one of {'constant', 'nearest', 'reflect', 'wrap'}, optional
            How to compute any values falling outside of the image.
        cval : float, optional
            If `mode` is 'constant', what constant value to use outside the
            image.
        Returns
        -------
        return_value : array
            The intensity profile along the scan line. The length of the
            profile is the ceil of the computed length of the scan line.
        Examples
        --------
        >>> x = np.array([[1, 1, 1, 2, 2, 2]])
        >>> img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
        >>> img
        array([[0, 0, 0, 0, 0, 0],
               [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2],
               [1, 1, 1, 2, 2, 2],
               [0, 0, 0, 0, 0, 0]])
        >>> profile_line(img, (2, 1), (2, 4))
        array([ 1.,  1.,  2.,  2.])
        Notes
        -----
        The destination point is included in the profile, in contrast to
        standard numpy indexing.
        """

        import scipy.ndimage as nd
        p0 = ((src[0] - axes[0].offset) / axes[0].scale,
              (src[1] - axes[1].offset) / axes[1].scale)
        p1 = ((dst[0] - axes[0].offset) / axes[0].scale,
              (dst[1] - axes[1].offset) / axes[1].scale)
        perp_lines = Line2DROI._line_profile_coordinates(p0, p1,
                                                         linewidth=linewidth)
        if img.ndim > 2:
            img = np.rollaxis(img, axes[0].index_in_array, 0)
            img = np.rollaxis(img, axes[1].index_in_array, 1)
            orig_shape = img.shape
            img = np.reshape(img, orig_shape[0:2] +
                             (np.product(orig_shape[2:]),))
            pixels = [nd.map_coordinates(img[..., i], perp_lines,
                                         order=order, mode=mode, cval=cval)
                      for i in xrange(img.shape[2])]
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
            intensities = pixels.mean(axis=1)
            intensities = np.rollaxis(
                np.reshape(intensities,
                           intensities.shape[0:1] + orig_shape[2:]),
                0, i0)
        else:
            pixels = nd.map_coordinates(img, perp_lines,
                                        order=order, mode=mode, cval=cval)
            intensities = pixels.mean(axis=1)

        return intensities

    def interactive(self, signal, navigation_signal="same", out=None,
                    **kwargs):
        """Creates an interactivley sliced Signal (sliced by this ROI) via
        hyperspy.interactive.

        Arguments:
        ----------
        signal : Signal
            The source signal to slice
        navigation_signal : Signal, None or "same" (default)
            If not None, it will automatically create a widget on
            navigation_signal. Passing "same" is identical to pasing the same
            signal to 'signal' and 'navigation_signal', but is less ambigous,
            and allows "same" to be the default value.
        out : Signal
            If not None, it will use 'out' as the output instead of returning
            a new Signal.
        """
        if navigation_signal == "same":
            navigation_signal = signal
        if navigation_signal is not None:
            if navigation_signal not in self.signal_map:
                self.add_widget(navigation_signal)
        if out is None:
            return hsi.interactive(self.__call__,
                                   event=self.events.roi_changed,
                                   signal=signal,
                                   **kwargs)
        else:
            return hsi.interactive(self.__call__,
                                   event=self.events.roi_changed,
                                   signal=signal, out=out, **kwargs)

    def __call__(self, signal, out=None, axes=None, order=0):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * "navigation" or "signal", in which the first axes of that
                  space's axes will be used.
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        """
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)
        profile = Line2DROI.profile_line(signal.data,
                                         (self.x1, self.y1),
                                         (self.x2, self.y2),
                                         axes=axes,
                                         linewidth=self.linewidth,
                                         order=order)
        length = np.linalg.norm(np.diff(
                np.array(self.coords).T, axis=0), axis=1)[0]
        if out is None:
            axm = signal.axes_manager.deepcopy()
            idx = []
            for ax in axes:
                idx.append(ax.index_in_axes_manager)
            for i in reversed(sorted(idx)):  # Remove in reversed order
                axm.remove(i)
            axis = DataAxis(len(profile),
                            scale=length/len(profile),
                            units=axes[0].units,
                            navigate=axes[0].navigate)
            axis.axes_manager = axm
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            axm._axes.insert(i0, axis)
            from hyperspy.signals import Signal
            roi = Signal(profile, axes=axm._get_axes_dicts(),
                         metadata=signal.metadata.deepcopy().as_dictionary(),
                         original_metadata=signal.original_metadata.
                         deepcopy().as_dictionary())
            return roi
        else:
            out.data = profile
            i0 = min(axes[0].index_in_array, axes[1].index_in_array)
            ax = out.axes_manager[i0 + 3j]
            size = len(profile)
            scale = length/len(profile)
            axchange = size != ax.size or scale != ax.scale
            if axchange:
                ax.size = len(profile)
                ax.scale = length/len(profile)
                out.events.axes_changed.trigger(ax)
            out.events.data_changed.trigger()

    def __repr__(self):
        return "%s(x1=%f, y1=%f, x2=%f, y2=%f, linewidth=%f)" % (
            self.__class__.__name__,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.linewidth)


class CircleROI(BaseInteractiveROI):

    cx, cy, r, r_inner = (t.CFloat(t.Undefined),) * 4

    def __init__(self, cx, cy, r, r_inner=None):
        super(CircleROI, self).__init__()
        self.cx, self.cy, self.r = cx, cy, r
        if r_inner:
            self.r_inner = r_inner

    def _get_ndim(self):
        return 2

    def _get_coords(self):
        return (self.cx,), (self.cy,), (self.r, self.r_inner)

    def _set_coords(self, value):
        if self.coords != value:
            (self.cx,), (self.cy,), (self.r, self.r_inner) = value

    def _set_coords_from_widget(self, widget):
        """Sets the internal representation of the ROI from the passed widget,
        without doing anything to events.
        """
        c = widget.coordinates
        s = widget.get_size_in_axes()
        self.coords = (c[0],), (c[1],), tuple(np.transpose(s).tolist())

    def _cx_changed(self, old, new):
        self.update()

    def _cy_changed(self, old, new):
        self.update()

    def _r_changed(self, old, new):
        self.update()

    def _r_inner_changed(self, old, new):
        self.update()

    def _apply_roi2widget(self, widget):
        widget.coordinates = np.array((self.cx, self.cy))
        inner = self.r_inner if self.r_inner != t.Undefined else 0.0
        widget.size = np.array((self.r/widget.axes[0].scale,
                                inner/widget.axes[0].scale))

    def _get_widget_type(self, axes, signal):
        return widgets.Draggable2DCircle

    def navigate(self, signal):
        raise NotImplementedError("CircleROI does not support navigation.")

    def __call__(self, signal, out=None, axes=None):
        """Slice the signal according to the ROI, and return it.

        Arguments
        ---------
        signal : Signal
            The signal to slice with the ROI.
        out : Signal, default = None
            If the 'out' argument is supplied, the sliced output will be put
            into this instead of returning a Signal. See Signal.__getitem__()
            for more details on 'out'.
        axes : specification of axes to use, default = None
            The axes argument specifies which axes the ROI will be applied on.
            The items in the collection can be either of the following:
                * "navigation" or "signal", in which the first axes of that
                  space's axes will be used.
                * a tuple of:
                    - DataAxis. These will not be checked with
                      signal.axes_manager.
                    - anything that will index signal.axes_manager
                * For any other value, it will check whether the navigation
                  space can fit the right number of axis, and use that if it
                  fits. If not, it will try the signal space.
        """
        if axes is None and signal in self.signal_map:
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager)

        natax = signal.axes_manager._get_axes_in_natural_order()
        # Slice original data with a circumscribed rectangle
        ranges = [[self.cx - self.r, self.cx + self.r],
                  [self.cy - self.r, self.cy + self.r]]
        slices = self._make_slices(natax, axes, ranges)
        ir = [slices[natax.index(axes[0])],
              slices[natax.index(axes[1])]]
        vx = axes[0].axis[ir[0]] - self.cx
        vy = axes[1].axis[ir[1]] - self.cy
        gx, gy = np.meshgrid(vx, vy)
        gr = gx**2 + gy**2
        mask = gr > self.r**2
        if self.r_inner != t.Undefined:
            mask |= gr < self.r_inner**2
        tiles = []
        shape = []
        for i in xrange(len(slices)):
            if i == natax.index(axes[0]):
                tiles.append(1)
                shape.append(mask.shape[0])
            elif i == natax.index(axes[1]):
                tiles.append(1)
                shape.append(mask.shape[1])
            else:
                tiles.append(signal.axes_manager.shape[i])
                shape.append(1)
        mask = mask.reshape(shape)
        mask = np.tile(mask, tiles)

        if out is None:
            roi = signal[slices]
            roi.data = np.ma.masked_array(roi.data, mask, hard_mask=True)
            return roi
        else:
            with out.events.suppress:
                signal.__getitem__(slices, out=out)
            out.data = np.ma.masked_array(out.data, mask, hard_mask=True)
            out.events.axes_changed.trigger()
            out.events.data_changed.trigger()

    def __repr__(self):
        if self.r_inner == t.Undefined:
            return "%s(cx=%f, cy=%f, r=%f)" % (
                self.__class__.__name__,
                self.cx,
                self.cy,
                self.r)
        else:
            return "%s(cx=%f, cy=%f, r=%f, r_inner=%f)" % (
                self.__class__.__name__,
                self.cx,
                self.cy,
                self.r,
                self.r_inner)
