import traits.api as t
import numpy as np
from hyperspy.events import Events, Event
from hyperspy.axes import DataAxis


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
        in 'axes_manager'. The axes defined in 'axes[i]' argument will be
        sliced with 'ranges[i]', all other axes with 'slice(None)'. If 'ranges'
        is None, the ranges defined by the ROI will be used.
        """
        if ranges is None:
            ranges = []
            ndim = len(self.coords)
            for i in xrange(ndim):
                c = self.coords[i]
                if len(c) == 1:
                    ranges.append((c[0],))
                elif len(c) == 2:
                    ranges.append((c[0], c[1]))
        slices = []
        for ax in axes_collecion:
            if ax in axes:
                i = axes.index(ax)
                ilow = ax.value2index(ranges[i][0])
                if len(ranges[i]) == 1:
                    ihigh = 1 + ilow
                else:
                    ihigh = 1 + ax.value2index(ranges[i][1],
                                               rounding=lambda x: round(x-1))
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
        """
        axes = self._parse_axes(axes, signal.axes_manager, signal._plot)

        natax = signal.axes_manager._get_axes_in_natural_order()
        slices = self._make_slices(natax, axes)
        if out is None:
            roi = signal[slices]
            return roi
        else:
            signal.__getitem__(slices, out=out)

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
        nd = len(axes)
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


class BasePointROI(BaseROI):
    """Base ROI class for point ROIs, i.e. ROIs with a unit size in each of its
    dimensions.
    """
    pass


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

    def __repr__(self):
        return "%s(value=%f)" % (
            self.__class__.__name__,
            self.value)


class SpanROI(BaseROI):
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

    def __repr__(self):
        return "%s(left=%f, right=%f)" % (
            self.__class__.__name__,
            self.left,
            self.right)


class RectangularROI(BaseROI):
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

    def __repr__(self):
        return "%s(left=%f, top=%f, right=%f, bottom=%f)" % (
            self.__class__.__name__,
            self.left,
            self.top,
            self.right,
            self.bottom)
