import traits.api as t
from hyperspy.events import Events, Event
from hyperspy.axes import DataAxis


class BaseROI(t.HasTraits):
    def __init__(self):
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()
        self.signal_map = dict()
       
    def _get_coords(self):
        raise NotImplementedError()
    
    def _set_coords(self, value):
        raise NotImplementedError()
    
    coords = property(lambda s: s._get_coords(), lambda s,v: s._set_coords(v))

    def update(self):
        if t.Undefined not in self.coords:
            self.events.roi_changed.trigger(self)

    def _make_slices(self, axes_collecion, axes, ranges=None):
        """
        Utility function to make a slice container that will slice the axes
        in axes_manager. The axis in 'axes[i]' argument will be sliced with 
        'ranges[i]', all other axes with 'slice(None)'.
        """
        if ranges is None:
            ranges= []
            ndim = len(self.coords)//2
            for i in xrange(ndim):
                ranges.append((self.coords[i], self.coords[ndim+i]))
        slices = []
        for ax in axes_collecion:
            if ax in axes:
                i = axes.index(ax)
                ilow = ax.value2index(ranges[i][0])
                ihigh = 1 + ax.value2index(ranges[i][1], rounding=lambda x: round(x-1))
                slices.append(slice(ilow, ihigh))
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __call__(self, signal, out=None, axes=None):
        if axes is None and self.signal_map.has_key(signal):
            axes = self.signal_map[signal][1]
        else:
            axes = self._parse_axes(axes, signal.axes_manager, signal._plot)
        
        natax = signal.axes_manager._get_axes_in_natural_order()
        slices = self._make_slices(natax, axes)
        if out is None:
            roi = signal[slices]
            return roi
        else:
            signal.__getitem__(slices, out=out)

class SpanROI(BaseROI):
    left, right = (t.CFloat(t.Undefined),) * 2
    
    def __init__(self, left, right):
        super(SpanROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.left, self.right = left, right
        
    def _get_coords(self):
        return self.left, self.right
        
    def _set_coords(self, value):
        if self.coords != value:
            self.left, self.right = value

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

    def _parse_axes(self, axes, axes_manager, plot):
        if isinstance(axes, basestring):
            # Specifies space
            if axes.startswith("nav"):
                ax = plot.navigator_plot.ax
                x = axes_manager.navigation_axes[0]
            elif axes.startswith("sig"):
                ax = plot.signal_plot.ax
                x = axes_manager.signal_axes[0]
        elif isinstance(axes, tuple):
            if isinstance(axes[0], DataAxis):
                x = axes[0]
            else:
                x = axes_manager[axes[0]]
            if x.navigate:
                ax = plot.navigator_plot.ax
            else:
                ax = plot.signal_plot.ax
        else:
            if axes_manager.navigation_dimension > 0:
                ax = plot.navigator_plot.ax
                x = axes_manager.navigation_axes[0]
            elif axes_manager.signal_dimension > 0:
                ax = plot.signal_plot.ax
                x = axes_manager.signal_axes[0]
            else:
                raise ValueError("Neither space has one dimensions")
        return (x,), ax

    def __repr__(self):
        return "%s(left=%f, right=%f)" % (
            self.__class__.__name__,
            self.left,
            self.right)


class RectangularROI(BaseROI):
    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4

    def __init__(self, left, top, right, bottom):
        super(RectangularROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        
    def _get_coords(self):
        return self.left, self.top, self.right, self.bottom
        
    def _set_coords(self, value):
        if self.coords != value:
            self.left, self.top, self.right, self.bottom = value

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

    def _parse_axes(self, axes, axes_manager, plot):
        if isinstance(axes, basestring):
            # Specifies space
            if axes.startswith("nav"):
                ax = plot.navigator_plot.ax
                x = axes_manager.navigation_axes[0]
                y = axes_manager.navigation_axes[1]
            elif axes.startswith("sig"):
                ax = plot.signal_plot.ax
                x = axes_manager.signal_axes[0]
                y = axes_manager.signal_axes[1]
        elif isinstance(axes, tuple):
            if isinstance(axes[0], DataAxis):
                x = axes[0]
            else:
                x = axes_manager[axes[0]]
            if isinstance(axes[1], DataAxis):
                y = axes[1]
            else:
                y = axes_manager[axes[1]]
            if x.navigate != y.navigate:
                raise ValueError("Axes need to be in same space")
            if x.navigate:
                ax = plot.navigator_plot.ax
            else:
                ax = plot.signal_plot.ax
        else:
            if axes_manager.navigation_dimension > 1:
                ax = plot.navigator_plot.ax
                x = axes_manager.navigation_axes[0]
                y = axes_manager.navigation_axes[1]
            elif axes_manager.signal_dimension > 1:
                ax = plot.signal_plot.ax
                x = axes_manager.signal_axes[0]
                y = axes_manager.signal_axes[1]
            else:
                raise ValueError("Neither space has two dimensions")
        return (x,y), ax

    def __repr__(self):
        return "%s(left=%f, top=%f, right=%f, bottom=%f)" % (
            self.__class__.__name__,
            self.left,
            self.top,
            self.right,
            self.bottom)
