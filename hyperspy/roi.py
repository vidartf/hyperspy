import traits.api as t
import numpy as np
from hyperspy.events import Events, Event
from hyperspy.axes import DataAxis


class BaseROI(t.HasTraits):
    def __init__(self):
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()
       
    def _get_coords(self):
        raise NotImplementedError()
    
    def _set_coords(self, value):
        raise NotImplementedError()
    
    coords = property(lambda s: s._get_coords(), lambda s,v: s._set_coords(v))

    def update(self):
        if t.Undefined not in np.ravel(self.coords):
            self.events.roi_changed.trigger(self)

    def _make_slices(self, axes_collecion, axes, ranges=None):
        """
        Utility function to make a slice container that will slice the axes
        in axes_manager. The axis in 'axes[i]' argument will be sliced with 
        'ranges[i]', all other axes with 'slice(None)'.
        """
        if ranges is None:
            ranges= []
            ndim = len(self.coords)
            for i in xrange(ndim):
                c = self.coords[i]
                if len(c) == 1:
                    ranges.append((c[0], None))
                elif len(c) == 2:
                    ranges.append((c[0], c[1]))
        slices = []
        for ax in axes_collecion:
            if ax in axes:
                i = axes.index(ax)
                ilow = ax.value2index(ranges[i][0])
                if ranges[i][1] is None:
                    ihigh = 1 + ilow
                else:
                    ihigh = 1 + ax.value2index(ranges[i][1], 
                                               rounding=lambda x: round(x-1))
                slices.append(slice(ilow, ihigh))
            else:
                slices.append(slice(None))
        return tuple(slices)

    def __call__(self, signal, out=None, axes=None):
        axes = self._parse_axes(axes, signal.axes_manager, signal._plot)
        
        natax = signal.axes_manager._get_axes_in_natural_order()
        slices = self._make_slices(natax, axes)
        if out is None:
            roi = signal[slices]
            return roi
        else:
            signal.__getitem__(slices, out=out)
    
    def _parse_axes(self, axes, axes_manager, plot):
        nd = len(axes)
        if isinstance(axes, basestring):
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
                if x.navigate != y.navigate:
                    raise ValueError("Axes need to be in same space")
        else:
            if axes_manager.navigation_dimension >= nd:
                x = axes_manager.navigation_axes[0]
                if nd > 1:
                    y = axes_manager.navigation_axes[1]
            elif axes_manager.signal_dimension >= nd:
                x = axes_manager.signal_axes[0]
                if nd > 1:
                    y = axes_manager.signal_axes[1]
            else:
                raise ValueError("Neither space has %d dimensions" % nd)
        if x.navigate:
            ax = plot.navigator_plot.ax
        else:
            ax = plot.signal_plot.ax
        
        if nd > 1:
            axes = (x,y)
        else:
            axes = (x,)
        return axes, ax


class BasePointROI(BaseROI):
    def _set_coords_from_widget(self, widget):
        c = widget.coordinates
        self.coords = zip(c)


class Point1DROI(BasePointROI):
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
            self.value = value[0,0]

    def __repr__(self):
        return "%s(value=%f)" % (
            self.__class__.__name__,
            self.value)


class Point2DROI(BaseROI):
    x, y = (t.CFloat(t.Undefined),) * 2
    
    def __init__(self, x, y):
        self.x = y
    
    def _get_coords(self):
        return ((self.x, self.y),)
        
    def _set_coords(self, value):
        if self.coords != value:
            self.x, self.y = value[0]

    def _x_changed(self, old, new):
        self.update()
        
    def _y_changed(self, old, new):
        self.update()

    def __repr__(self):
        return "%s(value=%f)" % (
            self.__class__.__name__,
            self.value)
        

class SpanROI(BaseROI):
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
