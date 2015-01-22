import traits.api as t
from hyperspy.events import Events, Event
import hyperspy.interactive
from hyperspy.axes import DataAxis
from hyperspy.drawing.widgets import ResizableDraggableRectangle


class BaseROI(t.HasTraits):
    def __init__(self):
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()
        self.widgets = set()
        self.signal_map = dict()


class RectangularROI(BaseROI):
    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4

    def __init__(self, left, top, right, bottom):
        super(RectangularROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

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

    def update(self):
        if t.Undefined not in (self.top, self.bottom, self.left, self.right):
            if not self.events.roi_changed.suppress:
                self._update_widgets()
            self.events.roi_changed.trigger(self)

    def _update_widgets(self, exclude=set()):
        if not isinstance(exclude, set):
            exclude = set(exclude)
        for w in self.widgets - exclude:
            with w.events.suppress:
                w.set_bounds(left=self.left, bottom=self.bottom, 
                             right=self.right, top=self.top)

    def interactive(self, signal, navigation_signal="same", out=None):
        if navigation_signal == "same":
            navigation_signal = signal
        if navigation_signal is not None:
            self.add_widget(navigation_signal)
        if out is None:
            return hyperspy.interactive.interactive(self.__call__, 
                                         event=self.events.roi_changed,
                                         signal=signal)
        else:
            return hyperspy.interactive.interactive(self.__call__, 
                                         event=self.events.roi_changed,
                                         signal=signal, out=out)

    def _make_slices(self, axes_manager, axes, ranges):
        """
        Utility function to make a slice container that will slice the axes
        in axes_manager. The axis in 'axes[i]' argument will be sliced with 
        'ranges[i]', all other axes with 'slice(None)'.
        """
        slices = []
        for ax in axes_manager._get_axes_in_natural_order():
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
        
        slices = self._make_slices(signal.axes_manager, axes, 
                                       ((self.left, self.right), 
                                        (self.top, self.bottom)))
        if out is None:
            roi = signal[slices]
            return roi
        else:
            signal.__getitem__(slices, out=out)

    def _on_widget_change(self, widget):
        with self.events.suppress:
            self._bounds_check = False
            try:
                self.left, self.top = widget.get_coordinates()
                w, h = widget._get_size_in_axes()
                self.right = self.left + w
                self.bottom = self.top + h
            finally:
                self._bounds_check = True
        self._update_widgets(exclude=(widget,))
        self.events.roi_changed.trigger(self)

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

    def add_widget(self, signal, axes=None, widget=None, color='green'):
        if widget is None:
            widget = ResizableDraggableRectangle(signal.axes_manager)
            widget.color = color
        axes, ax = self._parse_axes(axes, widget.axes_manager, signal._plot)
        
        # Remove existing ROI, if it exsists and axes match
        if self.signal_map.has_key(signal) and \
                self.signal_map[signal][1] == axes:
            self.remove_widget(signal)
        
        if axes is not None:
            # Set DataAxes
            widget.axes = axes
        with widget.events.suppress:
            widget.set_bounds(left=self.left, bottom=self.bottom, 
                              right=self.right, top=self.top)
        if widget.ax is None:
            widget.set_axis(ax)
            
        # Connect widget changes to on_widget_change
        widget.events.changed.connect(self._on_widget_change)
        # When widget closes, remove from internal list
        widget.events.closed.connect(self._remove_widget)
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
        if self.signal_map.has_key(signal):
            w = self.signal_map.pop(signal)[0]
            self._remove_widget(w)

    def __repr__(self):
        return "%s(top=%f, bottom=%f, left=%f, right=%f)" % (
            self.__class__.__name__,
            self.top,
            self.bottom,
            self.left,
            self.right)
