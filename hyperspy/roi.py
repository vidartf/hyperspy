import traits.api as t
from hyperspy.events import Events, Event
import hyperspy.interactive
from hyperspy.drawing.widgets import ResizableDraggableRectangle


class BaseROI(t.HasTraits):
    def __init__(self):
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()
        self.widgets = set()


class RectangularROI(BaseROI):
    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4

    def __init__(self, left, bottom, right, top):
        super(RectangularROI, self).__init__()
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        self._bounds_check = True   # Use reponsibly!

    def _top_changed(self, old, new):
        if self._bounds_check and \
                self.bottom is not t.Undefined and new <= self.bottom:
            self.top = old
        else:
            self.update()

    def _bottom_changed(self, old, new):
        if self._bounds_check and \
                self.top is not t.Undefined and new >= self.top:
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

    def interactive(self, signal):
        return hyperspy.interactive.interactive(signal, self.__call__, 
                                         event=self.events.roi_changed,
                                         signal=signal)

    def __call__(self, signal, out=None):
        if out is None:
            roi = signal[self.left:self.right, self.bottom:self.top]
            return roi
        else:
            signal.__getitem__((slice(self.left, self.right),
                                slice(self.bottom, self.top)),
                               out=out)

    def _on_widget_change(self, widget):
        with self.events.suppress:
            self._bounds_check = False
            try:
                self.left, self.bottom = widget.get_coordinates()
                w, h = widget._get_size_in_axes()
                self.right = self.left + w
                self.top = self.bottom + h
            finally:
                self._bounds_check = True
        self._update_widgets(exclude=(widget,))
        self.events.roi_changed.trigger()

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
            x = axes_manager[axes[0]]
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

    def add_widget(self, signal, axes=None, widget=None):
        if widget is None:
            widget = ResizableDraggableRectangle(signal.axes_manager)
            widget.color = 'green'
        axes, ax = self._parse_axes(axes, widget.axes_manager, signal._plot)
        if axes is not None:
            widget.xaxis = axes[0]
            widget.yaxis = axes[1]
        with widget.events.suppress:
            widget.set_bounds(left=self.left, bottom=self.bottom, 
                              right=self.right, top=self.top)
        if widget.ax is None:
            widget.set_axes(ax)
            
        # Connect widget changes to on_widget_change
        widget.events.changed.connect(self._on_widget_change)
        self.widgets.add(widget)

    def __repr__(self):
        return "%s(top=%f, bottom=%f, left=%f, right=%f)" % (
            self.__class__.__name__,
            self.top,
            self.bottom,
            self.left,
            self.right)
