import traits.api as t
import numpy as np
from hyperspy.events import Events, Event
import hyperspy.interactive
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
    
    coords = property(lambda s: s._get_coords(), lambda s,v: s._set_coords(v))

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
            ranges= []
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
    
    def _parse_axes(self, axes, axes_manager, plot):
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
        plot : MPL_HyperExplorer
            The space of the first DataAxis in axes will be used to extract the
            matplotlib Axes.
        
        Returns
        -------
        (tuple(<DataAxis>), matplotlib Axes)
        """
        nd = len(axes)
        if isinstance(axes, basestring) and (axes.startswith("nav") or \
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
                # We probably have a navigator plot icluding both nav and sig
                # axes. Use navigator plot.
                ax = plot.navigator_plot.ax
                x = axes_manager.signal_axes[0]
                y = axes_manager.navigation_axes[0]
            else:
                raise ValueError("Could not find valid axes configuration.")
        
        if nd > 1 and x.navigate != y.navigate:
            # Here we assume that the navigator plot includes one of 
            # the signal dimensions, and that the user wants to have 
            # the ROI there.
            ax = plot.navigator_plot.ax
        elif x.navigate:
            ax = plot.navigator_plot.ax
        else:
            ax = plot.signal_plot.ax
        
        if nd > 1:
            axes = (x,y)
        else:
            axes = (x,)
        return axes, ax


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

    def interactive(self, signal, navigation_signal="same", out=None):
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
            if not self.signal_map.has_key(navigation_signal):
                self.add_widget(navigation_signal)
        if out is None:
            return hyperspy.interactive.interactive(self.__call__, 
                                         event=self.events.roi_changed,
                                         signal=signal)
        else:
            return hyperspy.interactive.interactive(self.__call__, 
                                         event=self.events.roi_changed,
                                         signal=signal, out=out)

    def navigate(self, signal):
        """Make a widget for this ROI and use it as a navigator for passed 
        signal.
        """
        # Check valid plot and navdim >= roi dim
        ndim = len(self.coords)
        if signal._plot is None or \
                            signal.axes_manager.navigation_dimension < ndim:
            raise ValueError("Cannot navigate this signal with %s" % \
                             self.__class__.__name__, signal)

        nav_axes = signal.axes_manager.navigation_axes[0:ndim+1]

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
        self.coords = zip(c, c+s)

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
        axes, ax = self._parse_axes(axes, signal.axes_manager, signal._plot)
        if widget is None:
            widget = self._get_widget_type(axes, signal)(signal.axes_manager)
            widget.color = color
        
        # Remove existing ROI, if it exsists and axes match
        if self.signal_map.has_key(signal) and \
                self.signal_map[signal][1] == axes:
            self.remove_widget(signal)
        
        if axes is not None:
            # Set DataAxes
            widget.axes = axes
        with widget.events.changed.suppress_single(self._on_widget_change):
            self._apply_roi2widget(widget)
        if widget.ax is None:
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
        if self.signal_map.has_key(signal):
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
            self.value = value[0,0]

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



class Point2DROI(BaseInteractiveROI):
    """Selects a single point in a 2D space. The coordinates of the point in 
    the 2D space are stored in the traits 'x' and 'y'.
    """
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

    def _apply_roi2widget(self, widget):
        widget.coordinates = (self.x, self.y)
        
    def _get_widget_type(self, axes, signal):
        return widgets.DraggableSquare

    def __repr__(self):
        return "%s(value=%f)" % (
            self.__class__.__name__,
            self.value)
        

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
