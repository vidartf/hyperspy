import traits.api as t
import numpy as np
from hyperspy.events import Events, Event
import hyperspy.interactive
from hyperspy.axes import DataAxis
from hyperspy.drawing import widgets
                                     


class BaseROI(t.HasTraits):
    def __init__(self):
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()
        self.widgets = set()
        self.signal_map = dict()
       
    def _get_coords(self):
        raise NotImplementedError()
    
    def _set_coords(self, value):
        raise NotImplementedError()
    
    coords = property(lambda s: s._get_coords(), lambda s,v: s._set_coords(v))

    def update(self):
        if t.Undefined not in np.ravel(self.coords):
            if not self.events.roi_changed.suppress:
                self._update_widgets()
            self.events.roi_changed.trigger(self)

    def _update_widgets(self, exclude=set()):
        if not isinstance(exclude, set):
            exclude = set(exclude)
        for w in self.widgets - exclude:
            with w.events.suppress:
                self._apply_roi2widget(w)
            
    def _apply_roi2widget(self, widget):
        raise NotImplementedError()

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
            
    def _set_coords_from_widget(self, widget):
        c = widget.coordinates
        s = widget._get_size_in_axes()
        self.coords = zip(c, c+s)

    def _on_widget_change(self, widget):
        with self.events.suppress:
            self._bounds_check = False
            try:
                self._set_coords_from_widget(widget)
            finally:
                self._bounds_check = True
        self._update_widgets(exclude=(widget,))
        self.events.roi_changed.trigger(self)
        
    def _get_widget_type(self, axes, signal):
        raise NotImplementedError()

    def add_widget(self, signal, axes=None, widget=None, color='green'):
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
        with widget.events.suppress:
            self._apply_roi2widget(widget)
        if widget.ax is None:
            widget.set_mpl_ax(ax)
            
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

    def _apply_roi2widget(self, widget):
        widget.coordinates = self.value
        
    def _get_widget_type(self, axes, signal):
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

    def _apply_roi2widget(self, widget):
        widget.coordinates = (self.x, self.y)
        
    def _get_widget_type(self, axes, signal):
        return widgets.DraggableSquare

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

    def _apply_roi2widget(self, widget):
        widget.set_bounds(left=self.left, right=self.right)
        
    def _get_widget_type(self, axes, signal):
        return widgets.DraggableResizableRange

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
