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

    def _get_coordinates(self):
        """_get_coordinates() is the getter for the coordinates property, and
        should be implemented by inheritors to return a 2D tuple containing all
        the coordinates that are needed to define the ROI. The tuple should be
        of the structure:
        tuple([tuple( <all 1st axis coordinates> ), \
               tuple( <all 2nd axis coordinates> ), ... ])
        """
        raise NotImplementedError()

    def _set_coordinates(self, value):
        """_set_coordinates is the setter for the coordinates property, and
        should be implemented by inheritors to set its internal represenation
        of the ROI from a tuple of the following structure:
        tuple([tuple( <all 1st axis coordinates> ), \
               tuple( <all 2nd axis coordinates> ), ... ])
        """
        raise NotImplementedError()

    coordinates = property(lambda s: s._get_coordinates(),
                           lambda s, v: s._set_coordinates(v))

    def _get_ndim(self):
        return len(self.coordinates)

    ndim = property(lambda s: s._get_ndim())

    def update(self):
        """Function responsible for updating anything that depends on the ROI.
        It should be called by implementors whenever the ROI changes.
        The base implementation simply triggers the roi_changed event.
        """
        if t.Undefined not in np.ravel(self.coordinates):
            self.events.roi_changed.trigger(self)

    def _get_roi_ranges(self):
        """
        Utility function for getting the slicing ranges spanned by the ROI.
        This base implementation assumes a layout of the 'coordinates'
        attribute:

        coordinates = [axis1, axis2, ...]
            [minimum slice value OR point value,
             maximum slice value (optional)]
        """
        ranges = []
        for i in xrange(self.ndim):
            c = self.coordinates[i]
            if len(c) == 1:
                ranges.append((c[0],))
            elif len(c) == 2:
                ranges.append((c[0], c[1]))
        return ranges

    def _make_slices(self, axes_collecion, axes, ranges=None):
        """
        Utility function to make a slice structure that will slice all the axes
        in 'axes_collecion'. The axes defined in the 'axes' argument will be
        sliced according to the ROI, all other axes in 'axes_collection' will
        be sliced with 'slice(None)', i.e. not sliced.

        Alternatively to using the ROI limits, a set of custom ranges of the
        same shape as 'axes' can be passed. This will slice 'axes[i]' with
        'ranges[i]'.
        """
        if ranges is None:
            ranges = self._get_roi_ranges()
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
        axes = self._parse_axes(axes, signal.axes_manager)
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
        tuple(<DataAxis>)
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

    def _get_coordinates(self):
        return ((self.value,),)

    def _set_coordinates(self, value):
        if self.coordinates != value:
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
        super(Point2DROI, self).__init__()
        self.x, self.y = x, y

    def _get_coordinates(self):
        return ((self.x,), (self.y,))

    def _set_coordinates(self, value):
        if self.coordinates != value:
            (self.x,), (self.y,) = value

    def _x_changed(self, old, new):
        self.update()

    def _y_changed(self, old, new):
        self.update()

    def __repr__(self):
        return "%s(x=%f, y=%f)" % (
            self.__class__.__name__,
            self.x, self.y)


class SpanROI(BaseROI):

    """Selects a range in a 1D space. The coordinates of the range in
    the 1D space are stored in the traits 'left' and 'right'.
    """
    left, right = (t.CFloat(t.Undefined),) * 2

    def __init__(self, left, right):
        super(SpanROI, self).__init__()
        self._bounds_check = True   # Use reponsibly!
        self.left, self.right = left, right

    def _get_coordinates(self):
        return ((self.left, self.right),)

    def _set_coordinates(self, value):
        if self.coordinates != value:
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

    def _get_coordinates(self):
        return (self.left, self.right), (self.top, self.bottom)

    def _set_coordinates(self, value):
        if self.coordinates != value:
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


class Line2DROI(BaseROI):

    x1, y1, x2, y2, linewidth = (t.CFloat(t.Undefined),) * 5

    def __init__(self, x1, y1, x2, y2, linewidth):
        super(Line2DROI, self).__init__()
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.linewidth = linewidth

    def _get_coordinates(self):
        return (self.x1, self.x2), (self.y1, self.y2)

    def _set_coordinates(self, value):
        if self.coordinates != value:
            (self.x1, self.x2), (self.y1, self.y2) = value

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
        coordinates : array, shape (2, N, C), float
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
        axes = self._parse_axes(axes, signal.axes_manager)
        profile = Line2DROI.profile_line(signal.data,
                                         (self.x1, self.y1),
                                         (self.x2, self.y2),
                                         axes=axes,
                                         linewidth=self.linewidth,
                                         order=order)
        length = np.linalg.norm(np.diff(
                np.array(self.coordinates).T, axis=0), axis=1)[0]
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


class CircleROI(BaseROI):

    cx, cy, r, r_inner = (t.CFloat(t.Undefined),) * 4

    def __init__(self, cx, cy, r, r_inner=None):
        super(CircleROI, self).__init__()
        self.cx, self.cy, self.r = cx, cy, r
        if r_inner:
            self.r_inner = r_inner

    def _get_coordinates(self):
        return (self.cx, self.r, self.r_inner), (self.cy,)

    def _set_coordinates(self, value):
        if self.coordinates != value:
            (self.cx, self.r, self.r_inner), (self.cy,) = value

    def _cx_changed(self, old, new):
        self.update()

    def _cy_changed(self, old, new):
        self.update()

    def _r_changed(self, old, new):
        self.update()

    def _r_inner_changed(self, old, new):
        self.update()

    def _get_roi_ranges(self):
        ranges = [[self.cx - self.r, self.cx + self.r],
                  [self.cy - self.r, self.cy + self.r]]
        return ranges

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
        axes = self._parse_axes(axes, signal.axes_manager)
        natax = signal.axes_manager._get_axes_in_natural_order()

        # Slices to slice original data to the circumscribed rectangle
        slices = self._make_slices(natax, axes)
        ir = [slices[natax.index(axes[0])],     # Slices only for axes
              slices[natax.index(axes[1])]]

        # Mask the data in the rectangle that is not within circles

        # Setup axes grid with radius values squared:
        vx = axes[0].axis[ir[0]] - self.cx
        vy = axes[1].axis[ir[1]] - self.cy
        gx, gy = np.meshgrid(vx, vy)
        gr = gx**2 + gy**2
        # Compare values to radii squared to make a 2D mask:
        mask = gr > self.r**2
        if self.r_inner != t.Undefined:
            mask |= gr < self.r_inner**2
        # Broadcast mask through all applicable dimensions (project 2D mask):
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

        # Apply mask, and return OR trigger change events if 'out' supplied
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
