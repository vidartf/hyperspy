# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.widgets
import matplotlib.transforms as transforms
import numpy as np

from utils import on_figure_window_close
from hyperspy.misc.math_tools import closest_nice_number
from hyperspy.events import Events, Event


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between @D vectors 'v1' and 'v2'::

            >>> angle_between((1, 0), (0, 1))
            1.5707963267948966
            >>> angle_between((1, 0), (1, 0))
            0.0
            >>> angle_between((1, 0), (-1, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])
    # angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle


class InteractivePatchBase(object):

    """Base class for interactive widgets/patches. A widget creates and
    maintains an matplotlib patch, and manages the interaction code so that the
    user can maniuplate it on the fly.

    This base class implements functionality witch is common to all such
    widgets, mainly the code that manages the patch, axes management, and
    sets up common events ('changed' and 'closed').

    Any inherting subclasses must implement the following method:
        _set_patch(self)
        _on_navigate(obj, name, old, new)

    it should also make sure to initalize the 'axes' attribute as early as
    possible (but after the base class init), so that it is available when
    needed (however, this base class never uses the attribute).
    """

    def __init__(self, axes_manager=None):
        """
        Add a patch to ax.
        """
        self.axes_manager = axes_manager
        self.axes = list()
        self.ax = None
        self.picked = False
        self._size = 1.
        self.color = 'red'
        self.__is_on = True
        self.patch = None
        self.cids = list()
        self.blit = True
        self.background = None
        self.events = Events()
        self.events.changed = Event()
        self.events.closed = Event()
        self._navigating = False

    def is_on(self):
        """Determines if the patch is set to draw if valid.
        """
        return self.__is_on

    def set_on(self, value):
        """Change the on state of the widget. If turning off, all patches will
        be removed from the matplotlib axes, the widget will disconnect from
        all events. If turning on, the patch(es) will be added to the
        matplotlib axes, and the widget will connect to its default events.
        """
        if value is not self.is_on() and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                for container in [
                        self.ax.patches,
                        self.ax.lines,
                        self.ax.artists,
                        self.ax.texts]:
                    if self.patch in container:
                        container.remove(self.patch)
                self.disconnect(self.ax)
            try:
                self.draw_patch()
            except:  # figure does not exist
                pass
            if value is False:
                self.ax = None
        self.__is_on = value

    def _set_patch(self):
        """Create the matplotlib patch, and store it in self.patch
        """
        pass
        # Must be provided by the subclass

    def _add_patch_to(self, ax):
        """Create and add the matplotlib patch to 'ax'
        """
        self._set_patch()
        ax.add_artist(self.patch)
        self.patch.set_animated(hasattr(ax, 'hspy_fig'))

    def set_mpl_ax(self, ax):
        """Set the matplotlib Axes that the widget will draw to. If the widget
        on state is True, it will also add the patch to the Axes, and connect
        to its default events.
        """
        if ax is self.ax:
            return  # Do nothing
        # Disconnect from previous axes if set
        if self.ax is not None and self.is_on():
            self.disconnect(self.ax)
        self.ax = ax
        canvas = ax.figure.canvas
        if self.is_on() is True:
            self._add_patch_to(ax)
            self.connect(ax)
            if self._navigating:
                self.connect_navigate()
            canvas.draw()

    def connect(self, ax):
        """Connect to the matplotlib Axes' events.
        """
        on_figure_window_close(ax.figure, self.close)

    def connect_navigate(self):
        """Connect to the axes_manager such that changes in the widget or in
        the axes_manager are reflected in the other.
        """
        if self._navigating:
            self.disconnect_navigate()
        self.axes_manager.connect(self._on_navigate)
        self._navigating = True

    def disconnect_navigate(self):
        """Disconnect a previous naivgation connection.
        """
        self.axes_manager.disconnect(self._on_navigate)
        self._navigating = False

    def _on_navigate(self, obj, name, old, new):
        """Callback for axes_manager's change notification.
        """
        pass    # Implement in subclass!

    def disconnect(self, ax):
        """Disconnect from all events (both matplotlib and navigation).
        """
        for cid in self.cids:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except:
                pass
        if self._navigating:
            self.disconnect_navigate()

    def close(self, window=None):
        """Set the on state to off (removes patch and disconnects), and trigger
        events.closed.
        """
        self.set_on(False)
        self.events.closed.trigger(self)

    def draw_patch(self, *args):
        """Update the patch drawing.
        """
        if hasattr(self.ax, 'hspy_fig'):
            self.ax.hspy_fig._draw_animated()
        else:
            self.ax.figure.canvas.draw_idle()

    def _v2i(self, axis, v):
        """Wrapped version of DataAxis.value2index, which bounds the index
        inbetween axis.low_index and axis.high_index+1, and does not raise a
        ValueError.
        """
        try:
            return axis.value2index(v)
        except ValueError:
            if v > axis.high_value:
                return axis.high_index + 1
            elif v < axis.low_value:
                return axis.low_index
            else:
                raise


class DraggablePatchBase(InteractivePatchBase):

    """Adds the 'position' and 'coordinates' properties, and adds a framework
    for letting the user drag the patch around. Also adds the 'moved' event.

    The default behavior is that 'coordinates' always is locked to the values
    corresponding to the indices in 'position' (i.e. no subpixel values).

    Any inheritors must override these methods:
        _onmousemove(self, event)
        _update_patch_position(self)
        _set_patch(self)
    """

    def __init__(self, axes_manager):
        super(DraggablePatchBase, self).__init__(axes_manager)
        self._pos = np.array([0])
        self.events.moved = Event()

        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 0:
                self.axes = self.axes_manager.navigation_axes[0:1]
            else:
                self.axes = self.axes_manager.signal_axes[0:1]

    def _get_position(self):
        """Returns a tuple with the position (indices).
        """
        return tuple(
            self._pos.tolist())  # Don't pass reference, and make it clear

    def _set_position(self, value):
        """Sets the position of the widget (by indices). The dimensions should
        correspond to that of the 'axes' attribute. Calls _pos_changed if the
        value has changed, which is then responsible for triggering any
        relevant events.
        """
        value = self._validate_pos(value)
        if np.any(self._pos != value):
            self._pos = np.array(value)
            self._pos_changed()

    position = property(lambda s: s._get_position(),
                        lambda s, v: s._set_position(v))

    def _pos_changed(self):
        """Call when the position of the widget has changed. It triggers the
        relevant events, and updates the patch position.
        """
        if self._navigating:
            self.disconnect_navigate()
            for i in xrange(len(self.axes)):
                self.axes[i].index = self.position[i]
            self.connect_navigate()
        self.events.moved.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_position()

    def _validate_pos(self, pos):
        """Validates the passed position. Depending on the position and the
        implementation, this can either fire a ValueError, or return a modified
        position that has valid values.
        """
        if len(pos) != len(self.axes):
            raise ValueError()
        for i in xrange(len(pos)):
            if not (self.axes[i].low_index <= pos[i] <= self.axes[i].high_index):
                raise ValueError()
        return pos

    def _get_coordinates(self):
        """Providies the position of the widget (by values) in a tuple.
        """
        coord = []
        for i in xrange(len(self.axes)):
            coord.append(self.axes[i].index2value(self.position[i]))
        return tuple(coord)

    def _set_coordinates(self, coordinates):
        """Sets the position of the widget (by values). The dimensions should
        correspond to that of the 'axes' attribute. Calls _pos_changed if the
        value has changed, which is then responsible for triggering any
        relevant events.
        """
        if np.ndim(coordinates) == 0 and len(self.axes) == 1:
            self.position = [self.axes[0].value2index(coordinates)]
        elif len(self.axes) != len(coordinates):
            raise ValueError()
        else:
            p = []
            for i in xrange(len(self.axes)):
                p.append(self.axes[i].value2index(coordinates[i]))
            self.position = p

    coordinates = property(lambda s: s._get_coordinates(),
                           lambda s, v: s._set_coordinates(v))

    def connect(self, ax):
        super(DraggablePatchBase, self).connect(ax)
        canvas = ax.figure.canvas
        self.cids.append(
            canvas.mpl_connect('motion_notify_event', self._onmousemove))
        self.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        self.cids.append(canvas.mpl_connect(
            'button_release_event', self.button_release))

    def _on_navigate(self, obj, name, old, new):
        if obj in self.axes:
            i = self.axes.index(obj)
            p = list(self.position)
            p[i] = new
            self.position = p    # Use position to trigger events

    def onpick(self, event):
        self.picked = (event.artist is self.patch)

    def _onmousemove(self, event):
        """Callback for mouse movement. For dragging, the implementor would
        normally check that the widget is picked, and that the event.inaxes
        Axes equals self.ax.
        """
        # This method must be provided by the subclass
        pass

    def _update_patch_position(self):
        """Updates the position of the patch on the plot.
        """
        # This method must be provided by the subclass
        pass

    def _update_patch_geometry(self):
        """Updates all geometry of the patch on the plot.
        """
        self._update_patch_position()

    def button_release(self, event):
        """whenever a mouse button is released"""
        if event.button != 1:
            return
        if self.picked is True:
            self.picked = False


class ResizableDraggablePatchBase(DraggablePatchBase):

    """Adds the 'size' property and get_size_in_axes method, and adds a
    framework for letting the user resize the patch, including resizing by
    key strokes ('+', '-'). Also adds the 'resized' event.

    Utility functions for resizing are implemented by 'increase_size' and
    'decrease_size', which will in-/decrement the size by 1. Other utility
    functions include 'get_centre' which returns the center position (by
    indices), and the internal _apply_changes which helps make sure that only
    one 'changed' event is fired for a combined move and resize.

    Any inheritors must override these methods:
        _update_patch_position(self)
        _update_patch_size(self)
        _update_patch_geometry(self)
        _set_patch(self)
    """

    def __init__(self, axes_manager):
        super(ResizableDraggablePatchBase, self).__init__(axes_manager)
        self._size = np.array([1])
        self.size_step = 1
        self.events.resized = Event()

    def _get_size(self):
        """Getter for 'size' property. Returns the size as a tuple (to prevent
        unnoticed in-place changes).
        """
        return tuple(self._size.tolist())

    def _set_size(self, value):
        """Setter for the 'size' property. Calls _size_changed to handle size
        change, if the value has changed.
        """
        value = np.minimum(value, [ax.size for ax in self.axes])
        value = np.maximum(value, self.size_step)
        if np.any(self._size != value):
            self._size = value
            self._size_changed()

    size = property(lambda s: s._get_size(), lambda s, v: s._set_size(v))

    def increase_size(self):
        """Increment all sizes by 1. Applied via 'size' property.
        """
        self.size = np.array(self.size) + self.size_step

    def decrease_size(self):
        """Decrement all sizes by 1. Applied via 'size' property.
        """
        self.size = np.array(self.size) - self.size_step

    def _size_changed(self):
        """Triggers resize and changed events, and updates the patch.
        """
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_size()

    def get_size_in_axes(self):
        """Gets the size property converted to the value space (via 'axes'
        attribute).
        """
        s = list()
        for i in xrange(len(self.axes)):
            s.append(self.axes[i].scale * self._size[i])
        return np.array(s)

    def get_centre(self):
        """Get's the center position (in index space). The default
        implementation is simply the position + half the size, which should
        work for any symmetric widget, but more advanced widgets will need to
        decide whether to return the center of gravity or the geometrical
        center of the bounds.
        """
        return self._pos + self._size / 2.0

    def _update_patch_size(self):
        """Updates the size of the patch on the plot.
        """
        # This method must be provided by the subclass
        pass

    def _update_patch_geometry(self):
        """Updates all geometry of the patch on the plot.
        """
        # This method must be provided by the subclass
        pass

    def on_key_press(self, event):
        if event.key == "+":
            self.increase_size()
        if event.key == "-":
            self.decrease_size()

    def connect(self, ax):
        super(ResizableDraggablePatchBase, self).connect(ax)
        canvas = ax.figure.canvas
        self.cids.append(canvas.mpl_connect('key_press_event',
                                            self.on_key_press))

    def _apply_changes(self, old_size, old_position):
        """Evalutes whether the widget has been moved/resized, and triggers
        the correct events and updates the patch geometry. This function has
        the advantage that the geometry is updated only once, preventing
        flickering, and the 'changed' event only fires once.
        """
        moved = self.position != old_position
        resized = self.size != old_size
        if moved:
            if self._navigating:
                self.disconnect_navigate()
                for i in xrange(len(self.axes)):
                    self.axes[i].index = self.position[i]
                self.connect_navigate()
            self.events.moved.trigger(self)
        if resized:
            self.events.resized.trigger(self)
        if moved or resized:
            self.events.changed.trigger(self)
            if moved and resized:
                self._update_patch_geometry()
            elif moved:
                self._update_patch_position()
            else:
                self._update_patch_size()


class Patch2DBase(ResizableDraggablePatchBase):

    """A base class for 2D widgets. It sets the right dimensions for size and
    position/coordinates, adds the 'border_thickness' attribute and initalizes
    the 'axes' attribute to the first two navigation axes if possible, if not,
    the two first signal_axes are used. Other than that it mainly supplies
    common utility functions for inheritors, and implements required functions
    for ResizableDraggablePatchBase.

    The implementation for ResizableDraggablePatchBase methods all assume that
    a Rectangle patch will be used, centered on position. If not, the
    inheriting class will have to override those as applicable.
    """

    def __init__(self, axes_manager):
        super(Patch2DBase, self).__init__(axes_manager)
        self._pos = np.array([0, 0])
        self._size = np.array([1, 1])
        self.border_thickness = 2

        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 1:
                self.axes = self.axes_manager.navigation_axes[0:2]
            else:
                self.axes = self.axes_manager.signal_axes[0:2]

    def _set_patch(self):
        """Sets the patch to a matplotlib Rectangle with the correct geometry.
        The geometry is defined by _get_patch_xy, and get_size_in_axes.
        """
        xy = self._get_patch_xy()
        xs, ys = self.get_size_in_axes()
        self.patch = plt.Rectangle(
            xy, xs, ys,
            animated=self.blit,
            fill=False,
            lw=self.border_thickness,
            ec=self.color,
            picker=True,)

    def _get_patch_xy(self):
        """Returns the xy coordinate of the patch. In this implementation, the
        patch is centered on the position.
        """
        return np.array(self.coordinates) - self.get_size_in_axes() / 2.

    def _get_patch_bounds(self):
        """Returns the bounds of the patch in the form of a tuple in the order
        left, top, width, height. In matplotlib, 'bottom' is used instead of
        'top' as the naming assumes an upwards pointing y-axis, meaning the
        lowest value corresponds to bottom. However, our widgets will normally
        only go on images (which has an inverted y-axis by default), so we
        define the lowest value to be termed 'top'.
        """
        xy = self._get_patch_xy()
        xs, ys = self.get_size_in_axes()
        return (xy[0], xy[1], xs, ys)        # x,y,w,h

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_xy(self._get_patch_xy())
            self.draw_patch()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_bounds(*self._get_patch_bounds())
            self.draw_patch()


class DraggableSquare(Patch2DBase):

    """DraggableSquare is a symmetric, Rectangle-patch based widget, which can
    be dragged, and resized by keystrokes/code. As the widget is normally only
    meant to indicate position, the sizing is deemed purely visual, but there
    is nothing that forces this use. However, it should be noted that the outer
    bounds only correspond to pure inices for odd sizes.
    """

    def __init__(self, axes_manager):
        super(DraggableSquare, self).__init__(axes_manager)

    def _onmousemove(self, event):
        """on mouse motion move the patch if picked"""
        if self.picked is True and event.inaxes:
            ix = self.axes[0].value2index(event.xdata)
            iy = self.axes[1].value2index(event.ydata)
            self.position = (ix, iy)


class ResizableDraggableRectangle(Patch2DBase):

    """ResizableDraggableRectangle is a asymmetric, Rectangle-patch based
    widget, which can be dragged and resized by mouse/keys. For resizing by
    mouse, it adds a small Rectangle patch on the outer border of the main
    patch, to serve as resize handles. This feature can be enabled/disabled by
    the 'resizers' property, and the size/color of the handles are set by
    'resize_color'/'resize_pixel_size'.

    For optimized changes of geometry, the class implements two methods
    'set_bounds' and 'set_ibounds', to set the geomtry of the rectangle by
    value and index space coordinates, respectivly. It also adds the 'width'
    and 'height' properties for verbosity.

    For keyboard resizing, 'x'/'c' and 'y'/'u' will increase/decrease the size
    of the rectangle along the first and the second axis, respectively.

    Implements the internal method _validate_geometry to make sure the patch
    will always stay within bounds.
    """

    def __init__(self, axes_manager, resizers=True):
        super(ResizableDraggableRectangle, self).__init__(axes_manager)
        self.pick_on_frame = False
        self.pick_offset = (0, 0)
        self.resize_color = 'lime'
        self.resize_pixel_size = (5, 5)  # Set to None to make one data pixel
        self._resizers = resizers
        self._resizer_handles = []

    # --------- External interface ---------
    @property
    def resizers(self):
        return self._resizers

    @resizers.setter
    def resizers(self, value):
        if self._resizers != value:
            self._resizers = value
            self._set_resizers(value, self.ax)

    def _parse_bounds_args(self, args, kwargs):
        """Internal utility function to parse args/kwargs passed to set_bounds
        and set_ibounds.
        """
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
            return args
        elif len(kwargs) == 1 and 'bounds' in kwargs:
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self._pos[0]))
            y = kwargs.pop('y', kwargs.pop('top', self._pos[1]))
            if 'right' in kwargs:
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width', self._size[0]))
            if 'bottom' in kwargs:
                h = kwargs.pop('bottom') - y
            else:
                h = kwargs.pop('h', kwargs.pop('height', self._size[1]))
            return x, y, w, h

    def set_ibounds(self, *args, **kwargs):
        """
        Set bounds by indices. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, top, width, height)
         OR
         * 'x'/'left'
         * 'y'/'top'
         * 'w'/'width', alternatively 'right'
         * 'h'/'height', alternatively 'bottom'
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width/height will be kept, not right/bottom).
        """

        x, y, w, h = self._parse_bounds_args(args, kwargs)

        if not (self.axes[0].low_index <= x <= self.axes[0].high_index):
            raise ValueError()
        if not (self.axes[1].low_index <= y <= self.axes[1].high_index):
            raise ValueError()
        if not (self.axes[0].low_index <= x + w <= self.axes[0].high_index):
            raise ValueError()
        if not (self.axes[1].low_index <= y + h <= self.axes[1].high_index):
            raise ValueError()

        old_position, old_size = self.position, self.size
        self._pos = np.array([x, y])
        self._size = np.array([w, h])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def set_bounds(self, *args, **kwargs):
        """
        Set bounds by values. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, top, width, height)
         OR
         * 'x'/'left'
         * 'y'/'top'
         * 'w'/'width', alternatively 'right' (x+w)
         * 'h'/'height', alternatively 'bottom' (y+h)
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width/height will be kept, not right/bottom).
        """

        x, y, w, h = self._parse_bounds_args(args, kwargs)
        ix = self.axes[0].value2index(x)
        iy = self.axes[1].value2index(y)
        w = self._v2i(self.axes[0], x + w) - ix
        h = self._v2i(self.axes[1], y + h) - iy

        old_position, old_size = self.position, self.size
        self._pos = np.array([ix, iy])
        self._size = np.array([w, h])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def _validate_pos(self, value):
        """Constrict the position within bounds.
        """
        value = (min(value[0], self.axes[0].high_index - self._size[0] + 1),
                 min(value[1], self.axes[1].high_index - self._size[1] + 1))
        return super(ResizableDraggableRectangle, self)._validate_pos(value)

    @property
    def width(self):
        return self._size[0]

    @width.setter
    def width(self, value):
        if value == self._size[0]:
            return
        ix = self._pos[0] + value
        if value <= 0 or \
                not (self.axes[0].low_index <= ix <= self.axes[0].high_index):
            raise ValueError()
        self._set_a_size(0, value)

    @property
    def height(self):
        return self._size[1]

    @height.setter
    def height(self, value):
        if value == self._size[1]:
            return
        iy = self._pos[1] + value
        if value <= 0 or \
                not (self.axes[1].low_index <= iy <= self.axes[1].high_index):
            raise ValueError()
        self._set_a_size(1, value)

    # --------- Internal functions ---------

    # --- Internals that trigger events ---

    def _set_size(self, value):
        value = np.minimum(value, [ax.size for ax in self.axes])
        value = np.maximum(value, 1)
        if np.any(self._size != value):
            self._size = value
            self._validate_geometry()
            self._size_changed()

    def _set_a_size(self, idx, value):
        if self._size[idx] == value or value <= 0:
            return
        # If we are pushed "past" an edge, size towards it
        if self._navigating and self.axes[idx].index > self.position[idx]:
            if value < self._size[idx]:
                self._pos[idx] += self._size[idx] - value

        self._size[idx] = value
        self._validate_geometry()
        self._size_changed()

    def _increase_xsize(self):
        self._set_a_size(0, self._size[0] + 1)

    def _decrease_xsize(self):
        if self._size[0] >= 2:
            self._set_a_size(0, self._size[0] - 1)

    def _increase_ysize(self):
        self._set_a_size(1, self._size[1] + 1)

    def _decrease_ysize(self):
        if self._size[1] >= 2:
            self._set_a_size(1, self._size[1] - 1)

    def on_key_press(self, event):
        if event.key == "x":
            self._increase_xsize()
        elif event.key == "c":
            self._decrease_xsize()
        elif event.key == "y":
            self._increase_ysize()
        elif event.key == "u":
            self._decrease_ysize()
        else:
            super(ResizableDraggableRectangle, self).on_key_press(event)

    # --- End internals that trigger events ---

    def _get_patch_xy(self):
        """Get xy value for Rectangle with position being top left. This value
        deviates from the 'coordinates', as 'coordinates' correspond to the
        center value of the pixel. Here, xy corresponds to the top left of
        the pixel.
        """
        coordinates = np.array(self.coordinates)
        axsize = self.get_size_in_axes()
        return coordinates - np.array(axsize) / (2.0 * self._size)

    def _update_patch_position(self):
        # Override to include resizer positioning
        if self.is_on() and self.patch is not None:
            self.patch.set_xy(self._get_patch_xy())
            self._update_resizers()
            self.draw_patch()

    def _update_patch_geometry(self):
        # Override to include resizer positioning
        if self.is_on() and self.patch is not None:
            self.patch.set_bounds(*self._get_patch_bounds())
            self._update_resizers()
            self.draw_patch()

    # ------- Resizers code -------

    def _update_resizers(self):
        """Update resizer handles' patch geometry.
        """
        pos = self._get_resizer_pos()
        rsize = self._get_resizer_size()
        for i, r in enumerate(self._resizer_handles):
            r.set_xy(pos[i])
            r.set_width(rsize[0])
            r.set_height(rsize[1])

    def _set_patch(self):
        """Creates the resizer handles, irregardless of whether they will be
        used or not.
        """
        super(ResizableDraggableRectangle, self)._set_patch()

        self._resizer_handles = []
        rsize = self._get_resizer_size()
        pos = self._get_resizer_pos()
        for i in xrange(4):
            r = plt.Rectangle(pos[i], rsize[0], rsize[1], animated=self.blit,
                              fill=True, lw=0, fc=self.resize_color,
                              picker=True,)
            self._resizer_handles.append(r)

    def _set_resizers(self, value, ax):
        """Turns the resizers on/off, in much the same way that _set_patch
        works.
        """
        if ax is not None:
            if value:
                for r in self._resizer_handles:
                    ax.add_artist(r)
                    r.set_animated(hasattr(ax, 'hspy_fig'))
            else:
                for container in [
                        ax.patches,
                        ax.lines,
                        ax.artists,
                        ax.texts]:
                    for r in self._resizer_handles:
                        if r in container:
                            container.remove(r)
                self._resizer_handles = []
            self.draw_patch()

    def _get_resizer_size(self):
        """Gets the size of the resizer handles in axes coordinates. If
        'resize_pixel_size' is None, a size of one pixel will be used.
        """
        invtrans = self.ax.transData.inverted()
        if self.resize_pixel_size is None:
            rsize = self.get_size_in_axes() / self._size
        else:
            rsize = np.abs(invtrans.transform(self.resize_pixel_size) -
                           invtrans.transform((0, 0)))
        return rsize

    def _get_resizer_pos(self):
        """Get the positions of the four resizer handles
        """
        invtrans = self.ax.transData.inverted()
        border = self.border_thickness
        # Transform the border thickness into data values
        dl = np.abs(invtrans.transform((border, border)) -
                    invtrans.transform((0, 0))) / 2
        rsize = self._get_resizer_size()
        xs, ys = self.get_size_in_axes()

        positions = []
        rp = np.array(self._get_patch_xy())
        p = rp - rsize + dl                         # Top left
        positions.append(p)
        p = rp + (xs - dl[0], -rsize[1] + dl[1])    # Top right
        positions.append(p)
        p = rp + (-rsize[0] + dl[0], ys - dl[1])    # Bottom left
        positions.append(p)
        p = rp + (xs - dl[0], ys - dl[1])           # Bottom right
        positions.append(p)
        return positions

    def set_on(self, value):
        """Same as ancestor, but also turns on/off resizers.
        """
        if value is not self.is_on() and self.resizers:
            self._set_resizers(value, self.ax)
        super(ResizableDraggableRectangle, self).set_on(value)

    def _add_patch_to(self, ax):
        """Same as ancestor, but also adds resizers if 'resizers' property is
        True.
        """
        super(ResizableDraggableRectangle, self)._add_patch_to(ax)
        if self.resizers:
            self._set_resizers(True, ax)

    # ------- End resizers code -------

    def _validate_geometry(self, x1=None, y1=None):
        """Make sure the entire patch always stays within bounds. First the
        position (either from position property or from x1/y1 arguments), is
        limited within the bounds. Then, if the bottom/right edges are out of
        bounds, the position is changed so that they will be at the limit.

        The modified geometry is stored, but no change checks are performed.
        Call _apply_changes after this in order to process any changes (the
        size might change if it is set larger than the bounds size).
        """
        xaxis = self.axes[0]
        yaxis = self.axes[1]

        # Make sure widget size is not larger than axes
        self._size[0] = min(self._size[0], xaxis.size)
        self._size[1] = min(self._size[1], yaxis.size)

        # Make sure x1/y1 is within bounds
        if x1 is None:
            x1 = self.position[0]  # Get it if not supplied
        elif x1 < xaxis.low_index:
            x1 = xaxis.low_index
        elif x1 > xaxis.high_index:
            x1 = xaxis.high_index

        if y1 is None:
            y1 = self.position[1]
        elif y1 < yaxis.low_index:
            y1 = yaxis.low_index
        elif y1 > yaxis.high_index:
            y1 = yaxis.high_index

        # Make sure x2/y2 is with upper bound.
        # If not, keep dims, and change x1/y1!
        x2 = x1 + self._size[0]
        y2 = y1 + self._size[1]
        if x2 > xaxis.high_index + 1:
            x2 = xaxis.high_index + 1
            x1 = x2 - self._size[0]
        if y2 > yaxis.high_index + 1:
            y2 = yaxis.high_index + 1
            y1 = y2 - self._size[1]

        self._pos = np.array([x1, y1])

    def onpick(self, event):
        """Picking of main patch is same as for ancestor, but this also handles
        picking of the resize handles. If a resize handles is picked, 'picked'
        is set to True, and pick_on_frame is set to a integer indicating which
        handle was picked (0-3 for top left, top right, bottom left, bottom
        right).

        If the main patch is picked, the offset from picked pixel to 'position'
        is stored in 'pick_offset'. This is used in _onmousemove to ease
        dragging code.
        """
        super(ResizableDraggableRectangle, self).onpick(event)
        if event.artist in self._resizer_handles:
            corner = self._resizer_handles.index(event.artist)
            self.pick_on_frame = corner
            self.picked = True
        elif self.picked:
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            dx, dy = self.get_size_in_axes() / self._size
            ix = self._v2i(self.axes[0], x + 0.5 * dx)
            iy = self._v2i(self.axes[1], y + 0.5 * dy)
            p = self.position
            self.pick_offset = (ix - p[0], iy - p[1])
            self.pick_on_frame = False

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        # Simple checks to make sure we are dragging our patch:
        if self.picked is True and event.inaxes:
            # Setup reused parameters
            xaxis = self.axes[0]
            yaxis = self.axes[1]
            # Step in value per index increment:
            dx, dy = self.get_size_in_axes() / self._size
            # Mouse position in index space
            ix = self._v2i(xaxis, event.xdata + 0.5 * dx)
            iy = self._v2i(yaxis, event.ydata + 0.5 * dy)
            p = self.position
            # Old bounds in index space
            ibounds = [p[0], p[1], p[0] + self._size[0], p[1] + self._size[1]]

            # Store geometry for _apply_changes at end
            old_position, old_size = self.position, self.size

            if self.pick_on_frame is False:
                # Simply dragging main patch. Offset mouse position by
                # pick_offset to get new position, then validate it.
                ix -= self.pick_offset[0]
                iy -= self.pick_offset[1]
                self._validate_geometry(ix, iy)
            else:
                posx = None     # New x pos. If None, the old pos will be used
                posy = None     # Same for y
                corner = self.pick_on_frame
                if corner % 2 == 0:         # Left side start
                    if ix > ibounds[2]:     # flipped to right
                        posx = ibounds[2]   # New left is old right
                        # New size is mouse position - new left
                        self._size[0] = ix - posx
                        self.pick_on_frame += 1     # Switch pick to right
                    elif ix == ibounds[2]:  # This would give 0 width
                        posx = ix - 1       # So move pos one left from mouse
                        self._size[0] = ibounds[2] - posx   # Should be 1?
                    else:                   # Moving left edge
                        posx = ix           # Set left to mouse index
                        # Keep right still by changing size:
                        self._size[0] = ibounds[2] - posx
                else:                       # Right side start
                    if ix < ibounds[0]:     # Flipped to left
                        posx = ix           # Set left to mouse index
                        # Set size to old left - new left
                        self._size[0] = ibounds[0] - posx
                        self.pick_on_frame -= 1     # Switch pick to left
                    else:                   # Moving right edge
                        # Left should be left as it is, only size updates:
                        self._size[0] = ix - ibounds[0]  # mouse - old left
                if corner // 2 == 0:        # Top side start
                    if iy > ibounds[3]:     # flipped to botton
                        posy = ibounds[3]   # New top is old bottom
                        # New size is mouse position - new top
                        self._size[1] = iy - posy
                        self.pick_on_frame += 2     # Switch pick to bottom
                    elif iy == ibounds[3]:  # This would give 0 height
                        posy = iy - 1       # So move pos one up from mouse
                        self._size[1] = ibounds[3] - posy   # Should be 1?
                    else:                   # Moving top edge
                        posy = iy           # Set top to mouse index
                        # Keep bottom still by changing size:
                        self._size[1] = ibounds[3] - iy  # old bottom - new top
                else:                       # Bottom side start
                    if iy < ibounds[1]:     # Flipped to top
                        posy = iy           # Set top to mouse index
                        # Set size to old top - new top
                        self._size[1] = ibounds[1] - posy
                        self.pick_on_frame -= 2     # Switch pick to top
                    else:                   # Moving bottom edge
                        self._size[1] = iy - ibounds[1]  # mouse - old top
                # If for some reason the size has become less than 0, set to 1
                if self._size[0] < 1:
                    self._size[0] = 1
                if self._size[1] < 1:
                    self._size[1] = 1
                # Validate the geometry
                self._validate_geometry(posx, posy)
            # Finally, apply any changes and trigger events/redraw:
            self._apply_changes(old_size=old_size, old_position=old_position)


class Draggable2DCircle(Patch2DBase):

    """Draggable2DCircle is a symmetric, Cicle-patch based widget, which can
    be dragged, and resized by keystrokes/code.
    """

    def __init__(self, axes_manager):
        super(Draggable2DCircle, self).__init__(axes_manager)
        self.size_step = 0.5

    def _set_size(self, value):
        """Setter for the 'size' property. Calls _size_changed to handle size
        change, if the value has changed.
        """
        # Override so that r_inner can be 0
        value = np.minimum(value, [ax.size for ax in self.axes])
        value = np.maximum(value, (self.size_step, 0))  # Changed from base
        if np.any(self._size != value):
            self._size = value
            self._size_changed()

    def increase_size(self):
        """Increment all sizes by 1. Applied via 'size' property.
        """
        s = np.array(self.size)
        if self.size[1] > 0:
            s += self.size_step
        else:
            s[0] += self.size_step
        self.size = s

    def decrease_size(self):
        """Decrement all sizes by 1. Applied via 'size' property.
        """
        s = np.array(self.size)
        if self.size[1] > 0:
            s -= self.size_step
        else:
            s[0] -= self.size_step
        self.size = s

    def get_centre(self):
        return self._get_patch_xy()

    def _get_patch_xy(self):
        """Returns the xy coordinate of the patch. In this implementation, the
        patch is centered on the position.
        """
        return self.coordinates

    def _set_patch(self):
        """Sets the patch to a matplotlib Circle with the correct geometry.
        The geometry is defined by _get_patch_xy, and get_size_in_axes.
        """
        xy = self._get_patch_xy()
        ro, ri = self.get_size_in_axes()
        self.patch = plt.Circle(
            xy, radius=ro,
            animated=self.blit,
            fill=False,
            lw=self.border_thickness,
            ec=self.color,
            picker=True,)

    def get_size_in_axes(self):
        return np.array(self.axes[0].scale * self._size)

    def _onmousemove(self, event):
        'on mouse motion move the patch if picked'
        # TODO: Switch to delta moves
        if self.picked is True and event.inaxes:
            ix = self.axes[0].value2index(event.xdata)
            iy = self.axes[1].value2index(event.ydata)
            self.position = (ix, iy)

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.center = self._get_patch_xy()
            self.draw_patch()

    def _update_patch_size(self):
        if self.is_on() and self.patch is not None:
            ro, ri = self.get_size_in_axes()
            self.patch.radius = ro
            self.draw_patch()

    def _update_patch_geometry(self):
        if self.is_on() and self.patch is not None:
            ro, ri = self.get_size_in_axes()
            self.patch.center = self._get_patch_xy()
            self.patch.radius = ro
            self.draw_patch()


class DraggableHorizontalLine(DraggablePatchBase):

    """A draggable, horizontal line widget.
    """

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_ydata(self.coordinates[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = ax.axhline(
            self.coordinates[0],
            color=self.color,
            picker=5)

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (self.axes[0].value2index(event.ydata),)


class DraggableVerticalLine(DraggablePatchBase):

    """A draggable, vertical line widget.
    """

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_xdata(self.coordinates[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = ax.axvline(self.coordinates[0],
                                color=self.color,
                                picker=5)

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        if self.picked is True and event.inaxes:
            self.position = (self.axes[0].value2index(event.xdata),)


class DraggableLabel(DraggablePatchBase):

    """A draggable text widget. Adds the attributes 'string', 'text_color' and
    'bbox'. These are all arguments for matplotlib's Text artist. The default
    y-coordinate of the label is set to 0.9.
    """

    def __init__(self, axes_manager):
        super(DraggableLabel, self).__init__(axes_manager)
        self.string = ''
        self.coordinates = (0, 0.9)
        self.text_color = 'black'
        self.bbox = None

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_x(self.coordinates[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        self.patch = ax.text(
            self.coordinates[0],
            self.coordinates[1],
            self.string,
            color=self.text_color,
            picker=5,
            transform=trans,
            horizontalalignment='right',
            bbox=self.bbox,
            animated=self.blit)


class DraggableResizable2DLine(ResizableDraggablePatchBase):

    """A free-form line on a 2D plot. Enables dragging and moving the end
    points, but also allows rotation of the widget by moving the mouse beyond
    the end points of the line.

    The widget adds the 'linewidth' attribute, which is different from the size
    in the following regards: 'linewidth' is simply the width of the patch
    drawn from point to point. If 'size' is greater than 1, it will in
    principle select a rotated rectangle. If 'size' is greater than 4, the
    bounds of this rectangle will be visualized by two dashed lines along the
    outline of this rectangle, instead of a signle line in the center.

    The widget also adds the attributes 'radius_resize', 'radius_move' and
    'radius_rotate' (defaults: 5, 5, 10), which determines the picker radius
    for resizing, aka. moving the edge points (by picking within
    'radius_resize' from an edge point); for moving (by picking within
    'radius_move' from the body of the line); and for rotation (by picking
    within 'radius_rotate' of the edge points on the "outside" of the line).
    The priority is in the order resize, rotate, move; so the 'radius_rotate'
    should always be larger than 'radius_resize' if the function is to be
    accessible (putting it lower is an easy way to disable the functionality).


    NOTE: This widget's internal coordinates does not lock to axes points.
    NOTE: The 'position' is now a 2D tuple: tuple(tuple(x1, x2), tuple(y1, y2))
    NOTE: The 'size' property corresponds to line width, so it has a len() of
    only one.
    """

    # Bitfield values for different mouse interaction functions
    FUNC_NONE = 0       # Do nothing
    FUNC_MOVE = 1       # Move the widget
    FUNC_RESIZE = 2     # Move a vertex
    FUNC_ROTATE = 4     # Rotate
    FUNC_A = 32         # Resize/rotate by first vertex
    FUNC_B = 64         # Resize/rotate by second vertex

    def __init__(self, axes_manager):
        super(DraggableResizable2DLine, self).__init__(axes_manager)
        self._pos = np.array([[0, 0], [0, 0]])
        self._size = np.array([1])
        self.linewidth = 1
        self.radius_move = self.radius_resize = 5
        self.radius_rotate = 10
        self._mfunc = self.FUNC_NONE    # Mouse interaction function
        self._prev_pos = None
        self._rotate_orig = None
        self._width_indicators = []

        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 1:
                self.axes = self.axes_manager.navigation_axes[0:2]
            else:
                self.axes = self.axes_manager.signal_axes[0:2]

    def connect_navigate(self):
        raise NotImplementedError("2D lines cannot be used to navigate yet")

    def _get_position(self):
        # Switched to having internal _pos store in value space, so convert
        # to index space before returning
        ret = tuple()
        for i in xrange(np.shape(self._pos)[0]):
            ret += (tuple(self.axes[i].value2index(self._pos[i, :])), )
        return ret  # Don't pass reference, and make it clear

    def _set_position(self, value):
        # Switched to having internal _pos store in value space, so convert
        # from index space before storing. Validation/events now happens in
        # 'coordinates' setter.
        value = self._validate_pos(np.array(value))
        if np.any(self._pos != value):
            c = []
            for i in xrange(len(self.axes)):
                c.append(self.axes[i].index2value(value[:, i]))
            self.coordinates = np.array(c).T

    def _validate_pos(self, pos):
        """Make sure all vertices are within axis bounds.
        """
        ndim = np.shape(pos)[1]
        if ndim != len(self.axes):
            raise ValueError()
        for i in xrange(ndim):
            if not np.all((self.axes[i].low_index <= pos[:, i]) &
                          (pos[:, i] <= self.axes[i].high_index)):
                raise ValueError()
        return pos

    def _get_coordinates(self):
        return tuple(self._pos.tolist())

    def _set_coordinates(self, coordinates):
        coordinates = self._validate_coords(coordinates)
        if np.any(self._pos != coordinates):
            self._pos = np.array(coordinates)
            self._pos_changed()

    def _validate_coords(self, coords):
        """Make sure all points of 'pos' are within axis bounds.
        """
        coords = np.array(coords)
        ndim = np.shape(coords)[1]
        if ndim != len(self.axes):
            raise ValueError()
        for i in xrange(ndim):
            ax = self.axes[i]
            coords[:, i] = np.maximum(coords[:, i],
                                      ax.low_value - 0.5 * ax.scale)
            coords[:, i] = np.minimum(coords[:, i],
                                      ax.high_value + 0.5 * ax.scale)
        return coords

    def _set_size(self, value):
        """Setter for the 'size' property. Calls _size_changed to handle size
        change, if the value has changed.
        """
        try:
            value[0]
        except TypeError:
            value = np.array([value])
        value = np.maximum(value, 1)
        if np.any(self._size != value):
            self._size = value
            self._size_changed()

    def _get_line_normal(self):
        v = np.diff(self.coordinates, axis=0)   # Line vector
        x = -v[:, 1] * self.axes[0].scale / self.axes[1].scale
        y = v[:, 0] * self.axes[1].scale / self.axes[0].scale
        n = np.array([x, y]).T                    # Normal vector
        return n / np.linalg.norm(n)            # Normalized

    def get_size_in_axes(self):
        """Returns line length in axes coordinates. Requires units on all axes
        to be the same to make any physical sense.
        """
        return np.linalg.norm(np.diff(self.coordinates, axis=0), axis=1)

    def get_centre(self):
        """Get the line center, which is simply the mean position of its
        vertices.
        """
        return np.mean(self._pos, axis=0)

    def _get_width_indicator_coords(self):
        s = self.size * np.array([ax.scale for ax in self.axes])
        n = self._get_line_normal()
        n *= np.linalg.norm(n * s) / 2
        c = np.array(self.coordinates)
        return c + n, c - n

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        """Set line position, and set width indicator's if appropriate
        """
        if self.is_on() and self.patch is not None:
            self.patch.set_data(np.array(self.coordinates).T)
            self.draw_patch()
            wc = self._get_width_indicator_coords()
            for i in xrange(2):
                self._width_indicators[i].set_data(wc[i].T)

    def _set_patch(self):
        """Creates the line, and also creates the width indicators if
        appropriate.
        """
        self.ax.autoscale(False)   # Prevent plotting from rescaling
        xy = np.array(self.coordinates)
        max_r = max(self.radius_move, self.radius_resize,
                    self.radius_rotate)
        self.patch, = self.ax.plot(
            xy[:, 0], xy[:, 1],
            linestyle='-',
            animated=self.blit,
            lw=self.linewidth,
            c=self.color,
            marker='s',
            markersize=self.radius_resize,
            mew=0.1,
            mfc='lime',
            picker=max_r,)
        wc = self._get_width_indicator_coords()
        for i in xrange(2):
            wi, = self.ax.plot(
                wc[i][0], wc[i][1],
                linestyle=':',
                animated=self.blit,
                lw=self.linewidth,
                c=self.color)
            self._width_indicators.append(wi)

    def _set_width_indicators(self, value, ax):
        """Turns the width indicators on/off, in much the same way that
        _set_patch works.
        """
        if ax is not None:
            if value:
                for r in self._width_indicators:
                    ax.add_artist(r)
                    r.set_animated(hasattr(ax, 'hspy_fig'))
            else:
                for container in [
                        ax.patches,
                        ax.lines,
                        ax.artists,
                        ax.texts]:
                    for r in self._width_indicators:
                        if r in container:
                            container.remove(r)
            self.draw_patch()

    def set_on(self, value):
        """Same as ancestor, but also turns on/off width indicators.
        """
        if value is not self.is_on():
            self._set_width_indicators(value, self.ax)
        super(DraggableResizable2DLine, self).set_on(value)

    def _add_patch_to(self, ax):
        """Same as ancestor, but also adds width indicators if 'size' property
        is greater than 4.
        """
        super(DraggableResizable2DLine, self)._add_patch_to(ax)
        self._set_width_indicators(True, ax)

    def _get_vertex(self, event):
        """Check bitfield on self.func, and return vertex index.
        """
        if self.func & self.FUNC_A:
            return 0
        elif self.func & self.FUNC_B:
            return 1
        else:
            return None

    def _get_func_from_pos(self, cx, cy):
        """Get interaction function from pixel position (cx,cy)
        """
        if self.patch is None:
            return self.FUNC_NONE

        trans = self.ax.transData
        p = np.array(trans.transform(self.coordinates))

        # Calculate the distances to the vertecies, and find nearest one
        r2 = np.sum(np.power(p - np.array((cx, cy)), 2), axis=1)
        mini = np.argmin(r2)    # Index of nearest vertex
        minr2 = r2[mini]        # Distance squared to nearest vertex
        del r2
        # Check for resize: Click within radius_resize from edge points
        radius = self.radius_resize
        if minr2 <= radius ** 2:
            ret = self.FUNC_RESIZE
            ret |= self.FUNC_A if mini == 0 else self.FUNC_B
            return ret

        # Check for rotate: Click within radius_rotate on outside of edgepts
        radius = self.radius_rotate
        A = p[0, :]  # Vertex A
        B = p[1, :]  # Vertex B. Assumes one line segment only.
        c = np.array((cx, cy))   # mouse click position
        t = np.dot(c - A, B - A)    # t[0]: A->click, t[1]: A->B
        bas = np.linalg.norm(B - A)**2
        if minr2 <= radius**2:   # If within rotate radius
            if t < 0.0 and mini == 0:   # "Before" A on the line
                return self.FUNC_ROTATE | self.FUNC_A
            elif t > bas and mini == 1:  # "After" B on the line
                return self.FUNC_ROTATE | self.FUNC_B

        # Check for move: Click within radius_move from any point on the line
        radius = self.radius_move
        if 0 < t < bas:
            # A + (t/bas)*(B-A) is closest point on line
            if np.linalg.norm(A + (t / bas) * (B - A) - c) < radius:
                return self.FUNC_MOVE
        return self.FUNC_NONE

    def onpick(self, event):
        """Pick, and if picked, figure out which function to apply. Also store
        pouse position for use by _onmousemove. As rotation does not work very
        well with incremental rotations, the original points are stored if
        we're rotating.
        """
        super(DraggableResizable2DLine, self).onpick(event)
        if self.picked:
            me = event.mouseevent
            self.func = self._get_func_from_pos(me.x, me.y)
            self._prev_pos = [me.xdata, me.ydata]
            if self.func & self.FUNC_ROTATE:
                self._rotate_orig = np.array(self.coordinates)

    def _onmousemove(self, event):
        """Delegate to _move(), _resize() or _rotate().
        """
        if self.picked is True:
            if self.func & self.FUNC_MOVE and event.inaxes:
                self._move(event)
            elif self.func & self.FUNC_RESIZE and event.inaxes:
                self._resize(event)
            elif self.func & self.FUNC_ROTATE:
                self._rotate(event)

    def _get_diff(self, event):
        """Get difference in position in event and what is stored in _prev_pos,
        in value space.
        """
        if event.xdata is None:
            dx = 0
        else:
            dx = event.xdata - self._prev_pos[0]
        if event.ydata is None:
            dy = 0
        else:
            dy = event.ydata - self._prev_pos[1]
        return np.array((dx, dy))

    def _move(self, event):
        """Move line by difference from pick / last mouse move. Update
        '_prev_pos'.
        """
        dx = self._get_diff(event)
        self.coordinates += dx
        self._prev_pos += dx

    def _resize(self, event):
        """Move vertex by difference from pick / last mouse move. Update
        '_prev_pos'.
        """
        ip = self._get_vertex(event)
        dx = self._get_diff(event)
        p = np.array(self.coordinates)
        p[ip, 0:2] += dx
        self.coordinates = p
        self._prev_pos += dx

    def _rotate(self, event):
        """Rotate original points by the angle between mouse position and
        rotation start position (rotation center = line center).
        """
        if None in (event.xdata, event.ydata):
            return
        # Rotate does not update last pos, to avoid inaccuracies by deltas
        dx = self._get_diff(event)

        # Rotation should happen in screen coordinates, as anything else will
        # mix units
        trans = self.ax.transData
        scr_zero = np.array(trans.transform((0, 0)))
        dx = np.array(trans.transform(dx)) - scr_zero

        # Get center point = center of original line
        c = trans.transform(np.mean(self._rotate_orig, axis=0))

        # Figure out theta
        v1 = (event.x, event.y) - c     # Center to mouse
        v2 = v1 - dx                    # Center to start pos
        theta = angle_between(v2, v1)   # Rotation between start and mouse

        if event.key is not None and 'shift' in event.key:
            base = 30 * np.pi / 180
            theta = base * round(float(theta) / base)

        # vector from points to center
        w1 = c - trans.transform(self._rotate_orig)
        # rotate into w2 for next point
        w2 = np.array((w1[:, 0] * np.cos(theta) - w1[:, 1] * np.sin(theta),
                       w1[:, 1] * np.cos(theta) + w1[:, 0] * np.sin(theta)))
        self.coordinates = trans.inverted().transform(c + np.rot90(w2))


class Scale_Bar:

    def __init__(self, ax, units, pixel_size=None, color='white',
                 position=None, max_size_ratio=0.25, lw=2, length=None,
                 animated=False):
        """Add a scale bar to an image.

        Parameteres
        -----------
        ax : matplotlib axes
            The axes where to draw the scale bar.
        units : string
        pixel_size : {None, float}
            If None the axes of the image are supposed to be calibrated.
            Otherwise the pixel size must be specified.
        color : a valid matplotlib color
        position {None, (float, float)}
            If None the position is automatically determined.
        max_size_ratio : float
            The maximum size of the scale bar in respect to the
            length of the x axis
        lw : int
            The line width
        length : {None, float}
            If None the length is automatically calculated using the
            max_size_ratio.

        """

        self.animated = animated
        self.ax = ax
        self.units = units
        self.pixel_size = pixel_size
        self.xmin, self.xmax = ax.get_xlim()
        self.ymin, self.ymax = ax.get_ylim()
        self.text = None
        self.line = None
        self.tex_bold = False
        if length is None:
            self.calculate_size(max_size_ratio=max_size_ratio)
        else:
            self.length = length
        if position is None:
            self.position = self.calculate_line_position()
        else:
            self.position = position
        self.calculate_text_position()
        self.plot_scale(line_width=lw)
        self.set_color(color)

    def get_units_string(self):
        if self.tex_bold is True:
            if (self.units[0] and self.units[-1]) == '$':
                return r'$\mathbf{%g\,%s}$' % \
                    (self.length, self.units[1:-1])
            else:
                return r'$\mathbf{%g\,}$\textbf{%s}' % \
                    (self.length, self.units)
        else:
            return r'$%g\,$%s' % (self.length, self.units)

    def calculate_line_position(self, pad=0.05):
        return ((1 - pad) * self.xmin + pad * self.xmax,
                (1 - pad) * self.ymin + pad * self.ymax)

    def calculate_text_position(self, pad=1 / 100.):
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.length / ps, y1

        self.text_position = ((x1 + x2) / 2.,
                              y2 + (self.ymax - self.ymin) / ps * pad)

    def calculate_size(self, max_size_ratio=0.25):
        ps = self.pixel_size if self.pixel_size is not None else 1
        size = closest_nice_number(ps * (self.xmax - self.xmin) *
                                   max_size_ratio)
        self.length = size

    def remove(self):
        if self.line is not None:
            self.ax.lines.remove(self.line)
        if self.text is not None:
            self.ax.texts.remove(self.text)

    def plot_scale(self, line_width=1):
        self.remove()
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.length / ps, y1
        self.line, = self.ax.plot([x1, x2], [y1, y2],
                                  linestyle='-',
                                  lw=line_width,
                                  animated=self.animated)
        self.text = self.ax.text(*self.text_position,
                                 s=self.get_units_string(),
                                 ha='center',
                                 size='medium',
                                 animated=self.animated)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.figure.canvas.draw()

    def _set_position(self, x, y):
        self.position = x, y
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())

    def set_color(self, c):
        self.line.set_color(c)
        self.text.set_color(c)
        self.ax.figure.canvas.draw_idle()

    def set_length(self, length):
        color = self.line.get_color()
        self.length = length
        self.calculate_scale_size()
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())
        self.set_color(color)

    def set_tex_bold(self):
        self.tex_bold = True
        self.text.set_text(self.get_units_string())
        self.ax.figure.canvas.draw_idle()


def in_interval(number, interval):
    if interval[0] <= number <= interval[1]:
        return True
    else:
        return False


class DraggableResizableRange(ResizableDraggablePatchBase):

    """DraggableResizableRange is a span-patch based widget, which can be
    dragged and resized by mouse/keys. Basically a wrapper for
    ModifiablepanSelector so that it coforms to the common widget interface.

    For optimized changes of geometry, the class implements two methods
    'set_bounds' and 'set_ibounds', to set the geomtry of the rectangle by
    value and index space coordinates, respectivly.
    """

    def __init__(self, axes_manager):
        super(DraggableResizableRange, self).__init__(axes_manager)
        self.span = None

    def set_on(self, value):
        if value is not self.is_on() and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                self.disconnect(self.ax)
            try:
                self.ax.figure.canvas.draw()
            except:  # figure does not exist
                pass
            if value is False:
                self.ax = None
        self._InteractivePatchBase__is_on = value

    def _add_patch_to(self, ax):
        self.span = ModifiableSpanSelector(ax)
        self.span.set_initial(self._get_range())
        self.span.can_switch = True
        self.span.events.changed.connect(self._span_changed, 1)
        self.span.step_ax = self.axes[0]
        self.span.tolerance = 5
        self.patch = self.span.rect

    def _span_changed(self, span):
        r = self._get_range()
        pr = span.range
        if r != pr:
            dx = (self.get_size_in_axes() / self._size)[0]
            ix = self._v2i(self.axes[0], pr[0] + 0.5 * dx)
            w = self._v2i(self.axes[0], pr[1] + 0.5 * dx) - ix
            old_position, old_size = self.position, self.size
            self._pos = np.array([ix])
            self._size = np.array([w])
            self._apply_changes(old_size=old_size, old_position=old_position)

    def _get_range(self):
        c = self.coordinates[0]
        w = self.get_size_in_axes()[0]
        c -= w / (2.0 * self._size[0])
        return (c, c + w)

    def _parse_bounds_args(self, args, kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
            return args
        elif len(kwargs) == 1 and 'bounds' in kwargs:
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self._pos[0]))
            if 'right' in kwargs:
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width', self._size[0]))
            return x, w

    def set_ibounds(self, *args, **kwargs):
        """
        Set bounds by indices. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, width)
         OR
         * 'x'/'left'
         * 'w'/'width', alternatively 'right'
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width will be kept, not right).
        """

        x, w = self._parse_bounds_args(args, kwargs)

        if not (self.axes[0].low_index <= x <= self.axes[0].high_index):
            raise ValueError()
        if not (self.axes[0].low_index <= x + w <= self.axes[0].high_index):
            raise ValueError()

        old_position, old_size = self.position, self.size
        self._pos = np.array([x])
        self._size = np.array([w])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def set_bounds(self, *args, **kwargs):
        """
        Set bounds by values. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, width)
         OR
         * 'x'/'left'
         * 'w'/'width', alternatively 'right' (x+w)
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width will be kept, not right).
        """

        x, w = self._parse_bounds_args(args, kwargs)
        ix = self.axes[0].value2index(x)
        w = self._v2i(self.axes[0], x + w) - ix

        old_position, old_size = self.position, self.size
        self._pos = np.array([ix])
        self._size = np.array([w])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on() and self.span is not None:
            self.span.range = self._get_range()

    def disconnect(self, ax):
        super(DraggableResizableRange, self).disconnect(ax)
        if self.span and self.ax == ax:
            self.span.turn_off()
            self.span = None


class ModifiableSpanSelector(matplotlib.widgets.SpanSelector):

    def __init__(self, ax, **kwargs):
        onsel = kwargs.pop('onselect', self.dummy)
        matplotlib.widgets.SpanSelector.__init__(
            self, ax, onsel, direction='horizontal', useblit=False, **kwargs)
        # The tolerance in points to pick the rectangle sizes
        self.tolerance = 1
        self.on_move_cid = None
        self._range = None
        self.step_ax = None
        self.events = Events()
        self.events.changed = Event()
        self.events.moved = Event()
        self.events.resized = Event()
        self.can_switch = False

    def dummy(self, *args, **kwargs):
        pass

    def _get_range(self):
        self.update_range()
        return self._range

    def _set_range(self, value):
        self.update_range()
        if self._range != value:
            resized = (
                self._range[1] -
                self._range[0]) != (
                value[1] -
                value[0])
            moved = self._range[0] != value[0]
            self._range = value
            if moved:
                self.rect.set_x(value[0])
                self.events.moved.trigger(self)
            if resized:
                self.rect.set_width(value[1] - value[0])
                self.events.resized.trigger(self)
            if moved or resized:
                self.update()
                self.events.changed.trigger(self)

    range = property(_get_range, _set_range)

    def set_initial(self, initial_range=None):
        """
        Remove selection events, set the spanner, and go to modify mode.
        """
        if initial_range is not None:
            self.range = initial_range

        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        # And connect to the new ones
        self.cids.append(
            self.canvas.mpl_connect('button_press_event', self.mm_on_press))
        self.cids.append(
            self.canvas.mpl_connect('button_release_event',
                                    self.mm_on_release))
        self.cids.append(
            self.canvas.mpl_connect('draw_event', self.update_background))
        self.rect.set_visible(True)
        self.rect.contains = self.contains
        self.update()

    def contains(self, mouseevent):
        # Assert y is correct first
        x, y = self.rect.get_transform().inverted().transform_point(
            (mouseevent.x, mouseevent.y))
        if not (0.0 <= y <= 1.0):
            return False, {}
        invtrans = self.ax.transData.inverted()
        x_pt = self.tolerance * abs((invtrans.transform((1, 0)) -
                                     invtrans.transform((0, 0)))[0])
        hit = self._range[0] - x_pt, self._range[1] + x_pt
        if hit[0] < mouseevent.xdata < hit[1]:
            return True, {}
        return False, {}

    def release(self, event):
        """When the button is realeased, the span stays in the screen and the
        iteractivity machinery passes to modify mode"""
        if self.pressv is None or (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = False
        self.update_range()
        self.onselect()
        self.set_initial()

    def mm_on_press(self, event):
        if self.ignore(event) and not self.buttonDown:
            return
        self.buttonDown = True

        # Calculate the point size in data units
        invtrans = self.ax.transData.inverted()
        x_pt = self.tolerance * abs((invtrans.transform((1, 0)) -
                                     invtrans.transform((0, 0)))[0])

        # Determine the size of the regions for moving and stretching
        self.update_range()
        left_region = self._range[0] - x_pt, self._range[0] + x_pt
        right_region = self._range[1] - x_pt, self._range[1] + x_pt
        middle_region = self._range[0] + x_pt, self._range[1] - x_pt

        if in_interval(event.xdata, left_region) is True:
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_left)
        elif in_interval(event.xdata, right_region):
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_right)
        elif in_interval(event.xdata, middle_region):
            self.pressv = event.xdata
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_rect)
        else:
            return

    def update_range(self):
        self._range = (self.rect.get_x(),
                       self.rect.get_x() + self.rect.get_width())

    def switch_left_right(self, x, left_to_right):
        if left_to_right:
            if self.step_ax is not None:
                if x > self.step_ax.high_value + self.step_ax.scale:
                    return
            w = self._range[1] - self._range[0]
            r0 = self._range[1]
            self.rect.set_x(r0)
            r1 = r0 + w
            self.canvas.mpl_disconnect(self.on_move_cid)
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_right)
        else:
            if self.step_ax is not None:
                if x < self.step_ax.low_value - self.step_ax.scale:
                    return
            w = self._range[1] - self._range[0]
            r1 = self._range[0]
            r0 = r1 - w
            self.canvas.mpl_disconnect(self.on_move_cid)
            self.on_move_cid = \
                self.canvas.mpl_connect('motion_notify_event',
                                        self.move_left)
        self._range = (r0, r1)

    def move_left(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x = event.xdata
        if self.step_ax is not None:
            if x < self.step_ax.low_value - self.step_ax.scale:
                return
            rem = (x - self.step_ax.offset - 0.5 * self.step_ax.scale) \
                % self.step_ax.scale
            if rem / self.step_ax.scale < 0.5:
                rem = -rem
            else:
                rem = self.step_ax.scale - rem
            x += rem
        # Do not move the left edge beyond the right one.
        if x >= self._range[1]:
            if self.can_switch and x > self._range[1]:
                self.switch_left_right(x, True)
                self.move_right(event)
            return
        width_increment = self._range[0] - x
        if self.rect.get_width() + width_increment <= 0:
            return
        self.rect.set_x(x)
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        self.events.moved.trigger(self)
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        if self.onmove_callback is not None:
            self.onmove_callback(*self._range)
        self.update()

    def move_right(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x = event.xdata
        if self.step_ax is not None:
            if x > self.step_ax.high_value + self.step_ax.scale:
                return
            rem = (x - self.step_ax.offset + 0.5 * self.step_ax.scale) \
                % self.step_ax.scale
            if rem / self.step_ax.scale < 0.5:
                rem = -rem
            else:
                rem = self.step_ax.scale - rem
            x += rem
        # Do not move the right edge beyond the left one.
        if x <= self._range[0]:
            if self.can_switch and x < self._range[0]:
                self.switch_left_right(x, False)
                self.move_left(event)
            return
        width_increment = x - self._range[1]
        if self.rect.get_width() + width_increment <= 0:
            return
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        if self.onmove_callback is not None:
            self.onmove_callback(*self._range)
        self.update()

    def move_rect(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x_increment = event.xdata - self.pressv
        if self.step_ax is not None:
            rem = x_increment % self.step_ax.scale
            if rem / self.step_ax.scale < 0.5:
                rem = -rem
            else:
                rem = self.step_ax.scale - rem
            x_increment += rem
        self.rect.set_x(self.rect.get_x() + x_increment)
        self.update_range()
        self.pressv += x_increment
        self.events.moved.trigger(self)
        self.events.changed.trigger(self)
        if self.onmove_callback is not None:
            self.onmove_callback(*self._range)
        self.update()

    def mm_on_release(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        self.buttonDown = False
        self.canvas.mpl_disconnect(self.on_move_cid)
        self.on_move_cid = None

    def turn_off(self):
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        if self.on_move_cid is not None:
            self.canvas.mpl_disconnect(cid)
        self.ax.patches.remove(self.rect)
        self.ax.figure.canvas.draw()
