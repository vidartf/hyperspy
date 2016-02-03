# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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


import numpy as np
import matplotlib

from hyperspy.drawing.widgets import ResizableDraggableWidgetBase
from hyperspy.events import Events, Event


def in_interval(number, interval):
    if interval[0] <= number <= interval[1]:
        return True
    else:
        return False


class RangeWidget(ResizableDraggableWidgetBase):

    """RangeWidget is a span-patch based widget, which can be
    dragged and resized by mouse/keys. Basically a wrapper for
    ModifiablepanSelector so that it conforms to the common widget interface.

    For optimized changes of geometry, the class implements two methods
    'set_bounds' and 'set_ibounds', to set the geomtry of the rectangle by
    value and index space coordinates, respectivly.
    """

    def __init__(self, axes_manager):
        super(RangeWidget, self).__init__(axes_manager)
        self.span = None

    def set_on(self, value):
        if value is not self.is_on() and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                self.disconnect()
            try:
                self.ax.figure.canvas.draw()
            except:  # figure does not exist
                pass
            if value is False:
                self.ax = None
        self._WidgetBase__is_on = value

    def _add_patch_to(self, ax):
        self.span = ModifiableSpanSelector(ax)
        self.span.set_initial(self._get_range())
        self.span.bounds_check = True
        self.span.snap_position = self.snap_position
        self.span.snap_size = self.snap_size
        self.span.can_switch = True
        self.span.events.changed.connect(self._span_changed, {'obj': 'widget'})
        self.span.step_ax = self.axes[0]
        self.span.tolerance = 5
        self.patch = [self.span.rect]

    def _span_changed(self, widget):
        r = self._get_range()
        pr = widget.range
        if r != pr:
            dx = self.axes[0].scale
            x = pr[0] + 0.5 * dx
            w = pr[1] + 0.5 * dx - x
            old_position, old_size = self.position, self.size
            self._pos = np.array([x])
            self._size = np.array([w])
            self._apply_changes(old_size=old_size, old_position=old_position)

    def _get_range(self):
        p = self._pos[0]
        w = self._size[0]
        offset = self.axes[0].scale
        p -= 0.5 * offset
        return (p, p + w)

    def _parse_bounds_args(self, args, kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
            return args
        elif len(kwargs) == 1 and 'bounds' in kwargs:
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self.indices[0]))
            if 'right' in kwargs:
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width',
                                               self.get_size_in_indices()[0]))
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

        ix, iw = self._parse_bounds_args(args, kwargs)
        x = self.axes[0].index2value(ix)
        w = self._i2v(self.axes[0], ix + iw) - x

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

        if not (self.axes[0].low_value <= x <= self.axes[0].high_value):
            raise ValueError()
        if not (self.axes[0].low_value <= x + w <= self.axes[0].high_value +
                self.axes[0].scale):
            raise ValueError()

        old_position, old_size = self.position, self.size
        self._pos = np.array([x])
        self._size = np.array([w])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on() and self.span is not None:
            self.span.range = self._get_range()

    def disconnect(self):
        super(RangeWidget, self).disconnect()
        if self.span:
            self.span.turn_off()
            self.span = None

    def _set_snap_position(self, value):
        super(RangeWidget, self)._set_snap_position(value)
        self.span.snap_position = value
        self._update_patch_geometry()

    def _set_snap_size(self, value):
        super(RangeWidget, self)._set_snap_size(value)
        self.span.snap_size = value
        self._update_patch_size()


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
        self.bounds_check = False
        self.buttonDown = False
        self.snap_size = False
        self.snap_position = False
        self.events = Events()
        self.events.changed = Event(doc="""
            Event that triggers when the widget was changed.

            Arguments:
            ----------
                obj:
                    The widget that changed
            """, arguments=['obj'])
        self.events.moved = Event(doc="""
            Event that triggers when the widget was moved.

            Arguments:
            ----------
                obj:
                    The widget that changed
            """, arguments=['obj'])
        self.events.resized = Event(doc="""
            Event that triggers when the widget was resized.

            Arguments:
            ----------
                obj:
                    The widget that changed
            """, arguments=['obj'])
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
        x, y = self.rect.get_transform().inverted().transform_point(
            (mouseevent.x, mouseevent.y))
        # Assert y is correct first
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
            if (self.bounds_check and
                    x < self.step_ax.low_value - self.step_ax.scale):
                return
            if self.snap_position:
                snap_offset = self.step_ax.offset- 0.5 * self.step_ax.scale
            elif self.snap_size:
                snap_offset = self._range[1]
            if self.snap_position or self.snap_size:
                rem = (x - snap_offset) % self.step_ax.scale
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
            if (self.bounds_check and
                    x > self.step_ax.high_value + self.step_ax.scale):
                return
            if self.snap_size:
                snap_offset = self._range[0]
                rem = (x - snap_offset) % self.step_ax.scale
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
            if self.snap_position:
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
