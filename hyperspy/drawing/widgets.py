# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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
import traits

from utils import on_figure_window_close
from hyperspy.misc.math_tools import closest_nice_number
from hyperspy.events import Events, Event


class InteractivePatchBase(object):

    """
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
        return self.__is_on

    def set_on(self, value):
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
                self.ax = None
            self.__is_on = value
            try:
                self.ax.figure.canvas.draw()
            except:  # figure does not exist
                pass

    def _set_patch(self):
        pass
        # Must be provided by the subclass

    def _add_patch_to(self, ax):
        self._set_patch()
        ax.add_artist(self.patch)
        self.patch.set_animated(hasattr(ax, 'hspy_fig'))

    def set_axis(self, ax):
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
            canvas.draw()

    def connect(self, ax):
        on_figure_window_close(ax.figure, self.close)
        
    def connect_navigate(self):
        self.axes_manager.connect(self._on_navigate)
        self._navigating = True
        
    def disconnect_navigate(self):
        self.axes_manager.disconnect(self._on_navigate)
        self._navigating = False
        
    def _on_navigate(self, obj, name, old, new):
        pass    # Implement in subclass!

    def disconnect(self, ax):
        for cid in self.cids:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except:
                pass
        if self._navigating:
            self.disconnect_navigate()

    def close(self, window=None):
        self.set_on(False)
        self.events.closed.trigger(self)

    def draw_patch(self, *args):
        if hasattr(self.ax, 'hspy_fig'):
            self.ax.hspy_fig._draw_animated()
        else:
            self.ax.figure.canvas.draw_idle()
        
    def _v2i(self, axis, v):
        try:
            return axis.value2index(v)
        except ValueError:
            if v > axis.high_value:
                return axis.high_index+1
            elif v < axis.low_value:
                return axis.low_index
            else:
                raise
            
class DraggablePatchBase(InteractivePatchBase):
    
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
        return tuple(self._pos) # Don't pass reference, and make it clear
        
    def _set_position(self, value):
        value = self._validate_pos(value)
        if np.any(self._pos != value):
            self._pos = np.array(value)
            self._pos_changed()

    position = property(lambda s: s._get_position(), \
                        lambda s,v: s._set_position(v))
        
    def _pos_changed(self):
        if self._navigating:
            self.disconnect_navigate()
            for i in xrange(len(self.axes)):
                self.axes[i].index = self.position[i]
            self.connect_navigate()
        self.events.moved.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_position()
            
    def _validate_pos(self, pos):
        if len(pos) != len(self.axes):
            raise ValueError()
        for i in xrange(len(pos)):
            if not (self.axes[i].low_index <= pos[i] <= self.axes[i].high_index):
                raise ValueError()
        return pos
            
    def get_coordinates(self):
        coord = []
        for i in xrange(len(self.axes)):
            coord.append(self.axes[i].index2value(self.position[i]))
        return np.array(coord)
    
    def connect(self, ax):
        super(DraggablePatchBase, self).connect(ax)
        canvas = ax.figure.canvas
        self.cids.append(
            canvas.mpl_connect('motion_notify_event', self.onmousemove))
        self.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        self.cids.append(canvas.mpl_connect(
            'button_release_event', self.button_release))
        
    def _on_navigate(self, obj, name, old, new):
        if obj in self.axes:
            i = self.axes.index(obj)
            p = list(self.position)
            p[i] = new
            self.position = tuple(p)    # Use position to trigger events

    def onpick(self, event):
        self.picked = (event.artist is self.patch)

    def onmousemove(self, event):
        """This method must be provided by the subclass"""
        pass

    def _update_patch_position(self):
        """This method must be provided by the subclass"""
        pass
    
    def _update_patch_geometry(self):
        self._update_patch_position()

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        if self.picked is True:
            self.picked = False


class ResizableDraggablePatchBase(DraggablePatchBase):

    def __init__(self, axes_manager):
        super(ResizableDraggablePatchBase, self).__init__(axes_manager)
        self._size = np.array([1])
        self.events.resized = Event()
        
    def _get_size(self):
        return tuple(self._size)
        
    def _set_size(self, value):
        value = np.minimum(value, [ax.size for ax in self.axes])
        value = np.maximum(value, 1)
        if np.any(self._size != value):
            self._size = value
            self._size_changed()
    
    size = property(lambda s: s._get_size(), lambda s,v: s._set_size(v))

    def increase_size(self):
        self._set_size(self._size + 1)

    def decrease_size(self):
        self._set_size(self._size - 1)
            
    def _size_changed(self):
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_size()

    def _get_size_in_axes(self):
        s = list()
        for i in xrange(len(self.axes)):
            s.append(self.axes[i].scale * self._size[i])
        return np.array(s)
    
    def get_centre(self):
        return self._pos + self._size / 2.0

    def _update_patch_size(self):
        """This method must be provided by the subclass"""
        pass
    
    def _update_patch_geometry(self):
        """This method must be provided by the subclass"""
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
        moved = self.position != self.old_position
        resized = self.size != self.old_size
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
            self._update_patch_geometry()
    
                                            
class Patch2DBase(ResizableDraggablePatchBase):
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
        xy = self._get_patch_xy()
        xs, ys = self._get_size_in_axes()
        self.patch = plt.Rectangle(
            xy, xs, ys,
            animated=self.blit,
            fill=False,
            lw=self.border_thickness,
            ec=self.color,
            picker=True,)

    def _get_patch_xy(self):
        return self.get_coordinates() - self._get_size_in_axes() / 2.
            
    def _get_patch_bounds(self):
        # l,b,w,h
        xy = self._get_patch_xy()
        xs, ys = self._get_size_in_axes()
        return (xy[0], xy[1], xs, ys)
    
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

    def __init__(self, axes_manager):
        super(DraggableSquare, self).__init__(axes_manager)
                                            
    def onmousemove(self, event):
        'on mouse motion move the patch if picked'
        if self.picked is True and event.inaxes:
            ix = self.axes[0].value2index(event.xdata)
            iy = self.axes[1].value2index(event.ydata)
            self.position = (ix, iy)

class ResizableDraggableRectangle(Patch2DBase):
    
    def __init__(self, axes_manager, resizers=True):
        super(ResizableDraggableRectangle, self).__init__(axes_manager)
        self.pick_on_frame = False
        self.pick_offset = (0,0)
        self.resize_color = 'lime'
        self.resize_pixel_size = (5,5)  # Set to None to make one data pixel
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
            
    def _parse_bounds_args(self, args, kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
           return args
        elif len(kwargs) == 1 and kwargs.has_key('bounds'):
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self._pos[0]))
            y = kwargs.pop('y', kwargs.pop('top', self._pos[1]))
            if kwargs.has_key('right'):
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width', self._size[0]))
            if kwargs.has_key('bottom'):
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
        if not (self.axes[0].low_index <= x+w <= self.axes[0].high_index):
            raise ValueError()
        if not (self.axes[1].low_index <= y+h <= self.axes[1].high_index):
            raise ValueError()

        old_position, old_size = self.position, self.size
        with self.events.suppress:
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
        w = self._v2i(self.axes[0], x+w) - ix
        h = self._v2i(self.axes[1], y+h) - iy
            
        old_position, old_size = self.position, self.size
        with self.events.suppress:
            self._pos = np.array([ix, iy])
            self._size = np.array([w, h])
        self._apply_changes(old_size=old_size, old_position=old_position)
        
    def _validate_pos(self, value):
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
        coordinates = np.array(self.get_coordinates())
        axsize = self._get_size_in_axes()
        return coordinates - np.array(axsize) / (2.0 * self._size)

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_xy(self._get_patch_xy())
            self._update_resizers()
            self.draw_patch()
        
    def _update_patch_geometry(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_bounds(*self._get_patch_bounds())
            self._update_resizers()
            self.draw_patch()
        
    # ------- Resizers code -------
        
    def _update_resizers(self):
        pos = self._get_resizer_pos()
        rsize = self._get_resizer_size()
        for i in xrange(4):
            self._resizer_handles[i].set_xy(pos[i])
            self._resizer_handles[i].set_width(rsize[0])
            self._resizer_handles[i].set_height(rsize[1])

    def _set_patch(self):
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

    def _get_resizer_size(self):
        invtrans = self.ax.transData.inverted()
        if self.resize_pixel_size is None:
            rsize = self._get_size_in_axes() / self._size
        else:
            rsize = np.abs(invtrans.transform(self.resize_pixel_size) -
                        invtrans.transform((0, 0)))
        return rsize
        

    def _get_resizer_pos(self):
        """
        Get the positions of the four resizer handles
        """
        invtrans = self.ax.transData.inverted()
        border = self.border_thickness
        # Transform the border thickness into data values
        dl = np.abs(invtrans.transform((border, border)) -
                        invtrans.transform((0, 0)))/2
        rsize = self._get_resizer_size()
        xs, ys = self._get_size_in_axes()

        positions = []
        rp = np.array(self._get_patch_xy())
        p = rp - rsize + dl
        positions.append(p)
        p = rp + (xs - dl[0], -rsize[1] + dl[1])
        positions.append(p)
        p = rp + (-rsize[0] + dl[0], ys - dl[1])
        positions.append(p)
        p = rp + (xs - dl[0], ys - dl[1])
        positions.append(p)
        return positions
                
    def set_on(self, value):
        if value is not self.is_on() and self.resizers:
            self._set_resizers(value, self.ax)
        super(ResizableDraggableRectangle, self).set_on(value)

    def _add_patch_to(self, ax):
        super(ResizableDraggableRectangle, self)._add_patch_to(ax)
        if self.resizers:
            self._set_resizers(True, ax)
        
    # ------- End resizers code -------
        
        
    def _validate_geometry(self, x1=None, y1=None):
        xaxis = self.axes[0]
        yaxis = self.axes[1]
        
        # Make sure widget size is not larger than axes
        self._size[0] = min(self._size[0], xaxis.size)
        self._size[1] = min(self._size[1], yaxis.size)
        
        # Make sure x1/y1 is within bounds
        if x1 is None:
            x1 = self.position[0] # Get it if not supplied
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
        super(ResizableDraggableRectangle, self).onpick(event)
        if event.artist in self._resizer_handles:
            corner = self._resizer_handles.index(event.artist)
            self.pick_on_frame = corner
            self.picked = True
        elif self.picked:
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            dx, dy = self._get_size_in_axes() / self._size
            ix = self._v2i(self.axes[0], x + 0.5*dx)
            iy = self._v2i(self.axes[1], y + 0.5*dy)
            p = self.position
            self.pick_offset = (ix-p[0], iy-p[1])
            self.pick_on_frame = False
        
    def onmousemove(self, event):
        'on mouse motion draw the patch if picked'
        if self.picked is True and event.inaxes:
            xaxis = self.axes[0]
            yaxis = self.axes[1]
            dx, dy = self._get_size_in_axes() / self._size
            ix = self._v2i(xaxis, event.xdata + 0.5*dx)
            iy = self._v2i(yaxis, event.ydata + 0.5*dy)
            p = self.position
            ibounds = [p[0], p[1], p[0] + self._size[0], p[1] + self._size[1]]

            old_position, old_size = self.position, self.size
            with self.events.suppress:
                if self.pick_on_frame is not False:
                    posx = None
                    posy = None
                    corner = self.pick_on_frame
                    if corner % 2 == 0: # Left side start
                        if ix > ibounds[2]:    # flipped to right
                            posx = ibounds[2]
                            self._size[0] = ix - ibounds[2]
                            self.pick_on_frame += 1
                        elif ix == ibounds[2]:
                            posx = ix - 1
                            self._size[0] = ibounds[2] - posx
                        else:
                            posx = ix
                            self._size[0] = ibounds[2] - posx
                    else:   # Right side start
                        if ix < ibounds[0]:  # Flipped to left
                            posx = ix
                            self._size[0] = ibounds[0] - posx
                            self.pick_on_frame -= 1
                        else:
                            self._size[0] = ix - ibounds[0]
                    if corner // 2 == 0: # Top side start
                        if iy > ibounds[3]:    # flipped to botton
                            posy = ibounds[3]
                            self._size[1] = iy - ibounds[3]
                            self.pick_on_frame += 2
                        elif iy == ibounds[3]:
                            posy = iy - 1
                            self._size[1] = ibounds[3] - posy
                        else:
                            posy = iy
                            self._size[1] = ibounds[3] - iy
                    else:   # Bottom side start
                        if iy < ibounds[1]:  # Flipped to top
                            posy = iy
                            self._size[1] = ibounds[1] - iy
                            self.pick_on_frame -= 2
                        else:
                            self._size[1] = iy - ibounds[1]
                    if self._size[0] < 1:
                        self._size[0] = 1
                    if self._size[1] < 1:
                        self._size[1] = 1
                    self._validate_geometry(posx, posy)
                else:
                    ix -= self.pick_offset[0]
                    iy -= self.pick_offset[1]
                    self._validate_geometry(ix, iy)
            self._apply_changes(old_size=old_size, old_position=old_position)



class DraggableHorizontalLine(DraggablePatchBase):

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_ydata(self.get_coordinates()[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = ax.axhline(
            self.get_coordinates()[0],
            color=self.color,
            picker=5)

    def onmousemove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            self.position = (self.axes[0].value2index(event.ydata),)


class DraggableVerticalLine(DraggablePatchBase):

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_xdata(self.get_coordinates()[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = ax.axvline(self.get_coordinates()[0],
                                color=self.color,
                                picker=5)

    def onmousemove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            self.position = (self.axes[0].value2index(event.xdata),)


class DraggableLabel(DraggablePatchBase):

    def __init__(self, axes_manager):
        super(DraggableLabel, self).__init__(axes_manager)
        self.string = ''
        self.y = 0.9
        self.text_color = 'black'
        self.bbox = None

    def _update_patch_position(self):
        if self.is_on() and self.patch is not None:
            self.patch.set_x(self.get_coordinates()[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        self.patch = ax.text(
            self.get_coordinates()[0],
            self.y,  # Y value in axes coordinates
            self.string,
            color=self.text_color,
            picker=5,
            transform=trans,
            horizontalalignment='right',
            bbox=self.bbox,
            animated=self.blit)


class Scale_Bar():

    def __init__(self, ax, units, pixel_size=None, color='white',
                 position=None, max_size_ratio=0.25, lw=2, lenght=None,
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
            lenght of the x axis
        lw : int
            The line width
        lenght : {None, float}
            If None the lenght is automatically calculated using the
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
        if lenght is None:
            self.calculate_size(max_size_ratio=max_size_ratio)
        else:
            self.lenght = lenght
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
                    (self.lenght, self.units[1:-1])
            else:
                return r'$\mathbf{%g\,}$\textbf{%s}' % \
                    (self.lenght, self.units)
        else:
            return r'$%g\,$%s' % (self.lenght, self.units)

    def calculate_line_position(self, pad=0.05):
        return ((1 - pad) * self.xmin + pad * self.xmax,
                (1 - pad) * self.ymin + pad * self.ymax)

    def calculate_text_position(self, pad=1 / 100.):
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght / ps, y1

        self.text_position = ((x1 + x2) / 2.,
                              y2 + (self.ymax - self.ymin) / ps * pad)

    def calculate_size(self, max_size_ratio=0.25):
        ps = self.pixel_size if self.pixel_size is not None else 1
        size = closest_nice_number(ps * (self.xmax - self.xmin) *
                                   max_size_ratio)
        self.lenght = size

    def remove(self):
        if self.line is not None:
            self.ax.lines.remove(self.line)
        if self.text is not None:
            self.ax.texts.remove(self.text)

    def plot_scale(self, line_width=1):
        self.remove()
        ps = self.pixel_size if self.pixel_size is not None else 1
        x1, y1 = self.position
        x2, y2 = x1 + self.lenght / ps, y1
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

    def set_lenght(self, lenght):
        color = self.line.get_color()
        self.lenght = lenght
        self.calculate_scale_size()
        self.calculate_text_position()
        self.plot_scale(line_width=self.line.get_linewidth())
        self.set_color(color)

    def set_tex_bold(self):
        self.tex_bold = True
        self.text.set_text(self.get_units_string())
        self.ax.figure.canvas.draw_idle()


def in_interval(number, interval):
    if number >= interval[0] and number <= interval[1]:
        return True
    else:
        return False


class DraggableResizableRange(ResizableDraggablePatchBase):

    def set_on(self, value):
        if value is not self.is_on() and self.ax is not None:
            if value is True:
                self._add_patch_to(self.ax)
                self.connect(self.ax)
            elif value is False:
                self.disconnect(self.ax)
                self.ax = None
            self.__is_on = value
            try:
                self.ax.figure.canvas.draw()
            except:  # figure does not exist
                pass

    def _add_patch_to(self, ax):
        self.patch = ModifiableSpanSelector(ax)
        self.patch.set_initial(self._get_range())
        self.patch.events.changed.connect(self._patch_changed)
    
    def _patch_changed(self, patch):
        r = self._get_range()
        pr = patch.range
        if r != pr:
            dx = self._get_size_in_axes() / self._size
            ix = self._v2i(self.axes[0], pr[0] + 0.5*dx)
            w = self._v2i(self.axes[0], pr[1] + 0.5*dx) - ix
            old_position, old_size = self.position, self.size
            with self.events.suppress:
                self._pos = np.array([ix])
                self._size = np.array([w])
            self._apply_changes(old_size=old_size, old_position=old_position)
            
                                        
    def _get_range(self):
        c = self.get_coordinates()[0]
        w = self._get_size_in_axes()[0]
        return (c, c + w)
        
    def _parse_bounds_args(self, args, kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
           return args
        elif len(kwargs) == 1 and kwargs.has_key('bounds'):
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self._pos[0]))
            if kwargs.has_key('right'):
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
        if not (self.axes[0].low_index <= x+w <= self.axes[0].high_index):
            raise ValueError()
            
        old_position, old_size = self.position, self.size
        with self.events.suppress:
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
        w = self._v2i(self.axes[0], x+w) - ix
            
        old_position, old_size = self.position, self.size
        with self.events.suppress:
            self._pos = np.array([ix])
            self._size = np.array([w])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        if self.is_on() and self.patch is not None:
            self.patch.range = self._get_range()
        

    def disconnect(self, ax):
        super(DraggableResizableRange, self).disconnect(ax)
        if self.patch and self.ax == ax:
            self.patch.turn_off()
            self.patch = None
    

class ModifiableSpanSelector(matplotlib.widgets.SpanSelector):

    def __init__(self, ax, **kwargs):
        onsel = kwargs.pop('onselect', self.dummy)
        matplotlib.widgets.SpanSelector.__init__(
            self, ax, onsel, direction='horizontal', useblit=False, **kwargs)
        # The tolerance in points to pick the rectangle sizes
        self.tolerance = 1
        self.on_move_cid = None
        self._range = None
        self.events = Events()
        self.events.changed = Event()
        self.events.moved = Event()
        self.events.resized = Event()
        
    def dummy(self, *args, **kwargs):
        pass
    
    def _get_range(self):
        self.update_range()
        return self._range
        
    def _set_range(self, value):
        self.update_range()
        if self._range != value:
            resized = (self._range[1] - self._range[0]) != (value[1] - value[0])
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
            self.canvas.mpl_connect('button_release_event', self.mm_on_release))
        self.cids.append(
            self.canvas.mpl_connect('draw_event', self.update_background))
        self.rect.set_visible(True)
        self.update()
        

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
        if (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = True

        # Calculate the point size in data units
        invtrans = self.ax.transData.inverted()
        x_pt = abs((invtrans.transform((1, 0)) -
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

    def move_left(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        # Do not move the left edge beyond the right one.
        if event.xdata >= self._range[1]:
            return
        width_increment = self._range[0] - event.xdata
        self.rect.set_x(event.xdata)
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
        # Do not move the right edge beyond the left one.
        if event.xdata <= self._range[0]:
            return
        width_increment = \
            event.xdata - self._range[1]
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
        self.rect.set_x(self.rect.get_x() + x_increment)
        self.update_range()
        self.pressv = event.xdata
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
