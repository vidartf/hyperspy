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
            self.__is_on = value
            try:
                self.ax.figure.canvas.draw()
            except:  # figure does not exist
                pass
            else:
                self.ax = None

    def _set_patch(self):
        pass
        # Must be provided by the subclass

    def _add_patch_to(self, ax):
        self._set_patch()
        ax.add_patch(self.patch)
        self.patch.set_animated(hasattr(ax, 'hspy_fig'))

    def set_axes(self, ax):
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

    def disconnect(self, ax):
        for cid in self.cids:
            try:
                ax.figure.canvas.mpl_disconnect(cid)
            except:
                pass

    def close(self, window=None):
        self.set_on(False)

    def draw_patch(self, *args):
        if hasattr(self.ax, 'hspy_fig'):
            self.ax.hspy_fig._draw_animated()
        else:
            self.ax.figure.canvas.draw_idle()
            
class DraggablePatchBase(InteractivePatchBase):
    
    def __init__(self, axes_manager):
        super(DraggablePatchBase, self).__init__(axes_manager)
        self._pos = (0,)
        self.events.moved = Event()
        
    def get_position(self):
        return self._pos
        
    def set_position(self, value):
        value = self._validate_pos(value)
        if self._pos != value:
            self._pos = value
            self._pos_changed()
        
    def _pos_changed(self):
        self.events.moved.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_position()
    
    def connect(self, ax):
        super(DraggablePatchBase, self).connect(ax)
        canvas = ax.figure.canvas
        self.cids.append(
            canvas.mpl_connect('motion_notify_event', self.onmousemove))
        self.cids.append(canvas.mpl_connect('pick_event', self.onpick))
        self.cids.append(canvas.mpl_connect(
            'button_release_event', self.button_release))

    def onpick(self, event):
        self.picked = (event.artist is self.patch)

    def onmousemove(self, event):
        """This method must be provided by the subclass"""
        pass

    def _update_patch_position(self):
        """This method must be provided by the subclass"""
        pass

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        if self.picked is True:
            self.picked = False
    


class ResizableDraggablePatchBase(DraggablePatchBase):

    def __init__(self, axes_manager):
        super(ResizableDraggablePatchBase, self).__init__(axes_manager)
        self._size = 1.
        self.events.resized = Event()
        
    def get_size(self):
        return self._size
        
    def _set_size(self, value):
        if self._size != value:
            self._size = value
            self._size_changed()

    def increase_size(self):
        self._set_size(self._size + 1)

    def decrease_size(self):
        if self._size > 1:
            self._set_size(self._size - 1)
            
    def _size_changed(self):
        self.events.resized.trigger(self)
        self.events.changed.trigger(self)
        self._update_patch_size()

    def _update_patch_size(self):
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
                                            
            
class ResizableDraggableRectangle(ResizableDraggablePatchBase):
    
    def __init__(self, axes_manager, resizers=True):
        super(ResizableDraggableRectangle, self).__init__(axes_manager)
        self._pos = (0, 0)
        self._xsize = 1
        self._ysize = 1
        self.pick_on_frame = False
        self.pick_offset = (0,0)
        self.resize_color = 'lime'
        self.resize_pixel_size = (5,5)  # Set to None to make one data pixel
        self._resizers = resizers
        self._resizer_handles = []
        self.border_thickness = 2
        
        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 0:
                self.xaxis = self.axes_manager.navigation_axes[0]
                if self.axes_manager.navigation_dimension > 1:
                    self.yaxis = self.axes_manager.navigation_axes[1]
                else:
                    self.yaxis = self.axes_manager.signal_axes[0]
            else:
                self.xaxis = self.axes_manager.signal_axes[0]
                self.yaxis = self.axes_manager.signal_axes[1]
    
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
            y = kwargs.pop('y', kwargs.pop('bottom', self._pos[1]))
            if kwargs.has_key('right'):
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width', self._xsize))
            if kwargs.has_key('top'):
                h = kwargs.pop('top') - y
            else:
                h = kwargs.pop('h', kwargs.pop('height', self._ysize)) 
            return x, y, w, h
            
    def set_ibounds(self, *args, **kwargs):
        """
        Set bounds by indices. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, bottom, width, height)
         OR
         * 'x'/'left'
         * 'y'/'bottom'
         * 'w'/'width', alternatively 'right'
         * 'h'/'height', alternatively 'top'
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width/height will be kept, not right/top).
        """

        x, y, w, h = self._parse_bounds_args(args, kwargs)
            
        if not (self.xaxis.low_index <= x <= self.xaxis.high_index):
            raise ValueError()
        if not (self.yaxis.low_index <= y <= self.yaxis.high_index):
            raise ValueError()
        if not (self.xaxis.low_index <= x+w <= self.xaxis.high_index):
            raise ValueError()
        if not (self.yaxis.low_index <= y+h <= self.yaxis.high_index):
            raise ValueError()
            
        self._suspend()
        self._pos = (x, y)
        self._xsize = w
        self._ysize = h
        self._resume()
        
    def set_bounds(self, *args, **kwargs):
        """
        Set bounds by values. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, bottom, width, height)
         OR
         * 'x'/'left'
         * 'y'/'bottom'
         * 'w'/'width', alternatively 'right'
         * 'h'/'height', alternatively 'top'
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width/height will be kept, not right/top).
        """

        x, y, w, h = self._parse_bounds_args(args, kwargs)
            
        ix = self.xaxis.value2index(x)
        iy = self.yaxis.value2index(y)
        w = self.xaxis.value2index(x+w, rounding=np.floor) - ix
        h = self.yaxis.value2index(y+h, rounding=np.floor) - iy
            
        self._suspend()
        self._pos = (ix, iy)
        self._xsize = w
        self._ysize = h
        self._resume()
            
    @property
    def position(self):
        return self._pos
        
    @position.setter
    def position(self, value):
        ix = value[0]
        iy = value[1]
        if not (self.xaxis.low_index <= ix <= self.xaxis.high_index):
            raise ValueError()
        if not (self.yaxis.low_index <= iy <= self.yaxis.high_index):
            raise ValueError()
        
        if (ix, iy) != self._pos:
            self._pos = (ix, iy)
            resized = False
            if ix + self._xsize > self.xaxis.high_index:
                self._xsize = self.xaxis.high_index - ix
                resized = True
            if iy + self._ysize > self.yaxis.high_index:
                self._ysize = self.yaxis.high_index - iy
                resized = True
            self.events.moved.trigger(self)
            if resized:
                self.events.resized.trigger(self)
            self.events.changed.trigger(self)
            self._update_patch_geometry()
            
    
    @property
    def width(self):
        return self._xsize
        
    @width.setter
    def width(self, value):
        if value == self._xsize:
            return
        ix = self._pos[0] + value
        if not (self.xaxis.low_index <= ix <= self.xaxis.high_index):
            raise ValueError()
        self._set_xsize(value)
    
    @property
    def height(self):
        return self.yaxis.scale * self._ysize
        
    @height.setter
    def height(self, value):
        if value == self._ysize:
            return
        iy = self._pos[1] + value
        if not (self.yaxis.low_index <= iy <= self.yaxis.high_index):
            raise ValueError()
        self._set_ysize(value)
    
    @property
    def centre(self):
        return np.array((self.left + self.width/2.0,
                         self.bottom + self.height/2.0))
    
    
    
    # --------- Internal functions ---------
    
    # --- Internals that trigger events ---
    
    def _set_size(self, value):
        self._size = value
        if self._xsize != value or self._ysize != value:
            self._xsize = value
            self._ysize = value
            self._validate_geometry()
            self._size_changed()
            
    def _set_xsize(self, xsize):
        if self._xsize == xsize:
            return
        self._xsize = xsize
        self._size = max(self._xsize, self._ysize)
        self._validate_geometry()
        self._size_changed()

    def _increase_xsize(self):
        self._set_xsize(self._xsize + 1)

    def _decrease_xsize(self):
        if self._xsize >= 2:
            self._set_xsize(self._xsize - 1)

    def _set_ysize(self, ysize):
        if self._ysize == ysize:
            return
        self._ysize = ysize
        self._size = max(self._xsize, self._ysize)
        self._validate_geometry()
        self._size_changed()

    def _increase_ysize(self):
        self._set_ysize(self._ysize + 1)

    def _decrease_ysize(self):
        if self._ysize >= 2:
            self._set_ysize(self._ysize - 1)

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
            
    def _suspend(self):
        self._suppressor = self.events.suppress
        self._suppressor.__enter__()
        self._old = (self._pos, self._xsize, self._ysize)
        
    def _resume(self):
        self._suppressor.__exit__(None,None,None)
        moved = self._pos != self._old[0]
        resized = self._xsize != self._old[1] or self._ysize != self._old[2]
        if moved:
            self.events.moved.trigger(self)
        if resized:
            self.events.resized.trigger(self)
            self._size = max(self._xsize, self._ysize)
        if moved or resized:
            self.events.changed.trigger(self)
            self._update_patch_geometry()

    def _set_patch(self):
        xy = self._get_patch_xy()
        xs, ys = self._get_size_in_axes()
        self.patch = plt.Rectangle(
            xy, xs, ys,
            animated=self.blit,
            fill=False,
            lw=2,
            ec=self.color,
            picker=True,)
            
        self._resizer_handles = []
        rsize = self._get_resizer_size()
        pos = self._get_resizer_pos()
        for i in xrange(4):
            r = plt.Rectangle(pos[i], rsize[0], rsize[1], animated=self.blit,
                              fill=True, lw=0, fc=self.resize_color, 
                              picker=True,)
            self._resizer_handles.append(r)

    def _get_size_in_axes(self):
        xs = self.xaxis.scale * self._xsize
        ys = self.yaxis.scale * self._ysize
        return xs, ys
        
    def get_coordinates(self):
        x = self.xaxis.index2value(self.position[0])
        y = self.yaxis.index2value(self.position[1])
        return x, y

    def _get_patch_xy(self):
        coordinates = np.array(self.get_coordinates())
        xs, ys = self._get_size_in_axes()
        return coordinates - (xs / (2.*self._xsize), ys / (2.*self._ysize))
        
    def _get_patch_bounds(self):
        # l,b,w,h
        xy = self._get_patch_xy()
        xs, ys = self._get_size_in_axes()
        return (xy[0], xy[1], xs, ys)
            
    def _update_patch_size(self):
        self._update_patch_geometry()

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
        
    def _update_resizers(self):
        pos = self._get_resizer_pos()
        rsize = self._get_resizer_size()
        for i in xrange(4):
            self._resizer_handles[i].set_xy(pos[i])
            self._resizer_handles[i].set_width(rsize[0])
            self._resizer_handles[i].set_height(rsize[1])
        
    def _set_resizers(self, value, ax):
        if value:
            for r in self._resizer_handles:
                ax.add_artist(r)
                r.set_animated(hasattr(ax, 'hspy_fig'))
        elif ax is not None:
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
        xs, ys = self._get_size_in_axes()
        if self.resize_pixel_size is None:
            dx = xs / self._xsize
            dy = ys / self._ysize
            rsize = (dx, dy)
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
        if self.is_on() and value is False and self.resizers:
            self._set_resizers(False, self.ax)
        super(ResizableDraggableRectangle, self).set_on(value)

    def _add_patch_to(self, ax):
        super(ResizableDraggableRectangle, self)._add_patch_to(ax)
        if self.resizers:
            self._set_resizers(True, ax)
        
    def _validate_geometry(self, x1=None, y1=None):
        xaxis = self.xaxis
        yaxis = self.yaxis
        
        if x1 is None:
            x1 = self.get_position()[0]
        elif x1 < xaxis.low_index:
            x1 = xaxis.low_index
        elif x1 > xaxis.high_index:
            x1 = xaxis.high_index
           
        if y1 is None:
            y1 = self.get_position()[1]
        elif y1 < yaxis.low_index:
            y1 = yaxis.low_index
        elif y1 > yaxis.high_index:
            y1 = yaxis.high_index
            
        x2 = x1 + self._xsize
        y2 = y1 + self._ysize
        if x2 > xaxis.high_index + 1:
            x2 = xaxis.high_index + 1
            x1 = x2 - self._xsize
        if y2 > yaxis.high_index + 1:
            y2 = yaxis.high_index + 1
            y1 = y2 - self._ysize
        
        self._pos = (x1, y1)
        
    def _v2i(self, axis, v):
        try:
            return axis.value2index(v)
        except ValueError:
            return axis.high_index+1
        
    def onpick(self, event):
        super(ResizableDraggableRectangle, self).onpick(event)
        if event.artist in self._resizer_handles:
            corner = self._resizer_handles.index(event.artist)
            self.pick_on_frame = corner
            self.picked = True
        elif self.picked:
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            xs, ys = self._get_size_in_axes()
            dx = xs / self._xsize
            dy = ys / self._ysize
            xaxis = self.xaxis
            yaxis = self.yaxis
            ix = self._v2i(xaxis, x + 0.5*dx)
            iy = self._v2i(yaxis, y + 0.5*dy)
            p = self.get_position()
            self.pick_offset = (ix-p[0], iy-p[1])
            self.pick_on_frame = False
        
    def onmousemove(self, event):
        'on mouse motion draw the patch if picked'
        if self.picked is True and event.inaxes:
            xaxis = self.xaxis
            yaxis = self.yaxis
            xs, ys = self._get_size_in_axes()
            dx = xs / self._xsize
            dy = ys / self._ysize
            ix = self._v2i(xaxis, event.xdata + 0.5*dx)
            iy = self._v2i(yaxis, event.ydata + 0.5*dy)
            p = self.get_position()
            ibounds = [p[0], p[1], p[0] + self._xsize, p[1] + self._ysize]
            self._suspend()
            if self.pick_on_frame is not False:
                posx = None
                posy = None
                corner = self.pick_on_frame
                if corner % 2 == 0: # Left side start
                    if ix > ibounds[2]:    # flipped to right
                        posx = ibounds[2]
                        self._xsize = ix - ibounds[2]
                        self.pick_on_frame += 1
                    elif ix == ibounds[2]:
                        posx = ix - 1
                        self._xsize = ibounds[2] - posx
                    else:
                        posx = ix
                        self._xsize = ibounds[2] - posx
                else:   # Right side start
                    if ix < ibounds[0]:  # Flipped to left
                        posx = ix
                        self._xsize = ibounds[0] - posx
                        self.pick_on_frame -= 1
                    else:
                        self._xsize = ix - ibounds[0]
                if corner // 2 == 0: # Top side start
                    if iy > ibounds[3]:    # flipped to botton
                        posy = ibounds[3]
                        self._ysize = iy - ibounds[3]
                        self.pick_on_frame += 2
                    elif iy == ibounds[3]:
                        posy = iy - 1
                        self._ysize = ibounds[3] - posy
                    else:
                        posy = iy
                        self._ysize = ibounds[3] - iy
                else:   # Bottom side start
                    if iy < ibounds[1]:  # Flipped to top
                        posy = iy
                        self._ysize = ibounds[1] - iy
                        self.pick_on_frame -= 2
                    else:
                        self._ysize = iy - ibounds[1]
                if self._xsize < 1:
                    self._xsize = 1
                if self._ysize < 1:
                    self._ysize = 1
                self._validate_geometry(posx, posy)
            else:
                ix -= self.pick_offset[0]
                iy -= self.pick_offset[1]
                self._validate_geometry(ix, iy)
            self._resume()
            


class DraggableHorizontalLine(DraggablePatchBase):

    def __init__(self, axes_manager):
        super(DraggableHorizontalLine, self).__init__(axes_manager)
        self._2D = False
        # Despise the bug, we use blit for this one because otherwise the
        # it gets really slow

    def _update_patch_position(self):
        if self.patch is not None:
            self.patch.set_ydata(self.axes_manager.coordinates[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = ax.axhline(
            self.axes_manager.coordinates[0],
            color=self.color,
            picker=5)

    def onmousemove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            try:
                self.axes_manager.navigation_axes[0].value = event.ydata
            except traits.api.TraitError:
                # Index out of range, we do nothing
                pass


class DraggableVerticalLine(DraggablePatchBase):

    def __init__(self, axes_manager):
        super(DraggableVerticalLine, self).__init__(axes_manager)

    def _update_patch_position(self):
        if self.patch is not None:
            self.patch.set_xdata(self.axes_manager.coordinates[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        self.patch = ax.axvline(self.axes_manager.coordinates[0],
                                color=self.color,
                                picker=5)

    def onmousemove(self, event):
        'on mouse motion draw the cursor if picked'
        if self.picked is True and event.inaxes:
            try:
                self.axes_manager.navigation_axes[0].value = event.xdata
            except traits.api.TraitError:
                # Index out of range, we do nothing
                pass


class DraggableLabel(DraggablePatchBase):

    def __init__(self, axes_manager):
        super(DraggableLabel, self).__init__(axes_manager)
        self.string = ''
        self.y = 0.9
        self.text_color = 'black'
        self.bbox = None

    def _update_patch_position(self):
        if self.patch is not None:
            self.patch.set_x(self.axes_manager.coordinates[0])
            self.draw_patch()

    def _set_patch(self):
        ax = self.ax
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        self.patch = ax.text(
            self.axes_manager.coordinates[0],
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

    def set_position(self, x, y):
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


class ModifiableSpanSelector(matplotlib.widgets.SpanSelector):

    def __init__(self, ax, **kwargs):
        matplotlib.widgets.SpanSelector.__init__(
            self, ax, direction='horizontal', useblit=False, **kwargs)
        # The tolerance in points to pick the rectangle sizes
        self.tolerance = 1
        self.on_move_cid = None
        self.range = None

    def release(self, event):
        """When the button is realeased, the span stays in the screen and the
        iteractivity machinery passes to modify mode"""
        if self.pressv is None or (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = False
        self.update_range()
        self.onselect()
        # We first disconnect the previous signals
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)

        # And connect to the new ones
        self.cids.append(
            self.canvas.mpl_connect('button_press_event', self.mm_on_press))
        self.cids.append(
            self.canvas.mpl_connect('button_release_event', self.mm_on_release))
        self.cids.append(
            self.canvas.mpl_connect('draw_event', self.update_background))

    def mm_on_press(self, event):
        if (self.ignore(event) and not self.buttonDown):
            return
        self.buttonDown = True

        # Calculate the point size in data units
        invtrans = self.ax.transData.inverted()
        x_pt = abs((invtrans.transform((1, 0)) -
                    invtrans.transform((0, 0)))[0])

        # Determine the size of the regions for moving and stretching
        rect = self.rect
        self.range = rect.get_x(), rect.get_x() + rect.get_width()
        left_region = self.range[0] - x_pt, self.range[0] + x_pt
        right_region = self.range[1] - x_pt, self.range[1] + x_pt
        middle_region = self.range[0] + x_pt, self.range[1] - x_pt

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
        self.range = (self.rect.get_x(),
                      self.rect.get_x() + self.rect.get_width())

    def move_left(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        # Do not move the left edge beyond the right one.
        if event.xdata >= self.range[1]:
            return
        width_increment = self.range[0] - event.xdata
        self.rect.set_x(event.xdata)
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def move_right(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        # Do not move the right edge beyond the left one.
        if event.xdata <= self.range[0]:
            return
        width_increment = \
            event.xdata - self.range[1]
        self.rect.set_width(self.rect.get_width() + width_increment)
        self.update_range()
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
        self.update()

    def move_rect(self, event):
        if self.buttonDown is False or self.ignore(event):
            return
        x_increment = event.xdata - self.pressv
        self.rect.set_x(self.rect.get_x() + x_increment)
        self.update_range()
        self.pressv = event.xdata
        if self.onmove_callback is not None:
            self.onmove_callback(*self.range)
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
