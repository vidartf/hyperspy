# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.



import math
import numpy as np
from sklearn import mixture
from hyperspy import components
from scipy import ndimage
from hyperspy.signal import Signal1DTools
from hyperspy import hspy
from hyperspy.model import Model
from hyperspy.drawing.utils import on_figure_window_close
from hyperspy.decorators import interactive_range_selector
from hyperspy.axes import (AxesManager, DataAxis)
from hyperspy.drawing.widgets import (DraggableHorizontalLine,
                                      DraggableLabel)


sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))     
fwhm2sigma = 1 / (2 * math.sqrt(math.log(2) * 2))

class GaussianModel(Model):

    def __init__(self, spectrum, *args, **kwargs):
        if len(spectrum) != 1:
            raise TypeError('Signal currenty limited to on dimensional data')
        Model.__init__(self, spectrum, *args, **kwargs)
        self.gaussians = []
        self.offset = None
        self._offset_widget = []

    def setup(self, n_gauss=None, offset=None):
        self.coarse_fit(n_gauss)
        if offset is not None:
            self.offset = components.Offset(offset)
            self.offset.set_parameters_not_free()   # Lock offset, the estimation routine is not ideal for our scenario
            self.append(self.offset)
        self.fit()

    def coarse_fit(self, n_gauss):
        f = self.spectrum.data[self.channel_switches]

        # Use a small gaussian kernel to do take the derivatives
        df = ndimage.gaussian_filter1d(f, sigma=1, order=1, mode='wrap')
        df2 = ndimage.gaussian_filter1d(df, sigma=1, order=1, mode='wrap')
    
        # Find peak positions
        signs = np.sign(df)
        imax = np.where((np.diff(signs) != 0) & (signs[:-1] > 0))[0] + 1
        self.centres = self.spectrum.axes_manager[0].index2value(imax)

        if n_gauss is not None and n_gauss < imax.size:
            # Select strongest n_gauss
            (sigmas, amps) = self._calc_sigma_amp()
            imax.sort(key=lambda im: amps[im] * sigmas[im]) # Maybe use negative df2?
            imax = imax[:n_gauss]
            self.centres = self.spectrum.axes_manager[0].index2value(imax)

        # Add Gaussian components
        self._create_from_centres()

    def _calc_sigma_amp(self):
        self._sort_gaussians()

         # Setup nearest neighbour distances
        ds = np.diff(self.centres)
        ds = np.append(ds, ds[-1])
        ds[1:-1] = np.minimum(ds[:-2], ds[1:-1])

        # Estimate sigmas and amplitudes
        sigmas = ds*fwhm2sigma
        ic = self.spectrum.axes_manager[0].value2index(self.centres)
        amps = self.spectrum.data[ic]*sqrt2pi*sigmas

        return (sigmas, amps)


    def _sort_gaussians(self):
        self.gaussians.sort(key=lambda g: g.centre.value)
        for i, g in enumerate(self.gaussians):
            self.centres[i] = g.centre.value


    def _reset_gaussians(self):
        # Remove components from Model
        for g in self.gaussians:
            self.remove(g, False)
        self.gaussians = []


    def _create_from_centres(self):
        self._reset_gaussians()

        # Get sigmas and amplitudes
        (sigmas, amps) = self._calc_sigma_amp()
        
        # Create and add Gaussian components
        for i in xrange(self.centres.size):
            g = components.Gaussian(amps[i], sigmas[i], self.centres[i])
            self.gaussians.append(g)
            self.append(g)

    def _update_gaussians(self):
        self._sort_gaussians()
        (sigmas, amps) = self._calc_sigma_amp()
        # Update sigmas and amps for all
        self.suspend_update()
        for i, g in enumerate(self.gaussians):
            g.A.value = amps[i]
            g.sigma.value = sigmas[i]
            g.centre.value = self.centres[i]
        self.resume_update()

    def _on_pick(self, event):
        lines = [g._model_plot_line.line for g in self.gaussians]
        if event.artist is not None and event.artist not in lines: return
        if not event.mouseevent.dblclick: return

        self._remove_gaussian(lines.index(event.artist))
        del event.artist

    def start_picking_gaussian(self, callback):
        for g in self.gaussians:
            g._model_plot_line.line.set_picker(3)
        self._plot.signal_plot.figure.canvas.mpl_connect('pick_event', callback)

    def verify(self):
        self.plot(True)
        self.enable_adjust_position(show_label=False)
        self.enable_adjust_offset(show_label=False)
        _plot = self.spectrum._plot
        on_figure_window_close(_plot.signal_plot.figure,
                               self._on_verify_complete)
        self.start_picking_gaussian(self._on_pick)

    def _on_verify_complete(self):
        # Update all gaussians parameters based on new centres
        #self._update_gaussians()

        # Do final fitting
        #self.fit() 
        pass

    def fit(self, *args, **kwargs):
        self._update_gaussians()
        self.locked_fit(*args, **kwargs)  # Fit once with locked centres to refine first
        return super(GaussianModel, self).fit(*args, **kwargs)

    def locked_fit(self, parameters=None, *args, **kwargs):
        if parameters is None:
            parameters = ['centre', 'sigma']
        pushed = [(g.centre.free, g.sigma.free, g.A.free) for g in self.gaussians] # copy old locks
        self.set_parameters_not_free(self.gaussians, parameters) # lock sigmas and centres
        r = super(GaussianModel, self).fit(*args, **kwargs) # perform fit
        for g, old in zip(self.gaussians, pushed):  # reset old lock values
            (g.centre.free, g.sigma.free, g.A.free) = old

    def _shift_offset(self, value):
        if math.isnan(value):
            return
        diff = value - self.offset.offset.value
        if diff == 0.0:
            return
        self.suspend_update()
        self.offset.offset.value = value
        for g in self.gaussians:
            g.maximum = g.maximum - diff
        self.resume_update()

    def enable_adjust_offset(self, show_label=True):
        if self.offset is None:
            return

        if (self._plot is None or
                self._plot.is_active() is False):
            self.plot()
        if self._offset_widget:
            self.disable_adjust_offset()
        on_figure_window_close(self._plot.signal_plot.figure,
                               self.disable_adjust_offset)

        set_value = self._shift_offset
        get_value = self.offset.offset._getvalue

        value = get_value()

        # Create an AxesManager for the widget
        range = self._plot.signal_plot.ax.get_ybound()
        range = (min(value, range[0]), max(value, range[1]))
        axis_dict = DataAxis(range[1]-range[0], offset=range[0]).get_axis_dictionary()
        am = AxesManager([axis_dict, ])
        am._axes[0].navigate = True
        try:
            am._axes[0].value = value
        except TraitError:
            # The value is outside of the axis range
            return

        # Create the horizontal line and labels
        if show_label:
            self._offset_widget = [
                DraggableHorizontalLine(am),
                DraggableLabel(am),]
        else:
            self._offset_widget = [
                DraggableHorizontalLine(am),]
        # Store the component to reset its twin when disabling
        # adjust position
        self._offset_widget[-1].component = self.offset
        w = self._offset_widget[-1]
        #w.string = self.offset._get_short_description().replace(
        #    ' component', '')
        w.add_axes(self._plot.signal_plot.ax)
        if show_label:
            self._offset_widget[-2].add_axes(
                self._plot.signal_plot.ax)
        # Create widget -> parameter connection
        am._axes[0].continuous_value = True
        am._axes[0].on_trait_change(set_value, 'value')

    def disable_adjust_offset(self, components=None, fix_them=True):
        """Disables the interactive adjust offset feature

        See also
        --------
        enable_adjust_offset

        """
        while self._offset_widget:
            pw = self._offset_widget.pop()
            if hasattr(pw, 'component'):
                pw.component.offset.twin = None
                del pw.component
            pw.close()
            del pw

    def add_at(self, value):
        """Adds a new Gaussian at the given position. 
        
        Parameters
        ----------
        value : float or int
            The position can either be given as an index (int) or 
            axis value (float)
        """
        if isinstance(value, int):
            value = self.spectrum.axes_manager[0].index2value(value)

        # Add new centre in sorted order
        self.centres = np.append(self.centres, value)
        idx = np.argsort(self.centres)
        self.centres = self.centres[idx]

        # Add new Gaussian
        new_idx = np.argmax(idx)
        g = components.Gaussian(1, 1, self.centres[new_idx])
        self.gaussians.insert(new_idx, g)   # Keep gaussians in same order as centres. Could possibly be avoided by using dictionaries
        self.append(g)

        # Update all gaussians parameters
        self._update_gaussians()

    def _remove_gaussian(self, idx, *args, **kwargs):
        np.delete(self.centres, idx)
        g = self.gaussians[idx]
        self.gaussians.remove(g)
        super(GaussianModel, self).remove(g, *args, **kwargs)
        self.update_plot()

    def remove(self, object, *args, **kwargs):
        if object in self.gaussians:
            self._remove_gaussian(self.gaussians.index(object), *args, **kwargs)
        else:
            super(GaussianModel, self).remove(g, *args, **kwargs)
    
    def remove_at(self, value):
        if not isinstance(value, int):
            value = self.spectrum.axes_manager[0].index2value(value)
        idx = (np.abs(self.centres-value)).argmin()

        # Remove gaussian
        self._remove_gaussian(idx)
        
        # Update all gaussians parameters
        self._update_gaussians()