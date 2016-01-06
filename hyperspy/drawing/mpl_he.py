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

from traits.api import Undefined

from hyperspy.drawing import widgets, spectrum, image
from hyperspy.gui.axes import navigation_sliders


class MPL_HyperExplorer(object):

    """

    """

    def __init__(self):
        self.signal_data_function = None
        self.navigator_data_function = None
        self.axes_manager = None
        self.signal_title = ''
        self.navigator_title = ''
        self.signal_plot = None
        self.navigator_plot = None
        self.axis = None
        self.pointer = None
        self._key_nav_cid = None
        self._pointer_nav_dim = None

    def plot_signal(self):
        # This method should be implemented by the subclasses.
        # Doing nothing is good enough for signal_dimension==0 though.
        return

    def plot_navigator(self):
        if self.axes_manager.navigation_dimension == 0:
            return
        if self.navigator_data_function is None:
            return
        if self.navigator_data_function is "slider":
            navigation_sliders(
                self.axes_manager.navigation_axes,
                title=self.signal_title + " navigation sliders")
            return
        if self.navigator_plot is not None:
            self.navigator_plot.plot()
            return
        elif len(self.navigator_data_function().shape) == 1:
            # Create the figure
            sf = spectrum.SpectrumFigure(title=self.signal_title + ' Navigator'
                                         if self.signal_title
                                         else "")
            axis = self.axes_manager.navigation_axes[0]
            sf.xlabel = '%s' % str(axis)
            if axis.units is not Undefined:
                sf.xlabel += ' (%s)' % axis.units
            sf.ylabel = r'$\Sigma\mathrm{data\,over\,all\,other\,axes}$'
            sf.axis = axis.axis
            sf.axes_manager = self.axes_manager
            self.navigator_plot = sf
            # Create a line to the left axis with the default
            # indices
            sl = spectrum.SpectrumLine()
            sl.data_function = self.navigator_data_function
            sl.set_line_properties(color='blue',
                                   type='step')
            # Add the line to the figure
            sf.add_line(sl)
            sf.plot()
            self.pointer.set_mpl_ax(sf.ax)
            if self.axes_manager.navigation_dimension > 1:
                navigation_sliders(
                    self.axes_manager.navigation_axes,
                    title=self.signal_title + " navigation sliders")
                for axis in self.axes_manager.navigation_axes[:-2]:
                    axis.connect(sf.update)
            self.navigator_plot = sf
        elif len(self.navigator_data_function().shape) >= 2:
            imf = image.ImagePlot()
            imf.data_function = self.navigator_data_function
            # Navigator labels
            if self.axes_manager.navigation_dimension == 1:
                imf.yaxis = self.axes_manager.navigation_axes[0]
                imf.xaxis = self.axes_manager.signal_axes[0]
            elif self.axes_manager.navigation_dimension >= 2:
                imf.yaxis = self.axes_manager.navigation_axes[1]
                imf.xaxis = self.axes_manager.navigation_axes[0]
                if self.axes_manager.navigation_dimension > 2:
                    navigation_sliders(
                        self.axes_manager.navigation_axes,
                        title=self.signal_title + " navigation sliders")
                    for axis in self.axes_manager.navigation_axes[2:]:
                        axis.connect(imf._update)

            imf.title = self.signal_title + ' Navigator'
            imf.plot()
            self.pointer.set_mpl_ax(imf.ax)
            self.navigator_plot = imf

    def close_navigator_plot(self):
        self._disconnect()
        self.navigator_plot.close()

    def is_active(self):
        return True if self.signal_plot.figure else False

    def plot(self, **kwargs):
        if self.pointer is None:
            pointer = self.assign_pointer()
            if pointer is not None:
                self.pointer = pointer(self.axes_manager)
                self.pointer.color = 'red'
            self.plot_navigator()
        self.plot_signal(**kwargs)

    def assign_pointer(self):
        if self.navigator_data_function is None:
            nav_dim = 0
        elif self.navigator_data_function is "slider":
            nav_dim = 0
        else:
            nav_dim = len(self.navigator_data_function().shape)

        if nav_dim == 2:  # It is an image
            if self.axes_manager.navigation_dimension > 1:
                Pointer = widgets.DraggableSquare
            else:  # It is the image of a "spectrum stack"
                Pointer = widgets.DraggableHorizontalLine
        elif nav_dim == 1:  # It is a spectrum
            Pointer = widgets.DraggableVerticalLine
        else:
            Pointer = None
        self._pointer_nav_dim = nav_dim
        return Pointer

    def _disconnect(self):
        if (self.axes_manager.navigation_dimension > 2 and
                self.navigator_plot is not None):
            for axis in self.axes_manager.navigation_axes:
                axis.disconnect(self.navigator_plot.update)
        if self.pointer is not None:
            self.pointer.disconnect(self.navigator_plot.ax)

    def close(self):
        self._disconnect()
        self.signal_plot.close()
        self.navigator_plot.close()
