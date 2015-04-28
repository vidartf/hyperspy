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

from hyperspy.drawing.marker import MarkerBase


class Text(MarkerBase):

    """Text marker that can be added to the signal figure

    Parameters
    ---------
    x: array or float
        The position of the text in x. If float, the marker is fixed.
        If array, the marker will be updated when navigating. The array should
        have the same dimensions in the nagivation axes.
    y: array or float
        The position of the text in y. see x arguments
    text: array or str
        The text. see x arguments
    kwargs:
        Kewywords argument of axvline valid properties (i.e. recognized by
        mpl.plot).

    Example
    -------
    >>> s = signals.Spectrum(np.arange(100).reshape([10,10]))
    >>> s.plot(navigator='spectrum')
    >>> for i in range(10):
    >>>     m = utils.plot.markers.text(y=range(50,1000,100)[i],
    >>>                                 x=i, text='abcdefghij'[i])
    >>>     s.add_marker(m, plot_on_signal=False)
    >>> m = utils.plot.markers.text(x=5, y=range(7,110, 10),
    >>>                             text=[i for i in 'abcdefghij'])
    >>> s.add_marker(m)

    """

    def __init__(self, x, y, text, **kwargs):
        MarkerBase.__init__(self)
        lp = {}
        lp['color'] = 'black'
        self.marker_properties = lp
        self.set_data(x1=x, y1=y, text=text)
        self.set_marker_properties(**kwargs)

    def update(self):
        if self.auto_update is False:
            return
        self.marker.set_position([self.get_data_position('x1'),
                                  self.get_data_position('y1')])
        self.marker.set_text(self.get_data_position('text'))

    def plot(self):
        if self.ax is None:
            raise AttributeError(
                "To use this method the marker needs to be first add to a " +
                "figure using `s._plot.signal_plot.add_marker(m)` or " +
                "`s._plot.navigator_plot.add_marker(m)`")
        self.marker = self.ax.text(
            self.get_data_position('x1'), self.get_data_position('y1'),
            self.get_data_position('text'), **self.marker_properties)
        self.marker.set_animated(True)
        try:
            self.ax.hspy_fig._draw_animated()
        except:
            pass
