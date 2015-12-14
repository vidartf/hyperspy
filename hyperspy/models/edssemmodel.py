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

# To do: weight_fraction different for different pixe. (so basckground)
# Calibrate on standard and transfer dictionnary
# k-ratios

# import copy
# import numpy as np
# import math

from hyperspy.models.edsmodel import EDSModel
# import hyperspy.components as create_component


class EDSSEMModel(EDSModel):

    """Build a fit a model

    Parameters
    ----------
    spectrum : an EDSSEMSpectrum instance
    auto_add_lines : boolean
        If True, automatically add Gaussians for all X-rays generated
        in the energy range by an element, using the edsmodel.add_family_lines
        method
    auto_background : boolean
        If True, adds automatically a polynomial order 6 to the model,
        using the edsmodel.add_polynomial_background method.

    """

    def __init__(self, spectrum, auto_background=True,
                 auto_add_lines=True,
                 *args, **kwargs):
        EDSModel.__init__(self, spectrum, auto_add_lines, *args, **kwargs)
        self.background_components = list()
        if auto_background is True:
            self.add_polynomial_background()