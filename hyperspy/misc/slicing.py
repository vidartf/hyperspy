from operator import attrgetter
from hyperspy.misc.utils import attrsetter
from hyperspy.misc.export_dictionary import parse_flag_string

import numpy as np


def _slice_target(target, dims, both_slices, slice_nav=None, issignal=False):
    """Slices the target if appropriate

    Parameters
    ----------
    target : object
        Target object
    dims : tuple
        (navigation_dimensions, signal_dimensions) of the original object that
        is sliced
    both_slices : tuple
        (original_slices, array_slices) of the operation that is performed
    slice_nav : {bool, None}
        if None, target is returned as-is. Otherwise navigation and signal
        dimensions are sliced for True and False values respectively.
    issignal : bool
        if the target is signal and should be sliced as one
    """
    if slice_nav is None:
        return target
    if target is None:
        return None
    nav_dims, sig_dims = dims
    slices, array_slices = both_slices
    if slice_nav is True:  # check explicitly for safety
        if issignal:
            return target.inav[slices]
        if isinstance(target, np.ndarray):
            return np.atleast_1d(target[tuple(array_slices[:nav_dims])])
        raise ValueError(
            'tried to slice with navigation dimensions, but was neither a signal nor an array')
    if slice_nav is False:  # check explicitly
        if issignal:
            return target.isig[slices]
        if isinstance(target, np.ndarray):
            return np.atleast_1d(target[tuple(array_slices[-sig_dims:])])
        raise ValueError(
            'tried to slice with signal dimensions, but was neither a signal nor an array')
    # return thing


def copy_slice_from_whitelist(_from, _to, dims, both_slices, isNav):
    """Copies things from one object to another, according to whitelist, slicing
    where required.

    Parameters
    ----------
    _from : object
        Original object
    _to : object
        Target object
    dims : tuple
        (navigation_dimensions, signal_dimensions) of the original object that
        is sliced
    both_slices : tuple
        (original_slices, array_slices) of the operation that is performed
    isNav : bool
        if the slicing operation is performed on navigation dimensions of the
        object
    """

    def make_slice_navigation_decision(flags, isnav):
        if isnav:
            if 'inav' in flags:
                return True
            return None
        if 'isig' in flags:
            return False
        return None

    swl = None
    if hasattr(_from, '_slicing_whitelist'):
        swl = _from._slicing_whitelist

    for key, val in _from._whitelist.iteritems():
        if key == 'self':
            target = None
        else:
            target = attrgetter(key)(_from)

        if val is None:
            # attrsetter(_to, key, attrgetter(key)(_from))
            # continue
            flags = []
        else:
            flags_str = val[0]
            flags = parse_flag_string(flags_str)

        if swl is not None and key in swl:
            flags.extend(parse_flag_string(swl[key]))

        if 'init' in flags:
            continue
        if 'id' in flags:
            continue
        if 'inav' in flags or 'isig' in flags:
            slice_nav = make_slice_navigation_decision(flags, isNav)
            result = _slice_target(
                target,
                dims,
                both_slices,
                slice_nav,
                'sig' in flags)
            attrsetter(_to, key, result)
            continue
        else:
            # 'fn' in flag or no flags at all
            attrsetter(_to, key, target)
            continue


class SpecialSlicers(object):

    def __init__(self, obj, isNavigation):
        self.isNavigation = isNavigation
        self.obj = obj

    def __getitem__(self, slices, out=None):
        return self.obj._slicer(slices, self.isNavigation, out=out)


class FancySlicing(object):

    def _get_array_slices(self, slices, isNavigation=None):
        try:
            len(slices)
        except TypeError:
            slices = (slices,)
        _orig_slices = slices

        has_nav = True if isNavigation is None else isNavigation
        has_signal = True if isNavigation is None else not isNavigation

        # Create a deepcopy of self that contains a view of self.data

        nav_idx = [el.index_in_array for el in
                   self.axes_manager.navigation_axes]
        signal_idx = [el.index_in_array for el in
                      self.axes_manager.signal_axes]

        if not has_signal:
            idx = nav_idx
        elif not has_nav:
            idx = signal_idx
        else:
            idx = nav_idx + signal_idx

        # Add support for Ellipsis
        if Ellipsis in _orig_slices:
            _orig_slices = list(_orig_slices)
            # Expand the first Ellipsis
            ellipsis_index = _orig_slices.index(Ellipsis)
            _orig_slices.remove(Ellipsis)
            _orig_slices = (_orig_slices[:ellipsis_index] + [slice(None), ] *
                            max(0, len(idx) - len(_orig_slices)) +
                            _orig_slices[ellipsis_index:])
            # Replace all the following Ellipses by :
            while Ellipsis in _orig_slices:
                _orig_slices[_orig_slices.index(Ellipsis)] = slice(None)
            _orig_slices = tuple(_orig_slices)

        if len(_orig_slices) > len(idx):
            raise IndexError("too many indices")

        slices = np.array([slice(None,)] *
                          len(self.axes_manager._axes))

        slices[idx] = _orig_slices + (slice(None),) * max(
            0, len(idx) - len(_orig_slices))

        array_slices = []
        for slice_, axis in zip(slices, self.axes_manager._axes):
            if (isinstance(slice_, slice) or
                    len(self.axes_manager._axes) < 2):
                array_slices.append(axis._get_array_slices(slice_))
            else:
                if isinstance(slice_, float):
                    slice_ = axis.value2index(slice_)
                array_slices.append(slice_)
        return tuple(array_slices)

    def _slicer(self, slices, isNavigation=None, out=None):
        array_slices = self._get_array_slices(slices, isNavigation)
        if out is None:
            _obj = self._deepcopy_with_new_data(self.data[array_slices])
            for slice_, axis in zip(array_slices, _obj.axes_manager._axes):
                if (isinstance(slice_, slice) or
                        len(self.axes_manager._axes) < 2):
                    axis._slice_me(slice_)
                else:
                    _obj._remove_axis(axis.index_in_axes_manager)
        else:
            out.data = self.data[array_slices]
            _obj = out
            for slice_, axis_src, axis_dst in zip(
                    array_slices, self.axes_manager._axes,
                    out.axes_manager._axes):
                axis_src = axis_src.copy()
                if (isinstance(slice_, slice) or
                        len(self.axes_manager._axes) < 2):
                    axis_src._slice_me(slice_)
                axis_dst.update_from(axis_src, attributes=(
                    "scale", "offset", "size"))

        if hasattr(self, "_additional_slicing_targets"):
            for ta in self._additional_slicing_targets:
                try:
                    t = attrgetter(ta)(self)
                    if out is None:
                        if hasattr(t, '_slicer'):
                            attrsetter(
                                _obj,
                                ta,
                                t._slicer(
                                    slices,
                                    isNavigation))
                    else:
                        target = attrgetter(ta)(_obj)
                        t._slicer(
                            slices,
                            isNavigation,
                            out=target)

                except AttributeError:
                    pass
        _obj.get_dimensions_from_data()
        if out is None:
            return _obj
        else:
            out.events.data_changed.trigger(obj=out)

# vim: textwidth=80
