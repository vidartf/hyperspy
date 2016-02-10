﻿
Tools: the Signal class
***********************

The Signal class and its subclasses
-----------------------------------

.. WARNING::
    This subsection can be a bit confusing for beginners.
    Do not worry if you do not understand it all.


HyperSpy stores the data in the :py:class:`~.signal.Signal` class, that is
the object that you get when e.g. you load a single file using
:py:func:`~.io.load`. Most of the data analysis functions are also contained in
this class or its specialized subclasses. The :py:class:`~.signal.Signal` class
contains general functionality that is available to all the subclasses. The
subclasses provide functionality that is normally specific to a particular type
of data, e.g. the :py:class:`~._signals.spectrum.Spectrum` class provides common
functionality to deal with spectral data and
:py:class:`~._signals.eels.EELSSpectrum` (which is a subclass of
:py:class:`~._signals.spectrum.Spectrum`) adds extra functionality to the
:py:class:`~._signals.spectrum.Spectrum` class for electron energy-loss
spectroscopy data analysis.

Currently the following signal subclasses are available:

* :py:class:`~._signals.spectrum.Spectrum`
* :py:class:`~._signals.image.Image`
* :py:class:`~._signals.eels.EELSSpectrum`
* :py:class:`~._signals.eds_tem.EDSTEMSpectrum`
* :py:class:`~._signals.eds_sem.EDSSEMSpectrum`
* :py:class:`~._signals.spectrum_simulation.SpectrumSimulation`
* :py:class:`~._signals.image_simulation.ImageSimulation`

The :py:mod:`~.signals` module, which contains all available signal subclasses,
is imported in the user namespace when loading hyperspy. In the following
example we create an Image instance from a 2D numpy array:

.. code-block:: python

    >>> im = hs.signals.Image(np.random.random((64,64)))


The different signals store other objects in what are called attributes. For
examples, the data is stored in a numpy array in the
:py:attr:`~.signal.Signal.data` attribute, the original parameters in the
:py:attr:`~.signal.Signal.original_metadata` attribute, the mapped parameters
in the :py:attr:`~.signal.Signal.metadata` attribute and the axes
information (including calibration) can be accessed (and modified) in the
:py:attr:`~.signal.Signal.axes_manager` attribute.


.. _transforming.signal:

Transforming between signal subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The different subclasses are characterized by three
:py:attr:`~.signal.Signal.metadata` attributes (see the table below):

`record_by`
    Can be "spectrum", "image" or "", the latter meaning undefined.
    It describes the way the data is arranged in memory.
    It is possible to transform any :py:class:`~.signal.Signal` subclass in a
    :py:class:`~._signals.spectrum.Spectrum` or :py:class:`~._signals.image.Image`
    subclass using the following :py:class:`~.signal.Signal` methods:
    :py:meth:`~.signal.Signal.as_image` and :py:meth:`~.signal.Signal.as_spectrum`.
    In addition :py:class:`~._signals.spectrum.Spectrum` instances can be
    transformed in images using :py:meth:`~._signals.spectrum.Spectrum.to_image`
    and image instances in spectrum instances using
    :py:meth:`~._signals.image.Image.to_spectrum`. When transforming between
    spectrum and image classes the order in which the
    data array is stored in memory is modified to improve performance. Also,
    some functions, e.g. plotting or decomposing, will behave differently.

`signal_type`
    Describes the nature of the signal. It can be any string, normally the
    acronym associated with a
    particular signal. In certain cases HyperSpy provides features that are
    only available for a
    particular signal type through :py:class:`~.signal.Signal` subclasses.
    The :py:class:`~.signal.Signal` method
    :py:meth:`~.signal.Signal.set_signal_type`
    changes the signal_type in place, what may result in a
    :py:class:`~.signal.Signal`
    subclass transformation.

`signal_origin`
    Describes the origin of the signal and can be "simulation" or
    "experiment" or "",
    the latter meaning undefined. In certain cases HyperSpy provides features
    that are only available for a
    particular signal origin. The :py:class:`~.signal.Signal` method
    :py:meth:`~.signal.Signal.set_signal_origin`
    changes the signal_origin in place, what may result in a
    :py:class:`~.signal.Signal`
    subclass transformation.

.. table:: Signal subclass :py:attr:`~.signal.Signal.metadata` attributes.

    +---------------------------------------------------------------+-----------+-------------+---------------+
    |                       Signal subclass                         | record_by | signal_type | signal_origin |
    +===============================================================+===========+=============+===============+
    |                 :py:class:`~.signal.Signal`                   |     -     |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.spectrum.Spectrum`            | spectrum  |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    | :py:class:`~._signals.spectrum_simulation.SpectrumSimulation` | spectrum  |      -      |  simulation   |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eels.EELSSpectrum`            | spectrum  |    EELS     |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eds_sem.EDSSEMSpectrum`       | spectrum  |   EDS_SEM   |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |           :py:class:`~._signals.eds_tem.EDSTEMSpectrum`       | spectrum  |   EDS_TEM   |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |              :py:class:`~._signals.image.Image`               |   image   |      -      |       -       |
    +---------------------------------------------------------------+-----------+-------------+---------------+
    |    :py:class:`~._signals.image_simulation.ImageSimulation`    |   image   |      -      |  simulation   |
    +---------------------------------------------------------------+-----------+-------------+---------------+


The following example shows how to transform between different subclasses.

   .. code-block:: python

       >>> s = hs.signals.Spectrum(np.random.random((10,20,100)))
       >>> s
       <Spectrum, title: , dimensions: (20, 10|100)>
       >>> s.metadata
       ├── record_by = spectrum
       ├── signal_origin =
       ├── signal_type =
       └── title =
       >>> im = s.to_image()
       >>> im
       <Image, title: , dimensions: (100|20, 10)>
       >>> im.metadata
       ├── record_by = image
       ├── signal_origin =
       ├── signal_type =
       └── title =
       >>> s.set_si
       s.set_signal_origin  s.set_signal_type
       >>> s.set_signal_type("EELS")
       >>> s
       <EELSSpectrum, title: , dimensions: (20, 10|100)>
       >>> s.set_si
       s.set_signal_origin  s.set_signal_type
       >>> s.set_signal_origin("simulation")
       >>> s
       <EELSSpectrumSimulation, title: , dimensions: (20, 10|100)>


The navigation and signal dimensions
------------------------------------

HyperSpy can deal with data of arbitrary dimensions. Each dimension is
internally classified as either "navigation" or "signal" and the way this
classification is done determines the behaviour of the signal.

The concept is probably best understood with an example: let's imagine a three
dimensional dataset. This dataset could be an spectrum image acquired by
scanning over a sample in two dimensions. In HyperSpy's terminology the
spectrum dimension would be the signal dimension and the two other dimensions
would be the navigation dimensions. We could see the same dataset as an image
stack instead.  Actually it could has been acquired by capturing two
dimensional images at different wavelengths. Then it would be natural to
identify the two spatial dimensions as the signal dimensions and the wavelength
dimension as the navigation dimension.  However, for data analysis purposes,
one may like to operate with an image stack as if it was a set of spectra or
viceversa. One can easily switch between these two alternative ways of
classifiying the dimensions of a three-dimensional dataset by
:ref:`transforming between Spectrum and Image subclasses
<transforming.signal>`.

.. NOTE::

    Although each dimension can be arbitrarily classified as "navigation
    dimension" or "signal dimension", for most common tasks there is no need to
    modify HyperSpy's default choice.


.. _signal.binned:

Binned and unbinned signals
---------------------------

.. versionadded:: 0.7

Signals that are a histogram of a probability density function (pdf) should
have the ``signal.metadata.Signal.binned`` attribute set to
``True``. This is because some methods operate differently in signals that are
*binned*.

The default value of the ``binned`` attribute is shown in the
following table:

.. table:: Binned default values for the different subclasses.


    +---------------------------------------------------------------+--------+
    |                       Signal subclass                         | binned |
    +===============================================================+========+
    |                 :py:class:`~.signal.Signal`                   | False  |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.spectrum.Spectrum`            | False  |
    +---------------------------------------------------------------+--------+
    | :py:class:`~._signals.spectrum_simulation.SpectrumSimulation` | False  |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.eels.EELSSpectrum`            | True   |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.eds_sem.EDSSEMSpectrum`       | True   |
    +---------------------------------------------------------------+--------+
    |           :py:class:`~._signals.eds_tem.EDSTEMSpectrum`       | True   |
    +---------------------------------------------------------------+--------+
    |              :py:class:`~._signals.image.Image`               | False  |
    +---------------------------------------------------------------+--------+
    |    :py:class:`~._signals.image_simulation.ImageSimulation`    | False  |
    +---------------------------------------------------------------+--------+





To change the default value:

.. code-block:: python

    >>> s.metadata.Signal.binned = True

Generic tools
-------------

Below we briefly introduce some of the most commonly used tools (methods). For
more details about a particular method click on its name. For a detailed list
of all the methods available see the :py:class:`~.signal.Signal` documentation.

The methods of this section are available to all the signals. In other chapters
methods that are only available in specialized
subclasses.

Simple mathematical operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionchanged:: 0.9

A number of simple operations are supported by :py:class:`~.signal.Signal`. Most
of them are just wrapped numpy functions, as an example:

.. code-block:: python

    >>> s = hs.signals.Signal(np.random.random((2,4,6)))
    >>> s.axes_manager[0].name = 'E'
    >>> s
    <Signal, title: , dimensions: (4, 2|6)>
    >>> # by default perform operation over all navigation axes
    >>> s.sum()
    <Signal, title: , dimensions: (|6)>
    >>> # can also pass axes individually
    >>> s.sum('E')
    <Signal, title: , dimensions: (2|6)>
    >>> # or a tuple of axes to operate on, with duplicates, by index or directly
    >>> ans = s.sum((-1, s.axes_manager[1], 'E', 0))
    >>> ans
    <Signal, title: , dimensions: (|1)>
    >>> ans.axes_manager[0]
    <Scalar axis, size: 1>

Other functions that support similar behavior: :py:func:`~.signal.sum`,
:py:func:`~.signal.max`, :py:func:`~.signal.min`, :py:func:`~.signal.mean`,
:py:func:`~.signal.std`, :py:func:`~.signal.var`. Similar functions that can
only be performed on one axis at a time: :py:func:`~.signal.diff`,
:py:func:`~.signal.derivative`, :py:func:`~.signal.integrate_simpson`,
:py:func:`~.signal.integrate1D`, :py:func:`~.signal.valuemax`,
:py:func:`~.signal.indexmax`.

.. _signal.indexing:

Indexing
^^^^^^^^
.. versionadded:: 0.6

Indexing the :py:class:`~.signal.Signal` provides a powerful, convenient and
Pythonic way to access and modify its data.  It is a concept that might take
some time to grasp but, once mastered, it can greatly simplify many common
signal processing tasks.

Indexing refers to any use of the square brackets ([]) to index the data stored
in a :py:class:`~.signal.Signal`. The result of indexing a
:py:class:`~.signal.Signal` is another :py:class:`~.signal.Signal` that shares
a subset of the data of the original :py:class:`~.signal.Signal`.

HyperSpy's Signal indexing is similar to numpy array indexing and, therefore,
rather that explaining this feature in detail we will just give some examples
of usage here. The interested reader is encouraged to read the `numpy
documentation on the subject  <http://ipython.org/>`_ for a detailed
explanation of the concept. When doing so it is worth to keep in mind the
following main differences:

* HyperSpy (unlike numpy) does not support:

  + Indexing using arrays.
  + Adding new axes using the newaxis object.

* HyperSpy (unlike numpy):

  + Supports indexing with decimal numbers.
  + Uses the image order for indexing i.e. [x, y, z,...] (hyperspy) vs
    [...,z,y,x] (numpy)

Lets start by indexing a single spectrum:


.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (|10)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s[0]
    <Spectrum, title: , dimensions: (|1)>
    >>> s[0].data
    array([0])
    >>> s[9].data
    array([9])
    >>> s[-1].data
    array([9])
    >>> s[:5]
    <Spectrum, title: , dimensions: (|5)>
    >>> s[:5].data
    array([0, 1, 2, 3, 4])
    >>> s[5::-1]
    <Spectrum, title: , dimensions: (|6)>
    >>> s[5::-1]
    <Spectrum, title: , dimensions: (|6)>
    >>> s[5::2]
    <Spectrum, title: , dimensions: (|3)>
    >>> s[5::2].data
    array([5, 7, 9])


Unlike numpy, HyperSpy supports indexing using decimal numbers, in which case
HyperSpy indexes using the axis scales instead of the indices.

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (|10)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> s.axes_manager[0].scale = 0.5
    >>> s.axes_manager[0].axis
    array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])
    >>> s[0.5:4.].data
    array([1, 2, 3, 4, 5, 6, 7])
    >>> s[0.5:4].data
    array([1, 2, 3])
    >>> s[0.5:4:2].data
    array([1, 3])


Importantly the original :py:class:`~.signal.Signal` and its "indexed self"
share their data and, therefore, modifying the value of the data in one
modifies the same value in the other.

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(10))
    >>> s
    <Spectrum, title: , dimensions: (10,)>
    >>> s.data
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> si = s[::2]
    >>> si.data
    array([0, 2, 4, 6, 8])
    >>> si.data[:] = 10
    >>> si.data
    array([10, 10, 10, 10, 10])
    >>> s.data
    array([10,  1, 10,  3, 10,  5, 10,  7, 10,  9])
    >>> s.data[:] = 0
    >>> si.data
    array([0, 0, 0, 0, 0])

Of course it is also possible to use the same syntax to index multidimensional
data.  The first indexes are always the navigation indices in "natural order"
i.e. x,y,z...  and the following indexes are the signal indices also in natural
order.

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Spectrum, title: , dimensions: (10, 10, 10)>
    >>> s.data
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> s.axes_manager[0].name = 'x'
    >>> s.axes_manager[1].name = 'y'
    >>> s.axes_manager[2].name = 't'
    >>> s.axes_manager.signal_axes
    (<t axis, size: 4>,)
    >>> s.axes_manager.navigation_axes
    (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)
    >>> s[0,0].data
    array([0, 1, 2, 3])
    >>> s[0,0].axes_manager
    <Axes manager, axes: (<t axis, size: 4>,)>
    >>> s[0,0,::-1].data
    array([3, 2, 1, 0])
    >>> s[...,0]
    <Spectrum, title: , dimensions: (2, 3)>
    >>> s[...,0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s[...,0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])

For convenience and clarity it is possible to index the signal and navigation
dimensions independently:

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Spectrum, title: , dimensions: (10, 10, 10)>
    >>> s.data
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> s.axes_manager[0].name = 'x'
    >>> s.axes_manager[1].name = 'y'
    >>> s.axes_manager[2].name = 't'
    >>> s.axes_manager.signal_axes
    (<t axis, size: 4>,)
    >>> s.axes_manager.navigation_axes
    (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)
    >>> s.inav[0,0].data
    array([0, 1, 2, 3])
    >>> s.inav[0,0].axes_manager
    <Axes manager, axes: (<t axis, size: 4>,)>
    >>> s.isig[0]
    <Spectrum, title: , dimensions: (2, 3)>
    >>> s.isig[0].axes_manager
    <Axes manager, axes: (<x axis, size: 3, index: 0>, <y axis, size: 2, index: 0>)>
    >>> s.isig[0].data
    array([[ 0,  4,  8],
       [12, 16, 20]])


The same syntax can be used to set the data values:

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(2*3*4).reshape((2,3,4)))
    >>> s
    <Spectrum, title: , dimensions: (10, 10, 10)>
    >>> s.data
    array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> s.inav[0,0].data
    array([0, 1, 2, 3])
    >>> s.inav[0,0] = 1
    >>> s.inav[0,0].data
    array([1, 1, 1, 1])
    >>> s.inav[0,0] = s[1,1]
    >>> s.inav[0,0].data
    array([16, 17, 18, 19])



.. _signal.operations:

Signal operations
^^^^^^^^^^^^^^^^^
.. versionadded:: 0.6

:py:class:`~.signal.Signal` supports all the Python binary arithmetic
opearations (+, -, \*, //, %, divmod(), pow(), \*\*, <<, >>, &, ^, \|),
augmented binary assignments (+=, -=, \*=, /=, //=, %=, \*\*=, <<=, >>=, &=,
^=, \|=), unary operations (-, +, abs() and ~) and rich comparisons operations
(<, <=, ==, x!=y, <>, >, >=).

These operations are performed element-wise. When the dimensions of the signals
are not equal `numpy broadcasting rules apply
<http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ *first*. In
addition HyperSpy extend numpy's broadcasting rules to the following cases:


.. deprecated:: 0.8.1

    This broadcasting rules will change in HyperSpy 0.9.

    .. table:: Signal broadcasting rules (i).

        +------------+----------------------+------------------+
        | **Signal** | **NavigationShape**  | **SignalShape**  |
        +============+======================+==================+
        |   s1       |        a             |      b           |
        +------------+----------------------+------------------+
        |   s2       |       (0,)           |      a           |
        +------------+----------------------+------------------+
        |   s1 + s2  |       a              |      b           |
        +------------+----------------------+------------------+
        |   s2 + s1  |       a              |      b           |
        +------------+----------------------+------------------+


    .. table:: Signal broadcasting rules (ii).

        +------------+----------------------+------------------+
        | **Signal** | **NavigationShape**  | **SignalShape**  |
        +============+======================+==================+
        |   s1       |        a             |      b           |
        +------------+----------------------+------------------+
        |   s2       |       (0,)           |      b           |
        +------------+----------------------+------------------+
        |   s1 + s2  |       a              |      b           |
        +------------+----------------------+------------------+
        |   s2 + s1  |       a              |      b           |
        +------------+----------------------+------------------+


    .. table:: Signal broadcasting rules (iii).

        +------------+----------------------+------------------+
        | **Signal** | **NavigationShape**  | **SignalShape**  |
        +============+======================+==================+
        |   s1       |       (0,)           |      a           |
        +------------+----------------------+------------------+
        |   s2       |       (0,)           |      b           |
        +------------+----------------------+------------------+
        |   s1 + s2  |       b              |      a           |
        +------------+----------------------+------------------+
        |   s2 + s1  |       a              |      b           |
        +------------+----------------------+------------------+


.. _signal.iterator:

Iterating over the navigation axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signal instances are iterables over the navigation axes. For example, the
following code creates a stack of 10 images and saves them in separate "png"
files by iterating over the signal instance:

.. code-block:: python

    >>> image_stack = hs.signals.Image(np.random.random((2, 5, 64,64)))
    >>> for single_image in image_stack:
    ...    single_image.save("image %s.png" % str(image_stack.axes_manager.indices))
    The "image (0, 0).png" file was created.
    The "image (1, 0).png" file was created.
    The "image (2, 0).png" file was created.
    The "image (3, 0).png" file was created.
    The "image (4, 0).png" file was created.
    The "image (0, 1).png" file was created.
    The "image (1, 1).png" file was created.
    The "image (2, 1).png" file was created.
    The "image (3, 1).png" file was created.
    The "image (4, 1).png" file was created.

The data of the signal instance that is returned at each iteration is a view of
the original data, a property that we can use to perform operations on the
data.  For example, the following code rotates the image at each coordinate  by
a given angle and uses the :py:func:`~.utils.stack` function in combination
with `list comprehensions
<http://docs.python.org/2/tutorial/datastructures.html#list-comprehensions>`_
to make a horizontal "collage" of the image stack:

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = hs.signals.Image(np.array([scipy.misc.lena()]*5))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> for image, angle in zip(image_stack, (0, 45, 90, 135, 180)):
    ...    image.data[:] = scipy.ndimage.rotate(image.data, angle=angle,
    ...    reshape=False)
    >>> collage = hs.stack([image for image in image_stack], axis=0)
    >>> collage.plot()

.. figure::  images/rotate_lena.png
  :align:   center
  :width:   500

  Rotation of images by iteration.

.. versionadded:: 0.7


Transforming the data at each coordinate as in the previous example using an
external function can be more easily accomplished using the
:py:meth:`~.signal.Signal.map` method:

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = hs.signals.Image(np.array([scipy.misc.lena()]*4))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> image_stack.map(scipy.ndimage.rotate,
    ...                            angle=45,
    ...                            reshape=False)
    >>> collage = hs.stack([image for image in image_stack], axis=0)
    >>> collage.plot()

.. figure::  images/rotate_lena_apply_simple.png
  :align:   center
  :width:   500

  Rotation of images by the same amount using :py:meth:`~.signal.Signal.map`.

The :py:meth:`~.signal.Signal.map` method can also take variable
arguments as in the following example.

.. code-block:: python

    >>> import scipy.ndimage
    >>> image_stack = hs.signals.Image(np.array([scipy.misc.lena()]*4))
    >>> image_stack.axes_manager[1].name = "x"
    >>> image_stack.axes_manager[2].name = "y"
    >>> angles = hs.signals.Signal(np.array([0, 45, 90, 135]))
    >>> angles.axes_manager.set_signal_dimension(0)
    >>> modes = hs.signals.Signal(np.array(['constant', 'nearest', 'reflect', 'wrap']))
    >>> modes.axes_manager.set_signal_dimension(0)
    >>> image_stack.map(scipy.ndimage.rotate,
    ...                            angle=angles,
    ...                            reshape=False,
    ...                            mode=modes)
    calculating 100% |#############################################| ETA:  00:00:00Cropping

.. figure::  images/rotate_lena_apply_ndkwargs.png
  :align:   center
  :width:   500

  Rotation of images using :py:meth:`~.signal.Signal.map` with different
  arguments for each image in the stack.

Cropping
^^^^^^^^

Cropping can be performed in a very compact and powerful way using
:ref:`signal.indexing` . In addition it can be performed using the following
method or GUIs if cropping :ref:`spectra <spectrum.crop>` or :ref:`images
<image.crop>`. There is also a general :py:meth:`~.signal.Signal.crop`
method that operates *in place*.

Rebinning
^^^^^^^^^

The :py:meth:`~.signal.Signal.rebin` method rebins data in place down to a size
determined by the user.

Folding and unfolding
^^^^^^^^^^^^^^^^^^^^^

When dealing with multidimensional datasets it is sometimes useful to transform
the data into a two dimensional dataset. This can be accomplished using the
following two methods:

* :py:meth:`~.signal.Signal.fold`
* :py:meth:`~.signal.Signal.unfold`

It is also possible to unfold only the navigation or only the signal space:

* :py:meth:`~.signal.Signal.unfold_navigation_space`
* :py:meth:`~.signal.Signal.unfold_signal_space`


.. _signal.stack_split:

Splitting and stacking
^^^^^^^^^^^^^^^^^^^^^^

Several objects can be stacked together over an existing axis or over a
new axis using the :py:func:`~.utils.stack` function, if they share axis
with same dimension.

.. code-block:: python

    >>> image = hs.signals.Image(scipy.misc.lena())
    >>> image = hs.stack([hs.stack([image]*3,axis=0)]*3,axis=1)
    >>> image.plot()

.. figure::  images/stack_lena_3_3.png
  :align:   center
  :width:   500

  Stacking example.

An object can be splitted into several objects
with the :py:meth:`~.signal.Signal.split` method. This function can be used
to reverse the :py:func:`~.utils.stack` function:

.. code-block:: python

    >>> image = image.split()[0].split()[0]
    >>> image.plot()

.. figure::  images/split_lena_3_3.png
  :align:   center
  :width:   400

  Splitting example.


.. _signal.change_dtype:

Changing the data type
^^^^^^^^^^^^^^^^^^^^^^

Even if the original data is recorded with a limited dynamic range, it is often
desirable to perform the analysis operations with a higher precision.
Conversely, if space is limited, storing in a shorter data type can decrease
the file size. The :py:meth:`~.signal.Signal.change_dtype` changes the data
type in place, e.g.:

.. code-block:: python

    >>> s = hs.load('EELS Spectrum Image (high-loss).dm3')
        Title: EELS Spectrum Image (high-loss).dm3
        Signal type: EELS
        Data dimensions: (21, 42, 2048)
        Data representation: spectrum
        Data type: float32
    >>> s.change_dtype('float64')
    >>> print(s)
        Title: EELS Spectrum Image (high-loss).dm3
        Signal type: EELS
        Data dimensions: (21, 42, 2048)
        Data representation: spectrum
        Data type: float64


.. versionadded:: 0.7

    In addition to all standard numpy dtypes HyperSpy supports four extra
    dtypes for RGB images: rgb8, rgba8, rgb16 and rgba16. Changing
    from and to any rgbx dtype is more constrained than most other dtype
    conversions. To change to a rgbx dtype the signal `record_by` must be
    "spectrum", `signal_dimension` must be 3(4) for rgb(rgba) dtypes and the
    dtype must be uint8(uint16) for rgbx8(rgbx16).  After conversion
    `record_by` becomes `image` and the spectra dimension is removed. The dtype
    of images of dtype rgbx8(rgbx16) can only be changed to uint8(uint16) and
    the `record_by` becomes "spectrum".

    In the following example we create

   .. code-block:: python

        >>> rgb_test = np.zeros((1024, 1024, 3))
        >>> ly, lx = rgb_test.shape[:2]
        >>> offset_factor = 0.16
        >>> size_factor = 3
        >>> Y, X = np.ogrid[0:lx, 0:ly]
        >>> rgb_test[:,:,0] = (X - lx / 2 - lx*offset_factor) ** 2 + (Y - ly / 2 - ly*offset_factor) ** 2 < lx * ly / size_factor **2
        >>> rgb_test[:,:,1] = (X - lx / 2 + lx*offset_factor) ** 2 + (Y - ly / 2 - ly*offset_factor) ** 2 < lx * ly / size_factor **2
        >>> rgb_test[:,:,2] = (X - lx / 2) ** 2 + (Y - ly / 2 + ly*offset_factor) ** 2 < lx * ly / size_factor **2
        >>> rgb_test *= 2**16 - 1
        >>> s = hs.signals.Spectrum(rgb_test)
        >>> s.change_dtype("uint16")
        >>> s
        <Spectrum, title: , dimensions: (1024, 1024|3)>
        >>> s.change_dtype("rgb16")
        >>> s
        <Image, title: , dimensions: (|1024, 1024)>
        >>> s.plot()


   .. figure::  images/rgb_example.png
      :align:   center
      :width:   500

      RGB data type example.


Basic statistical analysis
--------------------------
.. versionadded:: 0.7

:py:meth:`~.signal.Signal.get_histogram` computes the histogram and
conveniently returns it as signal instance. It provides methods to
calculate the bins. :py:meth:`~.signal.Signal.print_summary_statistics` prints
the five-number summary statistics of the data.

These two methods can be combined with
:py:meth:`~.signal.Signal.get_current_signal` to compute the histogram or
print the summary stastics of the signal at the current coordinates, e.g:
.. code-block:: python

    >>> s = hs.signals.EELSSpectrum(np.random.normal(size=(10,100)))
    >>> s.print_summary_statistics()
    Summary statistics
    ------------------
    mean:	0.021
    std:	0.957
    min:	-3.991
    Q1:	-0.608
    median:	0.013
    Q3:	0.652
    max:	2.751

    >>> s.get_current_signal().print_summary_statistics()
    Summary statistics
    ------------------
    mean:   -0.019
    std:    0.855
    min:    -2.803
    Q1: -0.451
    median: -0.038
    Q3: 0.484
    max:    1.992

Histogram of different objects can be compared with the functions
:py:func:`~.drawing.utils.plot_histograms` (see
:ref:`visualisation <plot_spectra>` for the plotting options). For example,
with histograms of several random chi-square distributions:


.. code-block:: python

    >>> img = hs.signals.Image([np.random.chisquare(i+1,[100,100]) for i in range(5)])
    >>> hs.plot.plot_histograms(img,legend='auto')

.. figure::  images/plot_histograms_chisquare.png
   :align:   center
   :width:   500

   Comparing histograms.


.. _signal.noise_properties:

Setting the noise properties
----------------------------

Some data operations require the data variance. Those methods use the
``metadata.Signal.Noise_properties.variance`` attribute if it exists. You can
set this attribute as in the following example where we set the variance to be
10:

.. code-block:: python

    s.metadata.Signal.set_item("Noise_properties.variance", 10)

For heterocedastic noise the ``variance`` attribute must be a
:class:`~.signal.Signal`.  Poissonian noise is a common case  of
heterocedastic noise where the variance is equal to the expected value. The
:meth:`~.signal.Signal.estimate_poissonian_noise_variance`
:class:`~.signal.Signal` method can help setting the variance of data with
semi-poissonian noise. With the default arguments, this method simply sets the
variance attribute to the given ``expected_value``. However, more generally
(although then noise is not strictly poissonian), the variance may be proportional
to the expected value. Moreover, when the noise is a mixture of white
(gaussian) and poissonian noise, the variance is described by the following
linear model:

    .. math::

        \mathrm{Var}[X] = (a * \mathrm{E}[X] + b) * c

Where `a` is the ``gain_factor``, `b` is the ``gain_offset`` (the gaussian
noise variance) and `c` the ``correlation_factor``. The correlation
factor accounts for correlation of adjacent signal elements that can
be modeled as a convolution with a gaussian point spread function.
:meth:`~.signal.Signal.estimate_poissonian_noise_variance` can be used to set
the noise properties when the variance can be described by this linear model,
for example:


.. code-block:: python

  >>> s = hs.signals.SpectrumSimulation(np.ones(100))
  >>> s.add_poissonian_noise()
  >>> s.metadata
  ├── General
  │   └── title =
  └── Signal
      ├── binned = False
      ├── record_by = spectrum
      ├── signal_origin = simulation
      └── signal_type =

  >>> s.estimate_poissonian_noise_variance()
  >>> s.metadata
  ├── General
  │   └── title =
  └── Signal
      ├── Noise_properties
      │   ├── Variance_linear_model
      │   │   ├── correlation_factor = 1
      │   │   ├── gain_factor = 1
      │   │   └── gain_offset = 0
      │   └── variance = <SpectrumSimulation, title: Variance of , dimensions: (|100)>
      ├── binned = False
      ├── record_by = spectrum
      ├── signal_origin = simulation
      └── signal_type =


Speeding up operations
----------------------

.. versionadded:: 0.9

Reusing a Signal for output
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many signal methods create and return a new signal. For fast operations, the
new signal creation time is non-negligible. Also, when the operation is
repeated many times, for example in a loop, the cumulaive creation time can
become significant. Therefore, many operations on `Signal` accept an optional
argument `out`. If an existing signal is passed to `out`, the function output
will be placed into that signal, instead of being returned in a new signal.
The following example shows how to use this feature to slice a `Signal`. It is
important to know that the `Signal` instance passed in the `out` argument must
be well-suited for the purpose. Often this means that it must have the same
axes and data shape as the `Signal` that would normally be returned by the
operation.

.. code-block:: python

    >>> s = signals.Spectrum(np.arange(10))
    >>> s_sum = s.sum(0)
    >>> s_sum.data
    array(45)
    >>> s.inav[:5].sum(0, out=s_sum)
    >>> s_sum.data
    10
    >>> s_roi = s.inav[:3]
    >>> s_roi
    <Spectrum, title: , dimensions: (|3)>
    >>> s.inav.__getitem__(slice(None, 5), out=s_roi)
    >>> s_roi
    <Spectrum, title: , dimensions: (|5)>


.. _interactive:

Interactive operations
----------------------

.. versionadded:: 0.9


The function :py:func:`~.interactive.interactive` ease the task of defining
operations that are automatically updated when an event is triggered. By
default it recomputes the operation when data or the axes of the original
signal changes.

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(10.))
    >>> ssum = hs.interactive(s.sum, axis=0)
    >>> ssum.data
    array(45.0)
    >>> s.data /= 10
    >>> s.events.data_changed.trigger()
    >>> ssum.data
    4.5

The interactive opearations can be chained.

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(2 * 3 * 4).reshape((2, 3, 4)))
    >>> ssum = hs.interactive(s.sum, axis=0)
    >>> ssum_mean = hs.interactive(ssum.mean, axis=0)
    >>> ssum_mean.data
    array([ 30.,  33.,  36.,  39.])
    >>> s.data
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> s.data *= 10
    >>> s.events.data_changed.trigger(obj=s)
    >>> ssum_mean.data
    array([ 300.,  330.,  360.,  390.])

Region Of Interest (ROI)
------------------------

.. versionadded:: 0.9

A number of different ROIs are available:

* :py:class:`~.utils.roi.Point1DROI`
* :py:class:`~.utils.roi.Point2DROI`
* :py:class:`~.utils.roi.SpanROI`
* :py:class:`~.utils.roi.RectangularROI`
* :py:class:`~.utils.roi.CircleROI`
* :py:class:`~.utils.roi.Line2DROI`

Once created, a ROI can be used to return a part of any compatible signal:

.. code-block:: python

    >>> s = hs.signals.Spectrum(np.arange(2000).reshape((20,10,10)))
    >>> im = hs.signals.Image(np.arange(100).reshape((10,10)))
    >>> roi = hs.roi.RectangularROI(left=3, right=7, top=2, bottom=5)
    >>> sr = roi(s)
    >>> sr
    <Spectrum, title: , dimensions: (4, 3|10)>
    >>> imr = roi(im)
    >>> imr
    <Image, title: , dimensions: (|4, 3)>

ROIs can also be used :ref:`interactively <Interactive>` with widgets. Notably,
since ROIs are independent from the signals they sub-select, the widget can be
plotted on a different signal altogether.

.. code-block:: python

    >>> import scipy.misc
    >>> im = hs.signals.Image(scipy.misc.ascent())
    >>> s = hs.signals.Spectrum(np.random.rand(512, 512, 512))
    >>> roi = hs.roi.RectangularROI(left=30, right=77, top=20, bottom=50)
    >>> s.plot() # plot signal to have where to display the widget
    >>> imr = roi.interactive(im, navigation_signal=s)

ROIs are implemented in terms of physical coordinates and not pixels, so with
proper calibration will always point to the same region.
