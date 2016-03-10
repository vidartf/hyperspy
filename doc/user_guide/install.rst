Installing HyperSpy
===================

The easiest way to install HyperSpy in Microsoft Windows is installing the
:ref:`HyperSpy Bundle <hyperspy-bundle>`.

For quick instructions on how to install HyperSpy in Linux, MacOs or Windows
using the `Anaconda Python distribution <http://docs.continuum.io/anaconda/>`_
see  :ref:`quick-anaconda-install`.

Those experienced with Python may like to
:ref:`install-with-python-installers` or :ref:`install-source`.

.. warning::

    Since version 0.8.4 HyperSpy only supports Python 3. If you need to install
    HyperSpy in Python 2.7 install HyperSpy 0.8.3.

.. _hyperspy-bundle:

HyperSpy Bundle for Microsoft Windows
-------------------------------------

.. versionadded:: 0.6

The easiest way to install HyperSpy in Windows is installing the HyperSpy
Bundle. This is a customised `WinPython <http://winpython.github.io/>`_
distribution that includes HyperSpy, all its dependencies and many other
scientific Python packages. HyperSpy Bundle does not interact with any other
Python installation in your system, so it can be safely installed alongside
other Python distributions. Moreover it is portable, so it can be installed in
an USB key. When intalling it with administator rights for all users it adds
context menu entries to start the `Jupyter Notebook <http://jupyter.org>`_ and
`Juypter QtConsole <http://jupyter.org/qtconsole/stable/>`_. See
`start_jupyter_cm <https://github.com/hyperspy/start_jupyter_cm>`_ for details.


.. _quick-anaconda-install:

Quick instructions to install HyperSpy using Anaconda (Linux, MacOs, Windows)
-----------------------------------------------------------------------------

#. Download and install
   `Anaconda. <https://store.continuum.io/cshop/anaconda/>`_
   Anaconda is recommended for the best performance (it is compiled
   using Intel MKL libraries) and the easiest intallation (all the required
   libraries are included). The academic license is free.

   .. code-block:: bash

       $ pip install hyperspy

.. warning::
    Since version 0.8.4 HyperSpy only supports Python 3. If you need to
    install HyperSpy in Python 2.7 install version 0.8.3:

    .. code-block:: bash

        $ pip install --upgrade hyperspy==0.8.3-1

For convenience you may also consider installing `start_jupyter_cm
<https://github.com/hyperspy/start_jupyter_cm>`_.


For more options and details read the rest of the documentation.


.. _install-with-python-installers:

Install using Python installers
-------------------------------

HyperSpy is listed in the `Python Package Index
<http://pypi.python.org/pypi>`_. Therefore, it can be automatically downloaded
and installed  `pip <http://pypi.python.org/pypi/pip>`_. You may need to install
pip for the following commands to run.

Install using `pip`:

.. code-block:: bash

    $ pip install hyperspy

.. warning::
    Since version 0.8.4 HyperSpy only supports Python 3. If you need to
    install HyperSpy in Python 2.7 install version 0.8.3:

    .. code-block:: bash

        $ pip install --upgrade hyperspy==0.8.3-1


pip installs automatically the stricly required libraries. However, for full
functionality you may need to install some other dependencies,
see :ref:`install-dependencies`.

Creating Conda environment for HyperSpy
---------------------------------------

`Anaconda <https://www.continuum.io/downloads>`_ Python distribution can be
easily set up using environment files. The two required steps are:
 1. Download `HyperSpy environment file <https://raw.githubusercontent.com/hyperspy/hyperspy/0.8.x/anaconda_hyperspy_environment.yml>`_.
 2. Create and activate HyperSpy environment according to instructions `here <http://conda.pydata.org/docs/using/envs.html#use-environment-from-file>`_. For Unix, the following should work:

.. code-block:: bash

    $ conda env create -f anaconda_hyperspy_environment.yml
    $ source activate hyperspy



.. _install-binary:

Install from a binary
---------------------

We provide  binary distributions for Windows (`see the
Downloads section of the website <http://hyperspy.org/download.html>`_). To
install easily in other platforms see :ref:`install-with-python-installers`


.. _install-source:

Install from source
-------------------

.. _install-released-source:

Released version
^^^^^^^^^^^^^^^^

To install from source grab a tar.gz release and in Linux/Mac (requires to
:ref:`install-dependencies` manually):

.. code-block:: bash

    $ tar -xzf hyperspy.tar.gz
    $ cd hyperspy
    $ python setup.py install

You can also use a Python installer, e.g.

.. code-block:: bash

    $ pip install hyperspy.tar.gz

.. _install-dev:

Development version
^^^^^^^^^^^^^^^^^^^


To get the development version from our git repository you need to install `git
<http://git-scm.com//>`_. Then just do:

.. code-block:: bash

    $ git clone https://github.com/hyperspy/hyperspy.git

To install HyperSpy you could proceed like in :ref:`install-released-source`.
However, if you are installing from the development version most likely you
will prefer to install HyperSpy using  `pip <http://www.pip-installer.org>`_
development mode:


.. code-block:: bash

    $ cd hyperspy
    $ pip install -e ./

All required dependencies are automatically installed by pip. However, for extra
functonality you may need to install some extra dependencies, see
:ref:`install-dependencies`. Note the pip installer requires root to install,
so for Ubuntu:

.. code-block:: bash

    $ cd hyperspy
    $ sudo pip install -e ./

If using Arch Linux, the latest checkout of the master development branch can be
installed through the AUR by installing the `hyperspy-git package
<https://aur.archlinux.org/packages/hyperspy-git/>`_

.. _create-debian-binary:

Creating Debian/Ubuntu binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create binaries for Debian/Ubuntu from the source by running the
`release_debian` script

.. code-block:: bash

    $ ./release_debian

.. Warning::

    For this to work, the following packages must be installed in your system
    python-stdeb, debhelper, dpkg-dev and python-argparser are required.


.. _install-dependencies:

Installing the required libraries
---------------------------------


When installing HyperSpy using Python installers or from source the Python
programming language and the following libraries must be installed in the
system: numpy, scipy, matplotlib (>= 1.2), ipython, natsort, traits and
traitsui. For full functionality it is recommended to also install h5py and
scikit-learn. In addition, since version 0.7.2 the lowess filter requires
statsmodels. In Windows HyperSpy uses the Ipython's QtConsole and therefore Qt
and PyQt or PySide are also required.


In Debian/Ubuntu you can install the libraries as follows:

.. code-block:: bash

    $ sudo apt-get install python-numpy python-matplotlib ipython
    ipython-notebook python-traits python-traitsui python-h5py
    python-scikits-learn python-nose python-statsmodels

In Arch Linux, the following command should install the required packages to
get a fully functional installation:

.. code-block:: bash

    $ sudo pacman -Sy python2 python2-numpy	python2-matplotlib	python2-pip
    python2-traits python2-traitsui python2-h5py python2-scikit-learn python2-nose
    python2-statsmodels python2-pillow python2-pyqt4 python2-pyqt5 python2-scipy
    python2-pandas python2-setuptools ipython2	python2-jinja python2-pyzmq
    python2-pyqt4 python2-tornado python2-sip python2-pygments

    # Or, just run this command from the root hyperspy directory to import the
    # list of packages and install automatically:
    $ xargs sudo pacman -Sy --noconfirm < doc/package_lists/arch_linux_package_list.txt

    # Once these are installed, go to the HyperSpy directory and run:
    $ sudo pip2 install -e ./

    # If desired, the python2-seaborn library can also be installed from AUR for prettier plotting

.. _known-issues:

Known issues
------------

Windows
^^^^^^^

* If HyperSpy fails to start in Windows try installing the Microsoft Visual
  C++ 2008 redistributable packages (
  `64 bit <http://www.microsoft.com/download/en/details.aspx?id=15336>`_
  or `32 bit <http://www.microsoft.com/download/en/details.aspx?id=29>`_)
  before reporting a bug.
* In some Windows machines an error is printed at the end of the installation
  and the entries in the context menu and the Start Menu are not installed
  properly. In most cases the problem can be solved by restarting the computer
  and reinstalling HyperSpy.
* Due to a `Python bug <http://bugs.python.org/issue13276>`_ sometimes uninstalling
  HyperSpy does not uninstall the "HyperSpy Here" entries in the context menu.
  Please run the following code in a Windows Terminal with administrator rights
  to remove the entries manually:

  .. code-block:: bash

    $ uninstall_hyperspy_here
* If HyperSpy raise a MemoryError exceptions:

  * Install the 64bit if you're using the 32bit one and you are running
    HyperSpy in a 64bit system.
  * Increase the available RAM but closing other applications or physically
    adding more RAM to your computer.
