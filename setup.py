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

from __future__ import print_function

import sys

v = sys.version_info
if v[0] != 3:
    error = "ERROR: From version 0.8.4 HyperSpy requires Python 3. " \
            "For Python 2.7 install Hyperspy 0.8.3 e.g. " \
            "$ pip install --upgrade hyperspy==0.8.3"
    print(error, file=sys.stderr)
    sys.exit(1)

from setuptools import setup, Extension, Command

import warnings

import os
import subprocess
import fileinput

import re

# stuff to check presence of compiler:
import distutils.sysconfig
import distutils.ccompiler
from distutils.errors import CompileError, DistutilsPlatformError

setup_path = os.path.dirname(__file__)

import hyperspy.Release as Release

install_req = ['scipy',
               'ipython>=2.0',
               'matplotlib>=1.2',
               'numpy>=1.10',
               'traits>=4.5.0',
               'traitsui>=5.0',
               'natsort',
               'requests',
               'tqdm',
               'sympy',
               'dill',
               'h5py',
               'python-dateutil',
               'ipyparallel']

# the hack to deal with setuptools + installing the package in ReadTheDoc:
if 'readthedocs.org' in sys.executable:
    install_req = []


def update_version(version):
    release_path = "hyperspy/Release.py"
    lines = []
    with open(release_path, "r") as f:
        for line in f:
            if line.startswith("version = "):
                line = "version = \"%s\"\n" % version
            lines.append(line)
    with open(release_path, "w") as f:
        f.writelines(lines)


# Extensions. Add your extension here:
raw_extensions = [Extension("hyperspy.tests.misc.cython.test_cython_integration",
                            ['hyperspy/tests/misc/cython/test_cython_integration.pyx']),
                  Extension("hyperspy.io_plugins.unbcf_fast",
                            ['hyperspy/io_plugins/unbcf_fast.pyx']),
                  ]

cleanup_list = []
for leftover in raw_extensions:
    path, ext = os.path.splitext(leftover.sources[0])
    if ext in ('.pyx', '.py'):
        cleanup_list.append(os.path.join(setup_path, path + '.c*'))
        if os.name == 'nt':
            cleanup_list.append(
                os.path.join(
                    setup_path,
                    path +
                    '.cpython-*.pyd'))
        else:
            cleanup_list.append(
                os.path.join(
                    setup_path,
                    path +
                    '.cpython-*.so'))


def count_c_extensions(extensions):
    c_num = 0
    for extension in extensions:
        # if first source file with extension *.c or *.cpp exists
        # it is cythonised or pure c/c++ extension:
        sfile = extension.sources[0]
        path, ext = os.path.splitext(sfile)
        if os.path.exists(path + '.c') or os.path.exists(path + '.cpp'):
            c_num += 1
    return c_num


def cythonize_extensions(extensions):
    try:
        from Cython.Build import cythonize
        return cythonize(extensions)
    except ImportError:
        warnings.warn("""WARNING: cython required to generate fast c code is not found on this system.
Only slow pure python alternative functions will be available.
To use fast implementation of some functions writen in cython either:
a) install cython and re-run the installation,
b) try alternative source distribution containing cythonized C versions of fast code,
c) use binary distribution (i.e. wheels, egg).""")
        return []


def no_cythonize(extensions):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


# to cythonize, or not to cythonize... :
if len(raw_extensions) > count_c_extensions(raw_extensions):
    extensions = cythonize_extensions(raw_extensions)
else:
    extensions = no_cythonize(raw_extensions)


# to compile or not to compile... depends if compiler is present:
compiler = distutils.ccompiler.new_compiler()
assert isinstance(compiler, distutils.ccompiler.CCompiler)
distutils.sysconfig.customize_compiler(compiler)
try:
    compiler.compile([os.path.join(setup_path,
                                   'hyperspy/misc/etc/test_compilers.c')])
except (CompileError, DistutilsPlatformError):
    warnings.warn("""WARNING: C compiler can't be found.
Only slow pure python alternative functions will be available.
To use fast implementation of some functions writen in cython/c either:
a) check that you have compiler (EXACTLY SAME as your python
distribution was compiled with) installed,
b) use binary distribution of hyperspy (i.e. wheels, egg, (only osx and win)).
Installation will continue in 5 sec...""")
    extensions = []
    from time import sleep
    sleep(5)  # wait 5 secs for user to notice the message


# HOOKS ######
post_checout_hook_file = os.path.join(setup_path, '.git/hooks/post-checkout')
git_dir = os.path.join(setup_path, '.git')
hook_ignorer = os.path.join(setup_path, '.hook_ignore')


def find_post_checkout_cleanup_line():
    """find the line index in the git post-checkout hooks
    'rm extension1 extension2 ...'"""
    with open(post_checout_hook_file, 'r') as pchook:
        hook_lines = pchook.readlines()
        for i in range(1, len(hook_lines), 1):
            if re.search('#cleanup_cythonized_and_compiled:',
                         hook_lines[i]) is not None:
                return i + 1

# generate some git hook to clean up and re-build_ext --inplace
# after changing branches:
if os.path.exists(git_dir) and (not os.path.exists(hook_ignorer)):
    recythonize_str = ' '.join([sys.executable,
                                os.path.join(setup_path, 'setup.py'),
                                'clean --all build_ext --inplace \n'])
    if (not os.path.exists(post_checout_hook_file)):
        with open(post_checout_hook_file, 'w') as pchook:
            pchook.write('#!/bin/sh\n')
            pchook.write('#cleanup_cythonized_and_compiled:\n')
            pchook.write('rm ' + ' '.join([i for i in cleanup_list]) + '\n')
            pchook.write(recythonize_str)
        hook_mode = 0o777  # make it executable
        os.chmod(post_checout_hook_file, hook_mode)
    else:
        with open(post_checout_hook_file, 'r') as pchook:
            hook_lines = pchook.readlines()
        if re.search(r'#!/bin/.*?sh', hook_lines[0]) is not None:
            line_n = find_post_checkout_cleanup_line()
            if line_n is not None:
                hook_lines[line_n] = 'rm ' + \
                    ' '.join([i for i in cleanup_list]) + '\n'
                hook_lines[line_n + 1] = recythonize_str
            else:
                hook_lines.append('\n#cleanup_cythonized_and_compiled:')
                hook_lines.append(
                    '\nrm ' + ' '.join([i for i in cleanup_list]) + '\n')
                hook_lines.append(recythonize_str)
            with open(post_checout_hook_file, 'w') as pchook:
                pchook.writelines(hook_lines)


class Recythonize(Command):

    """cythonize all extensions"""
    description = "(re-)cythonize all changed cython extensions"

    user_options = []

    def initialize_options(self):
        """init options"""
        pass

    def finalize_options(self):
        """finalize options"""
        pass

    def run(self):
        # if there is no cython it is supposed to fail:
        from Cython.Build import cythonize
        global raw_extensions
        global extensions
        cythonize(extensions)


class update_version_when_dev:

    def __enter__(self):
        self.release_version = Release.version

        # Get the hash from the git repository if available
        self.restore_version = False
        git_master_path = ".git/refs/heads/master"
        if "+dev" in self.release_version and \
                os.path.isfile(git_master_path):
            p = subprocess.Popen(["git", "describe",
                                  "--tags", "--dirty", "--always"],
                                 stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            if p.returncode != 0:
                # Git is not available, we keep the version as is
                self.restore_version = False
            else:
                gd = stdout[1:].strip().decode()
                # Remove the tag
                gd = gd[gd.index("-") + 1:]
                self.version = self.release_version.replace("+dev", "-git-")
                self.version += gd
                update_version(self.version)
                self.restore_version = True
        else:
            self.version = self.release_version
        return self.version

    def __exit__(self, type, value, traceback):
        if self.restore_version is True:
            update_version(self.release_version)


with update_version_when_dev() as version:
    setup(
        name="hyperspy",
        package_dir={'hyperspy': 'hyperspy'},
        version=version,
        ext_modules=extensions,
        packages=['hyperspy',
                  'hyperspy.datasets',
                  'hyperspy._components',
                  'hyperspy.datasets',
                  'hyperspy.io_plugins',
                  'hyperspy.docstrings',
                  'hyperspy.drawing',
                  'hyperspy.drawing._markers',
                  'hyperspy.drawing._widgets',
                  'hyperspy.learn',
                  'hyperspy._signals',
                  'hyperspy.gui',
                  'hyperspy.utils',
                  'hyperspy.tests',
                  'hyperspy.tests.axes',
                  'hyperspy.tests.component',
                  'hyperspy.tests.datasets',
                  'hyperspy.tests.drawing',
                  'hyperspy.tests.io',
                  'hyperspy.tests.model',
                  'hyperspy.tests.mva',
                  'hyperspy.tests.samfire',
                  'hyperspy.tests.signal',
                  'hyperspy.tests.utils',
                  'hyperspy.tests.misc',
                  'hyperspy.models',
                  'hyperspy.misc',
                  'hyperspy.misc.eels',
                  'hyperspy.misc.eds',
                  'hyperspy.misc.io',
                  'hyperspy.misc.machine_learning',
                  'hyperspy.external',
                  'hyperspy.external.mpfit',
                  'hyperspy.external.astroML',
                  'hyperspy.samfire_utils',
                  'hyperspy.samfire_utils.segmenters',
                  'hyperspy.samfire_utils.weights',
                  'hyperspy.samfire_utils.goodness_of_fit_tests',
                  ],
        install_requires=install_req,
        package_data={
            'hyperspy':
            [
                'misc/eds/example_signals/*.hdf5',
                'tests/io/blockfile_data/*.blo',
                'tests/io/dens_data/*.dens',
                'tests/io/dm_stackbuilder_plugin/test_stackbuilder_imagestack.dm3',
                'tests/io/dm3_1D_data/*.dm3',
                'tests/io/dm3_2D_data/*.dm3',
                'tests/io/dm3_3D_data/*.dm3',
                'tests/io/dm4_1D_data/*.dm4',
                'tests/io/dm4_2D_data/*.dm4',
                'tests/io/dm4_3D_data/*.dm4',
                'tests/io/FEI_new/*.emi',
                'tests/io/FEI_new/*.ser',
                'tests/io/FEI_new/*.npy',
                'tests/io/FEI_old/*.emi',
                'tests/io/FEI_old/*.ser',
                'tests/io/FEI_old/*.npy',
                'tests/io/msa_files/*.msa',
                'tests/io/hdf5_files/*.hdf5',
                'tests/io/tiff_files/*.tif',
                'tests/io/tiff_files/*.dm3',
                'tests/io/npy_files/*.npy',
                'tests/io/unf_files/*.unf',
                'tests/io/bcf_data/*.bcf',
                'tests/io/bcf_data/*.npy',
                'tests/io/ripple_files/*.rpl',
                'tests/io/ripple_files/*.raw',
                'tests/io/emd_files/*.emd',
                'tests/drawing/*.ipynb',
                'tests/signal/test_find_peaks1D_ohaver/test_find_peaks1D_ohaver.hdf5',
            ],
        },
        author=Release.authors['all'][0],
        author_email=Release.authors['all'][1],
        maintainer='Francisco de la Peña',
        maintainer_email='fjd29@cam.ac.uk',
        description=Release.description,
        long_description=open('README.rst').read(),
        license=Release.license,
        platforms=Release.platforms,
        url=Release.url,
        keywords=Release.keywords,
        cmdclass={
            'recythonize': Recythonize,
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Physics",
        ],
    )
