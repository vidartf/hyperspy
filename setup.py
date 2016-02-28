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


from setuptools import setup

import distutils.dir_util

import os
import subprocess
import sys
import fileinput

import hyperspy.Release as Release
# clean the build directory so we aren't mixing Windows and Linux
# installations carelessly.
if os.path.exists('build'):
    distutils.dir_util.remove_tree('build')

install_req = ['scipy',
               'ipython (>= 2.0)',
               'matplotlib (>= 1.2)',
               'numpy',
               'traits',
               'traitsui',
               'sympy',
               'setuptools',
               ]


def are_we_building4windows():
    for arg in sys.argv:
        if 'wininst' in arg:
            return True

scripts = ['bin/hyperspy', ]


class update_version_when_dev:

    def __enter__(self):
        self.release_version = Release.version

        # Get the hash from the git repository if available
        self.restore_version = False
        git_master_path = ".git/refs/heads/master"
        if "+dev" in self.release_version and \
                os.path.isfile(git_master_path):
            try:
                p = subprocess.Popen(["git", "describe",
                                      "--tags", "--dirty", "--always"],
                                     stdout=subprocess.PIPE)
                stdout = p.communicate()[0]
                if p.returncode != 0:
                    raise EnvironmentError
                else:
                    version = stdout[1:].strip()
                    if str(self.release_version[:-4] + '-') in version:
                        version = version.replace(
                            self.release_version[:-4] + '-',
                            self.release_version[:-4] + '+git')
                    self.version = version
            except EnvironmentError:
                # Git is not available, but the .git directory exists
                # Therefore we can get just the master hash
                with open(git_master_path) as f:
                    masterhash = f.readline()
                self.version = self.release_version.replace(
                    "+dev", "+git-%s" % masterhash[:7])
            for line in fileinput.FileInput("hyperspy/Release.py",
                                            inplace=1):
                if line.startswith('version = '):
                    print "version = \"%s\"" % self.version
                else:
                    print line,
            self.restore_version = True
        else:
            self.version = self.release_version
        return self.version

    def __exit__(self, type, value, traceback):
        if self.restore_version is True:
            for line in fileinput.FileInput("hyperspy/Release.py",
                                            inplace=1):
                if line.startswith('version = '):
                    print "version = \"%s\"" % self.release_version
                else:
                    print line,


with update_version_when_dev() as version:
    setup(
        name="hyperspy",
        package_dir={'hyperspy': 'hyperspy'},
        version=version,
        packages=['hyperspy',
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
                  'hyperspy.tests.drawing',
                  'hyperspy.tests.io',
                  'hyperspy.tests.model',
                  'hyperspy.tests.mva',
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
                  ],
        requires=install_req,
        setup_requires=[
            'setuptools'
        ],
        scripts=scripts,
        package_data={
            'hyperspy':
            ['ipython_profile/*',
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
             'tests/io/msa_files/*.msa',
             'tests/io/hdf5_files/*.hdf5',
             'tests/io/tiff_files/*.tif',
             'tests/io/npy_files/*.npy',
             'tests/io/unf_files/*.unf',
             'tests/drawing/*.ipynb',
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
        classifiers=[
            "Programming Language :: Python :: 2.7",
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
