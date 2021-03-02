#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from setuptools import find_packages, setup


setup(
    name='autobahn',
    version='0.1.0',
    license='MIT',
    description='Simple neural network with activations defined on subgroups.',
    author='Erik Henning Thiede, Wenda Zhou',
    author_email='ehthiede@gmail.com',
    url='https://github.com/risilab/Autobahn',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
)
