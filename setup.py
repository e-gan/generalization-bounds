#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('readme.md') as readme_file:
    readme = readme_file.read()

test_requirements = [
    "pytest",
    "pytest-timeout",
]

setup(
    name='generalization_bounds',
    long_description=readme,
    author="Jack King",
    author_email='jackking@mit.edu',
    url='https://github.com/e-gan/generalization-bounds.git',
    packages=find_packages(exclude=['tests']),
    license="MIT license",
    zip_safe=False,
    test_suite='tests',
    tests_require=test_requirements
)
