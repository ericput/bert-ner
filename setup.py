#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='njuner',
    version='0.1.7',
    description=(
        'A NER tool from Nanjing University NLP Group.'
    ),
    url = 'https://github.com/ericput/bert-ner',
    author='Putong',
    author_email='put@nlp.nju.edu.cn',
    license='MIT License',
    packages=find_packages(),
    platforms=["Linux"],
    install_requires = ["torch>=0.4.1"],
    scripts=["bin/njuner"],
    python_requires='>=3.5.0'
)
