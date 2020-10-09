"""
MIT License

Copyright (c) 2020 Terence Parr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from setuptools import setup

tensorflow_requires = ['tensorflow']
torch_requires = ['torch']
all_requires = tensorflow_requires + torch_requires

exec(open('tsensor/version.py').read())
setup(
    name='tensor-sensor',
    version=__version__,
    url='https://github.com/parrt/tensor-sensor',
    license='MIT',
    py_modules=['tsensor.parsing', 'tsensor.ast', 'tsensor.analysis', 'tsensor.viz', 'tsensor.version'],
    author='Terence Parr',
    author_email='parrt@cs.usfca.edu',
    python_requires='>=3.6',
    install_requires=['graphviz>=0.14.1','numpy','IPython', 'matplotlib'],
    extras_require = {'all': all_requires,
                      'torch': torch_requires,
                      'tensorflow': tensorflow_requires
                     },
    description='The goal of this library is to generate more helpful exception messages for numpy/pytorch tensor algebra expressions.',
#    keywords='visualization data structures',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
