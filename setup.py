from setuptools import setup

setup(
    name='matricks',
    version='0.1',
    url='https://github.com/parrt/matricks',
    license='MIT',
    py_modules=['matricks'],
    author='Terence Parr',
    author_email='parrt@cs.usfca.edu',
    python_requires='>=3.6',
    install_requires=['graphviz','numpy','pytorch'],
    description='The goal of this library is to generate more helpful exception messages for numpy/pytorch matrix algebra expressions.',
#    keywords='visualization data structures',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
