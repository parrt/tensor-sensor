from setuptools import setup

setup(
    name='tensor-sensor',
    version='0.1a8',
    url='https://github.com/parrt/tensor-sensor',
    license='MIT',
    py_modules=['tsensor.parsing', 'tsensor.ast', 'tsensor.analysis', 'tsensor.viz'],
    author='Terence Parr',
    author_email='parrt@cs.usfca.edu',
    python_requires='>=3.6',
    install_requires=['graphviz','numpy','torch','tensorflow', 'IPython', 'matplotlib'],
    description='The goal of this library is to generate more helpful exception messages for numpy/pytorch tensor algebra expressions.',
#    keywords='visualization data structures',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
