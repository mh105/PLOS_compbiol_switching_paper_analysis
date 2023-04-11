import setuptools

setuptools.setup(
    name='pykalman',
    packages=['pykalman'],
    version='0.1',
    author='Mingjian He',
    author_email='mh105@mit.edu',
    description='Python Kalman filtering and smoothing',
    url='https://github.com/mh105/djkalman/tree/master/python',
    install_requires=['numpy', 'torch']
)
