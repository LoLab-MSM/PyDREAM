from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pydream',
      version='2.0.0',
      description='A Python implementation of the MT-DREAM(ZS) algorithm from Laloy and Vrugt 2012.',
      long_description=readme(),
      classifiers=['Programming Language :: Python :: 3'],
      url='https://github.com/LoLab-VU/PyDREAM',
      author='Erin Shockley',
      author_email='erin.shockley@vanderbilt.edu',
      packages=['pydream'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True
      )