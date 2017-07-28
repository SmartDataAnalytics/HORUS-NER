#from distutils.core import setup
from setuptools import setup


setup(name='horusner',
      version='0.1.5',
      long_description=open('README.txt').read(),
      description='HORUS Framework',
      author='Diego Esteves',
      author_email='diegoesteves@gmail.com',
      url='http://diegoesteves.github.io/horus-models/',
      install_requires=[
          'markdown',
      ],
      package_dir={'horusner': 'src/horus'},
      packages=['horusner','horusner.components',
                'horusner.experiments', 'horusner.postagger',
                'horusner.resource'],
      )
