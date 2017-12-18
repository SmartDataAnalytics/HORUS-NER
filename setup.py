#from distutils.core import setup
from setuptools import setup
from os import path
from codecs import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='horusner',
      version='0.1.5',
      long_description=long_description,
      #long_description=open('README.txt').read(),
      license='Apache License 2.0',
      description='HORUS Framework',
      keywords='ner twitter named-entity-recognition noisy entity',
      author='Diego Esteves',
      author_email='diegoesteves@gmail.com',
      url='http://diegoesteves.github.io/horus-models/',
      package_dir={'horusner': 'horus'},
      packages=['horusner','horusner.core',
                'horusner.resources', 'horusner.util'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
    ],
      )

#install_requires=[
#'markdown',
# ],