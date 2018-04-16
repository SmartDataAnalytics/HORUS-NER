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
      license='Apache License 2.0',
      description='HORUS Framework',
      keywords='nlp metadata microblog ner twitter named-entity-recognition noisy entity',
      author='Diego Esteves',
      author_email='diegoesteves@gmail.com',
      url='http://diegoesteves.github.io/horus-models/',
      include_package_data=True,
      zip_safe=False,
      package_dir={'horus': 'src/core'},
      packages=['horus.util',
                'horus.translation',
                'horusner.feature_extraction',
                'horusner.feature_extraction.object_detection',
                'horusner.feature_extraction.text_classification'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
    ],
      )

#install_requires=[
#'markdown',
# ],