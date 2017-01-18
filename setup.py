from distutils.core import setup

setup(name='horusner',
      version='0.2',
      description='HORUS Framework',
      author='Diego Esteves',
      author_email='diegoesteves@gmail.com',
      url='http://dnes85.github.io/horus-models/',
      package_dir={'horusner': 'src/horus'},
      packages=['horusner', 'horusner.components', 'horusner.experiments', 'horusner.postagger',
                'horusner.resource', 'horusner.sift'],
      )
