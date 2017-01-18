from distutils.core import setup
setup(name='definitions',
      version='0.1',
      description='HORUS basic metadata',
      author='Diego Esteves',
      author_email='diegoesteves@gmail.com',
      py_modules=['definitions'],
      packages=['horus', 'experiments'],
      package_dir={'horus': 'src/horus', 'experiments': 'src/experiments'},
      )
