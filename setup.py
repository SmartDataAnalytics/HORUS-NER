from distutils.core import setup
setup(name='horus',
      version='0.1',
      description='HORUS basic metadata',
      author='Diego Esteves',
      author_email='diegoesteves@gmail.com',
      packages=['src','components','resource'],
      package_dir={'src': 'src/',
                   'components': 'src/components/',
                   'resource': 'src/resource/'},
      )
