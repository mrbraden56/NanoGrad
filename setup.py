import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='nanograd',
      version='0.0.0',
      author='Braden Lockwood',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['slimgrad'],
      install_requires=['numpy'],
      python_requires='>=3.8',
      include_package_data=True)