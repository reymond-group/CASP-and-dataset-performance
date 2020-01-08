from setuptools import setup, find_packages

setup(name='aizynthfinder',
      version='0.1',
      description="""Retrosynthetic route finding using neural network guided Monte-Carlo tree search.""",
      author='Esben Jannik Bjerrum',
      author_email='esben.bjerrum@astrazeneca.com',
      license='proprietary',
      packages=find_packages(),
      install_requires=[
          'keras',
          'numpy',
          #'rdkit', RDKit installed via special channel
          'ipywidgets'
      ],
      zip_safe=False)