from setuptools import setup, find_packages

setup(
    packages=find_packages("ldriver"),  # include all packages under src
    package_dir={"": "ldriver"},   # tell distutils packages are under src
    name='ldriver',
    description='Tools for autonomous robot simulation driving',
    package_data={
        "ldriver": ["data/licence/*.png"],
    }
)
