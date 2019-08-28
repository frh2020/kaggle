
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [ 'gcsfs'
]

setup(
    name='kaggle-digit-cnn',
    version='1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='kaggle digit recognizer project',
    requires=[]
)
