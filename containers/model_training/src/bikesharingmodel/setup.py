from setuptools import find_packages, setup

NAME = 'trainer'
VERSION = '0.1'
AUTHOR = 'MA MHANNA'
EMAIL = 'maggie.mhanna@renault.com'

REQUIRED_PACKAGES = ['tensorflow==2.2.0', 
                     'tensorflow-cpu==2.2.0',
                     'tensorflow_transform==0.22.0',
                     'tensorflow_model_analysis==0.22.0', 
                     'apache_beam[gcp]==2.20.0', 
                     'pyarrow==0.16.0',
                     'tfx-bsl==0.22.0',
                     'absl-py==0.8.1']


setup(
    name=NAME,
    version=VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Bike Sharing Demand Prediction.',
    requires=[]
)
