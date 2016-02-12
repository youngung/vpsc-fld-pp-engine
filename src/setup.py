"""
Setup tool for VPSC-FLD
"""
from distutils.core import setup
from numpy.distutils.core import setup as setup_numpy

## 
## Forming Limit Diagram Calculation using Visco-Plastic Self-Consistent crystal plasticity model
setup(
    name='VPSC-FLD',
    version='2.1.3',
    description='FLD calculation using VPSC crystal plasticity',
    author='Youngun Jeong',
    author_email='youngung.jeong@gmail.com',
    packages=['vpscfld'],
    package_dir={'vpscfld':'src/py_pack/fld/'})

## Yield-Function and associated parameter calculation using Visco-Plastic Self-Consistent crystal plasticity model
setup(
    name='VPSC-YLD',
    version='0.1',
    description='YLD calculation using VPSC crystal plasticity',
    author='Youngun Jeong',
    author_email='youngung.jeong@gmail.com',
    packages=['vpscyld'],
    package_dir={'vpscyld':'src/py_pack/yld_hah/'})
