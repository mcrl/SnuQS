from setuptools import setup
from cmake_build_extension import CMakeExtension, BuildExtension

setup(
    ext_modules=[
        CMakeExtension('snuqs')
    ],
    cmdclass={
        'build_ext': BuildExtension,
        }
    )
