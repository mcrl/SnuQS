import os
import pathlib
from glob import glob

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuildExt(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
            super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
                '-DCMAKE_BUILD_TYPE=' + config
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j16'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
            os.chdir(str(cwd))

setup_requires = [
]

install_requires = [
]

dependency_links = [
]

setup(
    name='snuqs',
    version='2.0',
    description='SnuQS',
    author='Daeyoung Park',
    author_email='dypshong@gmail.com',
    packages=find_packages(),
    package_data={
        "snuqs": ["*.inc"],
    },
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    dependency_links=dependency_links,
    ext_modules=[
        CMakeExtension('snuqs'),
    ],
    cmdclass={
        'build_ext': CMakeBuildExt,
        }
    )
