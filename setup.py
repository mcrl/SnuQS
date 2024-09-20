import os
import pathlib

from setuptools import setup, Extension, find_packages, find_namespace_packages
from setuptools.command.build_ext import build_ext as build_ext_orig

install_requires = [
    'pybind11==2.12.0',
    'numpy==1.26.4',
    'pyyaml==6.0.1',
    'cmake==3.29.0',
]


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

setup(
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    ext_modules=[
        CMakeExtension('braket.snuqs.csrc'),
    ],
    cmdclass={
        'build_ext': CMakeBuildExt,
    },
)

# setup(
#    name='snuqs-braket',
#    version='1.1.rc1',
#    description='Braket+SnuQS',
#    author='Daeyoung Park',
#    author_email='dypshong@gmail.com',
#    packages=find_namespace_packages(where="src", exclude=("test",)),
#    package_dir={"": "src"},
#    include_package_data=True,
#    install_requires=install_requires,
#    package_data={
#        "": ["*.inc"],
#    },
#
#    ext_modules=[
#        CMakeExtension('braket.snuqs.csrc'),
#    ],
#    cmdclass={
#        'build_ext': CMakeBuildExt,
#    },
#
#    entry_points={
#        "braket.simulators": [
#            "snuqs = braket.snuqs.simulator:StateVectorSimulator",
#        ]
#    },
# )
