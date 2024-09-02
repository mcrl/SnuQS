
from setuptools import find_namespace_packages, setup
setup(
    name="braket-snuqs-simulator",
    version="0.0.1",
    package_dir={"": "src"},
    entry_points={
        "braket.simulators": [
            "snuqs = braket.snuqs.snuqs_simulator:SnuQSStateVectorSimulator",
        ]
    },
)
