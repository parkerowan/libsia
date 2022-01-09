from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import os
import re


# Get the project version
def get_version() -> str:
    with open("version.txt") as f:
        content = f.readlines()
    vdict = {}
    for x in content:
        pattern = "(\S*) (\d[0-9]*)"
        match = re.search(pattern, x.strip())
        vdict[match.group(1)] = match.group(2)
    ver = "{}.{}.{}".format(vdict["VERSION_MAJOR"], vdict["VERSION_MINOR"],
                            vdict["VERSION_PATCH"])
    return ver


__version__ = get_version()

project_dir = os.path.join(os.getcwd(), os.path.dirname(__file__))

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
ext_modules = [
    Pybind11Extension(
        "pysia",
        [
            "python/py_belief.cpp",
            "python/py_controllers.cpp",
            "python/py_estimators.cpp",
            "python/py_math.cpp",
            "python/py_models.cpp",
            "python/py_optimizers.cpp",
            # "python/py_runner.cpp",
            "python/pysia.cpp",
        ],
        # Example: passing in the version to the compiled code
        define_macros=[('VERSION_INFO', __version__)],
        include_dirs=[
            project_dir,
            "/usr/include/eigen3",
        ],
        libraries=["sia"]),
]

setup(
    name="pysia",
    version=__version__,
    author="Parker Owan",
    author_email="mrpowan10@gmail.com",
    url="https://gitlab.com/parkerowan/librtx",
    description="C++/Python Statistical estimation, fusion and inference",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=True,
)
