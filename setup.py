from setuptools import setup, find_packages

setup(
    name = "green-learning", 
    packages=[
        "src", 
        "FF_CNN",
        "utils",
    ],
    package_dir={
        "": ".",
        "src": "./src",
        "FF_CNN": "./src/FF_CNN",
        "utils": "./src/utils",
    },
)