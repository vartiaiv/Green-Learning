from setuptools import setup, find_packages

setup(
    name = "green_learning", 
    packages = [
        # "src", 
        "SAAB_FF",
        "utils",
        ],
    package_dir = {
        "": ".",
        # "src": "./src",
        "SAAB_FF": "./src/SAAB_FF",
        "utils": "./src/utils",
        },
    )