from setuptools import setup, find_packages

setup(
    name="temperature_prediction",
    version="0.1.0",
    description="A temperature prediction project",
    author="Marco Giacobbe",
    packages=find_packages(where="src"),   # Cerca i package dentro src
    package_dir={"": "src"},                 # La root dei package è src
    python_requires=">=3.7",
)