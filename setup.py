from setuptools import setup, find_packages

setup(
    name='temperature-prediction',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # to fill
    ],
    author='Marco Giacobbe',
    description='',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tuo-username/temperature-prediction-paper',
    python_requires='>=3.7',
)
