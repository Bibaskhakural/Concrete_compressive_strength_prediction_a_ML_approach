from setuptools import setup, find_packages

setup(
    name='concrete_strength_prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
    ],
)
