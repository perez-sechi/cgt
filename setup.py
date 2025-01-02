from setuptools import setup, find_packages
setup(
    name='cgt_perezsechi',
    version='0.0.8',
    description='Cooperative Game Theory Visualization Tools',
    url = 'https://github.com/perezsechi/cgt',
    download_url='https://github.com/perezsechi/cgt/archive/refs/tags/0.0.8.tar.gz',
    author='Carlos I. Pérez-Sechi',
    author_email='ci.perez@yahoo.es',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'scipy',
        'pandas',
        'multiprocess',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'shap',
        'networkx',
        'nbconvert',
        'seaborn',
        'pingouin',
        'statsmodels',
        'nbconvert',
        'setuptools',
        'pytest'
    ]
)
