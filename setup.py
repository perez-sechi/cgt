from setuptools import setup, find_packages
setup(
    name='cgt_perez-sechi',
    version='0.0.7',
    description='Cooperative Game Theory Visualization Tools',
    url = 'https://github.com/perez-sechi/cgt',
    download_url='https://github.com/perez-sechi/cgt/archive/refs/tags/0.0.7.tar.gz',
    author='Carlos I. Pérez-Sechi',
    author_email='secci.jr@gmail.com',
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
