from setuptools import setup, find_packages
setup(
    name='cgt_seccijr',
    version='0.0.4',
    description='Cooperative Game Theory Visualization Tools',
    url = 'https://github.com/seccijr/cgt',
    download_url='https://github.com/seccijr/cgt/archive/refs/tags/0.0.4.tar.gz',
    author='Carlos I. Pérez-Sechi',
    author_email='secci.jr@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'scipy',
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
