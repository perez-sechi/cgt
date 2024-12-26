from setuptools import setup
setup(
    name='cgt_seccijr',
    version='0.0.3',
    description='Cooperative Game Theory Visualization Tools',
    url='#',
    author='Carlos I. Pérez-Sechi',
    author_email='secci.jr@gmail.com',
    license='MIT',
    packages=['cgt_seccijr'],
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
