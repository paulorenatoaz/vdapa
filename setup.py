from setuptools import setup, find_packages

setup(
    name='vdapa',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
        'beautifulsoup4',
        'pandas',
        'matplotlib',
        'PyYAML',
    ],
    author='Seu Nome',
    description='Pacote para análise de dados de vulnerabilidades',
)
