# setup.py
from setuptools import setup, find_packages

setup(
    name="scLightGAT",
    version="0.1.0",
    description="An integrated ​bioinformatics tool for single-cell RNA-seq cell type annotation using DVAE, LightGBM, and GAT.",
    author="Tsung-Hsien, Chuang",
    author_email="s4580963@gmail.com",
    url="https://github.com/your-github/scLightGAT",
    packages=find_packages(), 
    install_requires=[
        "anndata",
        "scanpy",
        "torch",
        "torch-geometric",
        "lightgbm",
        "scikit-learn",
        "optuna",
        "seaborn",
        "matplotlib",
        "pandas",
        "numpy",
        "scikit-misc",
        "imbalanced-learn",
        "harmonypy",
        "scrublet",
        "igraph",
        "leidenalg"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
