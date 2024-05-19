from setuptools import setup, find_packages

setup(
    name="DatasetPreprocessor",
    version="0.1.0",
    description="A data preprocessing library for machine learning.",
    author="Abhishek Nair, Priyanshi Furiya",
    author_email="abhishek,naiir@gmail.com, furiyapriyanshi@gmail.com",
    url="https://github.com/AbhishekNair050/Dataset-Processor",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
