from setuptools import setup, find_packages

setup(
    name="data_perturber",
    version="0.1.0",
    description="Library for perturbing AMR graphs with various error types",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Fabhianto Maoludyo",
    author_email="fabhianto.maoludyo@gmail.com",
    packages=find_packages(),
    install_requires=[
        "penman>=1.2.0",
        "networkx>=2.6.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
