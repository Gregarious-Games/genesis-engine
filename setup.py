"""
Genesis Neural Network Library - Setup
"""

from setuptools import setup, find_packages

setup(
    name="genesis_nn",
    version="0.1.0",
    author="Genesis Project (Greg Starkins, Claude)",
    author_email="genesis@example.com",
    description="Oscillatory neural networks with golden ratio dynamics",
    long_description=open("README.md").read() if __file__ else "",
    long_description_content_type="text/markdown",
    url="https://github.com/genesis-project/genesis-nn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
