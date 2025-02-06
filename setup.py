from setuptools import setup, find_packages

setup(
    name="pyptx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    author="Daron Popov",
    author_email="daron94545@gmail.com",
    description="A Python framework for PTX code generation and GPU computation",
    long_description=open("README.md", encoding="utf-8").read(),
    url="https://github.com/DaronPopov/pyptx",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Swag License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
