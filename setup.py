from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pixel-stretching-animation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Create mesmerizing pixel-stretching animations from static images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesondh/pixel-stretching-animation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "scipy>=1.11.0",
        "noise>=1.2.2",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "pixel-stretch=src.cli:main",
        ],
    },
)