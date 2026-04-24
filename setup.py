"""
Setup script for password-guesser
"""

from setuptools import setup, find_packages

setup(
    name="password-guesser",
    version="1.0.0",
    description="AI-powered targeted password guessing system",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/your-repo/password_guesser",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "password-guesser=password_guesser.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="password security machine-learning mamba differential-evolution",
)
