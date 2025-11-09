"""
Setup configuration for DHC-SSM Enhanced Architecture v3.1.0

Deterministic Hierarchical Causal State Space Model
A production-ready AI architecture with deterministic learning approach,
recursive self-improvement, and comprehensive uncertainty quantification.

v3.1.0 Features:
- Fixed type safety (Dict[str, Any])
- Real uncertainty quantification integration
- Adaptive statistical thresholds
- Comprehensive testing with CI/CD
- Production-ready with realistic dependencies
- O(n) linear complexity maintained
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8")

# Core requirements (must match requirements.txt)
requirements = [
    "torch>=2.0.0,<3.0.0",
    "torchvision>=0.15.0,<1.0.0",
    "torch-geometric>=2.3.0",
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "pandas>=1.4.0",
    "pillow>=9.0.0",
    "typing-extensions>=4.0.0",
    "psutil>=5.8.0",
]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.6.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "isort>=5.10.0",
    "mypy>=0.991",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
    "tensorboard>=2.10.0",
    "pre-commit>=2.15.0",
]

setup(
    name="dhc-ssm-agi",
    version="3.1.0",
    author="Sung hun kwag",
    author_email="sunghunkwag@gmail.com",
    description="DHC-SSM v3.1 - Deterministic O(n) AGI Architecture with Recursive Self-Improvement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunghunkwag/DHC-SSM-AGI",
    project_urls={
        "Bug Reports": "https://github.com/sunghunkwag/DHC-SSM-AGI/issues",
        "Source": "https://github.com/sunghunkwag/DHC-SSM-AGI",
        "Documentation": "https://github.com/sunghunkwag/DHC-SSM-AGI/blob/master/README.md",
        "Changelog": "https://github.com/sunghunkwag/DHC-SSM-AGI/blob/master/CHANGELOG.md",
        "CI/CD": "https://github.com/sunghunkwag/DHC-SSM-AGI/actions",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Typing :: Typed",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    keywords=[
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "AGI",
        "state space models",
        "causal reasoning",
        "deterministic learning",
        "graph neural networks",
        "uncertainty quantification",
        "recursive self-improvement",
        "meta-learning",
        "O(n) complexity",
        "transformer alternative",
        "production ready",
        "pytorch",
    ],
    include_package_data=True,
    zip_safe=False,
)
