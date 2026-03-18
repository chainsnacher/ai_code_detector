"""
Setup script for AI Code Detection System
"""

import os
from setuptools import setup, find_packages

# ✅ 1. Get the absolute path to the directory containing setup.py
base_dir = os.path.abspath(os.path.dirname(__file__))

# ✅ 2. Safely read the README file
long_description = "Advanced AI Code Detection System"
readme_path = os.path.join(base_dir, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

# ✅ 3. Safely read requirements (CRASH-PROOF)
requirements = []
requirements_path = os.path.join(base_dir, "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]
else:
    # This prevents the FileNotFoundError that is stopping your build
    print("WARNING: requirements.txt not found at build time. Using empty list.")

setup(
    name="ai-code-detector",
    version="1.0.0",
    author="AI Code Detection Team",
    author_email="team@ai-code-detector.com",
    description="Advanced AI Code Detection System with Multi-Modal Feature Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chainsnacher/ai_code_detector",
    packages=find_packages(),

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Education",
    ],

    python_requires=">=3.8",
    install_requires=requirements,

    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "gpu": [
            "torch==2.0.1+cpu",
            "torchvision>=0.10.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "ai-code-detector=main:main",
            "ai-code-detector-train=main:main",
            "ai-code-detector-web=web_app.app:main",
        ],
    },

    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },

    keywords=[
        "ai", "code", "detection", "machine-learning", "nlp",
        "transformer", "codebert", "ensemble", "adversarial",
        "robustness", "explainable-ai"
    ],

    project_urls={
        "Bug Reports": "https://github.com/chainsnacher/ai_code_detector/issues",
        "Source": "https://github.com/chainsnacher/ai_code_detector",
        "Documentation": "https://ai-code-detector.readthedocs.io/",
        "Homepage": "https://ai-code-detector.com",
    },
)
