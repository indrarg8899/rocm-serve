# rocm-serve — LLM inference server for AMD ROCm/MI300X
#
# Install with: pip install -e .
# Run with:     rocm-serve --model meta-llama/Llama-3-70B-Instruct

from setuptools import setup, find_packages

setup(
    name="rocm-serve",
    version="0.1.0",
    description="Production-ready LLM serving framework for AMD ROCm/MI300X",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="indrarg8899",
    author_email="",
    url="https://github.com/indrarg8899/rocm-serve",
    project_urls={
        "Documentation": "https://github.com/indrarg8899/rocm-serve/tree/main/docs",
        "Bug Tracker": "https://github.com/indrarg8899/rocm-serve/issues",
        "Source Code": "https://github.com/indrarg8899/rocm-serve",
    },
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.4.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "rocm-serve=server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["rocm", "llm", "serving", "amd", "mi300x", "inference", "openai-compatible"],
)
