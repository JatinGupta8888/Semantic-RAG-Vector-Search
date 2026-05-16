from setuptools import setup, find_packages

setup(
    name="semantic-rag-engine",
    version="1.0.0",
    description="Context-Aware Retrieval Engine with Strategy A/B Benchmarking",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "numpy>=1.26.0",
        "tabulate>=0.9.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
        ]
    },
)
