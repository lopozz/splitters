from setuptools import setup

from splitters import __version__

INSTALL_REQUIRES = [
    "transformers",
    "spacy",
    "tiktoken",
]

EXTRA_TEST = [
    "pytest>=8",
    "pytest-cov>=5",
]

EXTRA_DEV = [
    *EXTRA_TEST,
]

EXTRAS_REQUIRE = {
    "dev": EXTRA_DEV,
}

setup(
    name="splitters",
    version=__version__,
    author="Lorenzo pozzi",
    author_email="lorenzopozzi17@yahoo.it",
    description="Collect text segmentation functionalities.",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
