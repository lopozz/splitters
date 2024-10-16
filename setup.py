from setuptools import setup, find_packages

from splitters import __version__

INSTALL_REQUIRES = [
    "transformers",
    "spacy",
    "tiktoken",
]

EXTRA_TEST = [
    'pytest>=8',
    'pytest-cov>=5',
]

EXTRA_DEV = [
    *EXTRA_TEST,
]

EXTRAS_REQUIRE={
    'dev': EXTRA_DEV,
}

setup(
    name='splitters',
    version='0.1.0',
    author='Lorenzo pozzi',
    author_email='lorenzopozzi17@yahoo.it',
    description='Collect text segmentation functionalities.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)