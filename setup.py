from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="constriction",
    version="0.1.3",
    rust_extensions=[RustExtension(
        "constriction.constriction", binding=Binding.PyO3, features=['pybindings'])],
    packages=["constriction"],
    author='Robert Bamler',
    author_email='robert.bamler@uni-tuebingen.de',
    url='https://bamler-lab.github.io/constriction/',
    description='Composable entropy coding primitives for research and production (Python and Rust).',
    keywords=['compression', 'machine-learning',
              'entropy-coding', 'range-coding', 'ANS'],
    license='MIT OR Apache-2.0 OR BSL-1.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3',
    install_requires=[
        'msgpack>=1.0.0',
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
