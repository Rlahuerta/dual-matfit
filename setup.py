from setuptools import setup, find_packages

# FIXME: use just the pyproject.toml to install the project locally
# This setup.py is maintained for backward compatibility but pyproject.toml is the source of truth
setup(
    name="dualmatfit",
    description="Dual Material Fitting",
    long_description=open('README.md').read(),
    packages=find_packages(where='dualmatfit'),
    author="rilah",
    maintainer='Jarvis',
    platforms=['any'],
    keywords=['numpy', 'sympy', 'scipy', 'ipopt'],
    python_requires='>=3.11,<3.14',
    install_requires=[],
    zip_safe=False,
)