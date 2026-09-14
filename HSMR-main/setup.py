from setuptools import setup, find_packages


# import `__version__`
with open('lib/version.py') as f:
    exec(f.read())

setup(
    name         = 'lib',
    version      = __version__,  # type: ignore
    author       = 'Yan Xia',
    author_email = 'isshikihugh@gmail.com',
    description  = 'Official implementation of HSMR.',
    packages     = find_packages(),
)