from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import pytest
import sys


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['-v']

    def run_tests(self):
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="my_project",
    version="0.1",
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    entry_points={
        'console_scripts': [
            'generate_readme=main.documentation.readme_generator:generate',
        ],
    },
)
