from setuptools import setup
from setuptools.command.install import install
import subprocess

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.check_call(['python', 'install_cmdstan.py'])

setup(
    name='your_project_name',
    version='0.1',
    packages=['your_project_name'],
    cmdclass={
        'install': CustomInstallCommand,
    },
)