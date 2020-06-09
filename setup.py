
"""setup.py: setuptools control."""


import re
from setuptools import setup
from distutils.command.install import INSTALL_SCHEMES

for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('evobridge/evobridge.py').read(),
    re.M
).group(1)


# with open("README.md", "rb") as f:
#long_descr = f.read().decode("utf-8")


setup(
    name="evobridge",
    packages=["evobridge", "evobridge.gui",
              "evobridge.lib", "evobridge.gui.icons"],
    package_data={'': ["*.png"]},
    # data_files=[('', ['evobridge/gui/icons/new-node.png',
    #                  'evobridge/gui/icons/new-rock.png'])],
    install_requires=["numpy", "PyQt5"],
    # include_package_data=True,
    entry_points={
        "console_scripts": ['evobridge = evobridge.evobridge:main']
    },
    version=version,
    description="Bridge truss optimizier using evolutionary algorithms.",
    long_description="",
    author="Michael Schmidt (virtuaCode)",
    author_email="michael.schmidt.dev@gmail.com",
    url="https://github.com/virtuaCode/evobridge",
)
