from setuptools import setup

setup(
    name="eVTOLsizing",
    version="0.0.1",
    description="eVTOL UAV sizing code",
    url="https://github.com/kanekosh/eVTOL_sizing",
    author="Shugo Kaneko",
    license="",
    packages=["evtolsizing"],
    package_dir={"" : "src"},
    install_requires=["numpy", "matplotlib", "openmdao>=3.16.0"],
)