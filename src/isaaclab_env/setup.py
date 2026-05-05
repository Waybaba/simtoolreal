from setuptools import find_packages, setup


setup(
    name="isaaclab_env",
    version="0.1.0",
    description="Isaac Lab task package for SimToolReal migration experiments",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
