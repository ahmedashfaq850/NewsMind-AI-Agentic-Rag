from setuptools import setup, find_packages

setup(
    name="news_mind_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
    ],
)
