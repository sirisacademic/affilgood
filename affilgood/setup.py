from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []

setup(
    name="affilgood",
    version="2.0.0",
    author="Nicolau Duran-Silva, Pablo Accuosto",
    author_email="nicolau.duransilva@sirisacademic.com, pablo.accuosto@sirisacademic.com",
    description="AffilGood provides tools for institution name disambiguation in scientific literature.",
    url="https://github.com/sirisacademic/affilgood",
    packages=find_packages(),  # Automatically find packages within affilgood/
    install_requires=requirements,  # Load dependencies from requirements.txt
    python_requires=">=3.9",
    include_package_data=True,  # Include additional files specified in MANIFEST.in
)