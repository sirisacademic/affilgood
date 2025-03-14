from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="affilgood",  # Replace with your project name
    version="0.1",
    packages=find_packages(),
    install_requires=required,  # Load dependencies from requirements.txt
    description="AffilGood provides annotated datasets and tools to improve the accuracy of attributing scientific works to research organizations, especially in multilingual and complex contexts.",
    author="Nicolau Duran-Silva, Pablo Accuosto, Berta Grimau",
    author_email="nicolau.duransilva@sirisacademic.com, pablo.accuosto@sirisacademic.com, berta.grimau@sirisacademic.com ",
    url="https://github.com/sirisacademic/affilgood",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Adjust Python version as needed
)

