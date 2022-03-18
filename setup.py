import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LGBMLreturn",
    version="0.0.1",
    author="Arnaud tauveron",
    author_email="https://www.linkedin.com/in/arnaud-tauveron/",
    description="A small example package to predict log-returns on a given table",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
