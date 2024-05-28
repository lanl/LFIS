import setuptools

with open("README.md", "r", encoding="utf-8") as f:
   long_description = f.read()


__version__ = "1.0.0"

REPO_NAME = "LFIS"
AUTHOR_USER_NAME = "lanl"
SRC_REPO = "LFIS"
AUTHOR_EMAIL = "yifengtian@lanl.gov"


setuptools.setup(
   name=SRC_REPO,
   version=__version__,
   author=AUTHOR_USER_NAME,
   author_email=AUTHOR_EMAIL,
   description="Source code for manuscript Liouville Flow Importance Sampler",
   long_description=long_description,
   long_description_content="text/markdown",
   url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
   project_urls={
       "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
   },
   package_dir={"": "src"},
   packages=setuptools.find_packages(where="src")
)

