from setuptools import setup, find_packages
from pathlib import Path

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Financial and Insurance Industry
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Natural Language :: English
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Office/Business :: Financial
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name="option-wiz",
    version="0.0.1",
    description="Package for option calculations",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url='https://github.com/nisaac21/option_py',
    author="Neil Isaac et al.",
    author_email="neilisaac08@gmail.com",
    license='MIT',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    packages=find_packages(where="option_py"),
    package_dir={"": "option_py"},
    python_requires=">=3.6"
)
