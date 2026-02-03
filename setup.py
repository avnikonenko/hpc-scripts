from setuptools import find_packages, setup
from pathlib import Path


def read_readme() -> str:
    readme = Path(__file__).parent / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


setup(
    name="hpc-stats-scripts",
    version="1.2",
    description="Utilities for HPC clusters including PBS/Slurm job statistics and a psutil-based resource monitor.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="hpc-stats-scripts contributors",
    python_requires=">=3.9",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["psutil"],
    extras_require={
        "plot": ["matplotlib", "numpy"],
        "gpu": ["nvidia-ml-py3"],
        "all": ["matplotlib", "numpy", "nvidia-ml-py3"],
    },
    entry_points={
        "console_scripts": [
            "pbs-bulk-user-stats=hpc_scripts.pbs_bulk_user_stats:main",
            "psutil-monitor=hpc_scripts.psutil_monitor:main",
            "slurm-bulk-user-stats=hpc_scripts.slurm_bulk_user_stats:main",
        ]
    },
)
