from setuptools import find_packages, setup

setup(
    name="qrepeater_twin",
    version="2.0.0",
    description="Digital Twin of a Quantum Repeater with a predictive admission controller (EdgeLSTM)",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "qiskit>=1.0",
        "qiskit-aer>=0.14",
    ],
    extras_require={
        "dev": ["pytest>=7.4"],
        # XGBoost baseline (qrepeater_twin.baselines.XGBoostFidelityModel);
        # everything else works without it -- the comparison run skips this
        # baseline with a warning if it's absent.
        "xgboost": ["xgboost>=2.0"],
    },
)
