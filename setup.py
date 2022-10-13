from os import path
from setuptools import setup


here = path.abspath(path.dirname(__file__))

setup(
    name="eforecast",

    packages=[],

    package_data={},

    install_requires=[

	"pandas==1.5.0",
	"matplotlib==3.6.1",
	"numpy==1.23.4",
	"datetime==4.7",
	"seaborn==0.12.0",
	"xgboost==1.6.2",
	"scikit-learn==1.1.2",
	"tqdm==4.64.1",
	"statsmodels==0.13.2",
	"ciso8601==2.2.0",
	"jupyter==1.0.0"

    ],
    extras_require={
        "dev": [],
        "test": [
            "pyflakes==2.2.0",
            "pytest==6.1.2",
            "pytest-cov==2.10.1",
        ],
    },
    include_package_data=True,
)
