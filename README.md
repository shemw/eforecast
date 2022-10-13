# eforecast

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Virtual Environment](#virtual-environment)
4. [Installing Dependencies](#installing-dependencies)
5. [Running Tests](#running-tests)

## Introduction

Code for implementing a forecast model to predict daily houshold energy consumption 7 days in advance.
The forecaster applies a linear regression model based on time series features. 
It then applies XGBoost to predict the residuals from the linear regression model.
The two models are combined to form a hybrid boosted model, attempting to improve accuracy.

## Usage

Interact with the model using the notebook in `interactive/runner.ipynb` which contains instructions for each step.

The module is packaged with test data to get started. If you would like to explore other data sets, see
the notebook in `interactive/prepare_data.ipynb` which has links to a larger data source and procesing steps.

## Virtual Environment

It is recommended to run the model from a virtual environment (e.g. pyenv), installing the dependencies as specified below.

## Installing Dependencies

```bash
pip install -e .[dev,test]

OR

pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest
pyflakes .
```

