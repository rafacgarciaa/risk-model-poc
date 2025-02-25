# POC Risk Engine Model (Home Credit Default Risk)

## The Business problem

This is a binary Classification task: we want to predict whether the person applying for a home credit will be able to repay their debt or not. Our model will have to predict a 1 indicating the client will have payment difficulties: he/she will have late payment of more than X days on at least one of the first Y installments of the loan in our sample, 0 in all other cases.

We will use [Area Under the ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=es_419) as the evaluation metric, so our models will have to return the probabilities that a loan is not paid for each input data.

## About the data

The original dataset is composed of multiple files with different information about loans taken. In this project, we will work exclusively with the primary files: `application_train_aai.csv` and `application_test_aai.csv`. The URLs are provided in a google drive by a third party and such files have real but "cross walked" data to avoid legal issues.

## Installation

A `requirements.txt` file is provided with all the needed Python libraries for running this project. For installing the dependencies just run:

```console
$ pip install -r requirements.txt
```

*Note:* We encourage you to install those inside a virtual environment.

## Code Style

Following a style guide keeps the code's aesthetics clean and improves readability, making contributions and code reviews easier. Automated Python code formatters make sure your codebase stays in a consistent style without any manual work on your end. If adhering to a specific style of coding is important to you, employing an automated to do that job is the obvious thing to do. This avoids bike-shedding on nitpicks during code reviews, saving you an enormous amount of time overall.

We use [Black](https://black.readthedocs.io/) and [isort](https://pycqa.github.io/isort/) for automated code formatting in this project, you can run it with:

```console
$ isort --profile=black . && black --line-length 88 .
```

## Tests

We provide unit tests trying to cover minimum requirements of correctness. To run just execute:

```console
$ pytest tests/
```

## Final Model

We've documented our POC in the Jupyter file `RE-POC.ipynb` where we've provided details about the full path walked and the different classifiers tested along with the ROC and AUC curves for each case.

To finish, the result model is located in the python file `src/predictor.py`.
