# NRFI
Building a probabilistic classifier to predict the probability of a run in the first inning of MLB games

This project walks through the steps to pull up-to-date data from the pybaseball API, preprocess it into a game log by each half inning, choose the best prediction model, and evaluate model performance. 

## Contents

Data Preprocessing - Pulling data from pybaseball.statcast() and setting up the main dataframe

Model Selection - Creating base models including decision tree, xgboost, bayes classifiers, and random forest to find the best

Model Tuning - Tuning hyperparameters for random forest model

Model Evaluation - Evaluating the expected accuracy of the tuned model

Predictions - Using user inputs to allow for predictions on new data

NOTE: Because data can change as more games are played, decisions made here may not apply for months or seasons in the future. Change models and hyperparameters as necessary. 

NOTE: Predictions are set to be made by the user in the code. Variables are set to allow for easier deployment to a GUI or web app if desired.

## Outcome

When tracked through the first four months of the 2023 MLB season (including regular data updates), the model performed adequately, with an accuracy rate of 61%. Using the predictions module, users can compare the model's predicted odds of a NRFI (No Run First Inning) with Vegas's implied odds given the odds on any public sportsbook. If a user chose to bet $100 on every game with a >5% difference between model odds and Vegas odds, they would have made a total of $3,400 in that timeframe.
