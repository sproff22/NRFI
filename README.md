# NRFI
Building a probabilistic classifier to predict the probability of a run in the first inning of MLB games

This project walks through the steps to pull up to date data from the pybaseball API, preprocess it into a game log by each half inning, choose the best model for prediction, and evaluate model performance. 

## Contents

Data Preprocessing - Pulling data from pybaseball.statcast() and setting up the main dataframe
Model Selection - Creating base models including decision tree, xgboost, bayes classifiers, and random forest to find the best
Model Tuning - Tuning hyperperameters for random forest model
Model Evaluation - Evaluating expected accuracy of the tuned model
Predictions - Using user inputs to allow for predictions on new data

NOTE: Because data can change as more games are played, decisions made here may not apply for months or seasons in the future. Change models and hyperperameters as necessary. 
