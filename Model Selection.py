# Convert to notebook form to best analyze different models

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier 
from xgboost import XGBRegressor
from sklearn.isotonic import IsotonicRegression
import math
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
import h2o
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
import shap
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

###

2*model_df.run_first_inning_yes.mean()

model_df[model_df['hitting_runs']>0.5].count()
model_df[model_df['hitting_runs']<0.5].count()

plt.figure(figsize=(16, 8))
sns.heatmap(model_df.corr())


##
X = model_df.drop(["hitting_runs","run_first_inning_yes"], axis=1)
y = model_df["hitting_runs"]
y_log = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# RIDGE REGRESSION
ridge_reg = Ridge(alpha=1.0)

ridge_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:",math.sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("R Square:", r2)

print(y_pred[:10])
print("mean runs predicted:",y_pred.mean())
print("maximum runs predicted:", y_pred.max())

print("minimum:",y_pred.min())

# Bayes Classifier

X_nb = model_df_log.drop(['run_first_inning_yes'], axis=1)
y_nb = model_df_log['run_first_inning_yes']

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.3, random_state=42)

nb_clf = GaussianNB()
nb_clf.fit(X_train_nb, y_train_nb)

# Evaluate the performance of the classifier on the test data
accuracy = nb_clf.score(X_test_nb, y_test_nb)
print('Accuracy:', accuracy)

y_prob = nb_clf.predict_proba(model_df_log.drop(columns = ['run_first_inning_yes']))[:,:1]
nb_test_df = model_df_log
nb_test_df['pred_prob'] = y_prob
print(nb_test_df['pred_prob'].mean())
print(nb_test_df['run_first_inning_yes'].mean())

# XG Boost -- Classifier 
X_train_xgclass, X_test_xgclass, y_train_xgclass, y_test_xgclass = train_test_split(model_df_log.drop(["run_first_inning_yes"], axis=1), model_df_log["run_first_inning_yes"], test_size=0.2, random_state=42)

xgb_class = XGBClassifier(objective = 'binary:logistic',
                          reg_lambda=.001)
xgb_class.fit(X_train_xgclass, y_train_xgclass)

xgb_class_prob = xgb_class.predict_proba(X_test_xgclass)
xgb_class_pred = xgb_class.predict(X_test_xgclass)


cm_xgbc = confusion_matrix(y_test_xgclass, xgb_class_pred)

print(cm_xgbc)
accuracy_xgbc = (cm_xgbc[0,0] + cm_xgbc[1,1]) / np.sum(cm_xgbc)
precision_xgbc = cm_xgbc[0,0] / (cm_xgbc[0,0] + cm_xgbc[1,0])
recall_xgbc = cm_xgbc[0,0] / (cm_xgbc[0,0] + cm_xgbc[0,1])
f1_score_xgbc = 2 * (precision_xgbc * recall_xgbc) / (precision_xgbc + recall_xgbc)
print("acuracy:",accuracy_xgbc)
print("precision:",precision_xgbc)
print("recall:",recall_xgbc)
print("f1 score:",f1_score_xgbc)

xgb_class_prob_test = xgb_class.predict_proba(model_df.drop(columns = ['run_first_inning_yes','hitting_runs']))[:,:1]

print("overall predicted mean:",xgb_class_prob_test.mean())
print("true mean:",model_df['run_first_inning_yes'].mean())

print(xgb_class_prob_test)

# DECISION TREE CLASSIFIER
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(model_df_log.drop(["run_first_inning_yes"], axis=1), model_df_log["run_first_inning_yes"], test_size=0.2, random_state=42)


# Build best model
final_clf = DecisionTreeClassifier()
final_clf.fit(X_train_dt, y_train_dt)

# Predict the labels of the test set using the final classifier
y_pred_dt = final_clf.predict(X_test_dt)
y_prob_dt = clf.predict_proba(X_test_dt)

# Calculate the accuracy score and confusion matrix of the final classifier
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
cm_dt = confusion_matrix(y_test_dt, y_pred_dt)

# Print the best depth, accuracy score, and confusion matrix
print("Best depth:", best_depth)
print("Accuracy:", accuracy_dt)
print("Confusion matrix:")
print(cm_dt)

print(y_prob_dt[:5])

# random forest binary
model_df_log.sort_index(axis = 1)
# Split the data into training and test sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(model_df_log.drop(["run_first_inning_yes"], axis=1), model_df_log["run_first_inning_yes"], test_size=0.2, random_state=42)

# Train the random forest classifier
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train_rf, y_train_rf)

# Make predictions on the test set
predictions_rf = forest.predict_proba(X_test_rf)[:,1:]
test_preds_rf = forest.predict(X_test_rf)

# The first column of predictions is the probability of the negative class
# and the second column is the probability of the positive class
print(predictions_rf[:5])

# Compute the confusion matrix
cm_rf = confusion_matrix(y_test_rf, test_preds_rf)

print("confusion matrix:",
      cm_rf)
accuracy_rf = (cm_rf[0,0] + cm_rf[1,1]) / np.sum(cm_rf)
precision_rf = cm_rf[0,0] / (cm_rf[0,0] + cm_rf[1,0])
recall_rf = cm_rf[0,0] / (cm_rf[0,0] + cm_rf[0,1])
f1_score_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
print("acuracy:",accuracy_rf)
print("precision:",precision_rf)
print("recall:",recall_rf)
print("f1 score:",f1_score_rf)

print("mean probability:", predictions_rf.mean())

plt.hist(predictions_rf)
plt.show()

plt.hist(predictions_rf + predictions_rf - (predictions_rf*predictions_rf))
plt.show()

######

## RANDOM FOREST IS BEST MODEL
