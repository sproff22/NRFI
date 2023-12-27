mport statsmodels.api as sm
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
import joblib
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
from sklearn.metrics import roc_curve, auc
import sklearn
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

##

# n_estimators
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(X_train_rf, y_train_rf)
    train_pred = rf.predict(X_train_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_rf, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_rf, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()

#n_estimators = 64

# max_depths
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators = 64, n_jobs=-1)
    rf.fit(X_train_rf, y_train_rf)
    train_pred = rf.predict(X_train_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_rf, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_rf, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('‘AUC score’')
plt.xlabel('‘Tree depth’')
plt.show()

#max_depths = 6

# min_samples_splits
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    rf = RandomForestClassifier(min_samples_split=min_samples_split, n_jobs=-1)
    rf.fit(X_train_rf, y_train_rf)
    train_pred = rf.predict(X_train_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_rf, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_rf, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label='”Train AUC”')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='”Test AUC”')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('‘AUC score’')
plt.xlabel('‘min samples split’')
plt.show()
#min_samples_split = .15

# min_samples_leafs
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, min_samples_split=.15, n_estimators = 64, n_jobs=-1)
    rf.fit(X_train_rf, y_train_rf)
    train_pred = rf.predict(X_train_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_rf, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test_rf)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_rf, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, 'b', label='”Train AUC”')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='”Test AUC”')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('‘AUC score’')
plt.xlabel('‘min samples leaf’')
plt.show()

#delete min_samples_leaf

# GRID SEARCH
param_grid = {
    'n_estimators': [64],
    'criterion': ['entropy'],
    'max_depth': [6],
    'min_samples_split': [.15],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.01, 0.05],
    'min_impurity_split': [None, 0.1, 0.2],
    'bootstrap': [True, False],
    'n_jobs': [1],
    'random_state': [42],
    'warm_start': [False, True],
    'class_weight': [None],
    'ccp_alpha': [0.0,0.5,1,3],
    'class_weight': [None]
}

rfc = RandomForestClassifier()
grid_search = GridSearchCV(
    rfc, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='balanced_accuracy')

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(model_df_log.drop(["run_first_inning_yes"], axis=1), model_df_log["run_first_inning_yes"], test_size=0.2, random_state=42)

grid_search.fit(X_train_rf, y_train_rf)

print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


