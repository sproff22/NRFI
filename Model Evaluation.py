import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pybaseball
from pybaseball import schedule_and_record
from pybaseball import statcast
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from pybaseball import team_batting
from pybaseball import team_pitching
from pybaseball import batting_stats_range
from pybaseball import pitching_stats_range
from pybaseball import statcast_single_game
from pybaseball import playerid_reverse_lookup


X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(model_df_log.drop(["run_first_inning_yes"], axis=1), model_df_log["run_first_inning_yes"], test_size=0.2, random_state=42)

# Train the random forest classifier
forest = RandomForestClassifier(bootstrap = False, 
                                ccp_alpha= 0.0, 
                                class_weight= None, 
                                criterion= 'entropy', 
                                max_depth= 6, 
                                max_leaf_nodes= 20, #matters
                                min_impurity_decrease= 0.0, 
                                min_impurity_split= None, 
                                min_samples_split= 0.15, 
                                min_weight_fraction_leaf= 0.0, #matters
                                n_estimators= 64, 
                                n_jobs= 1, 
                                random_state= 42, 
                                warm_start= False)
forest.fit(X_train_rf, y_train_rf)

# save the model
#joblib.dump(forest, 'NRFI_rfc.pkl') 

# Make predictions on the test set
predictions_rf = forest.predict_proba(X_test_rf)[:,1:]
test_preds_rf = forest.predict(X_test_rf)

print(predictions_rf[:5])

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
#print("true mean:", y_test_rf['run_first_inning_yes'].mean())


plt.hist(predictions_rf)
plt.show()

plt.hist((predictions_rf + predictions_rf - (predictions_rf*predictions_rf)))
plt.show()



print("Mean of full inning yes prob",np.mean((predictions_rf + predictions_rf - (predictions_rf*predictions_rf))))

random_forest_test = model_df.sort_index(axis = 1)
random_forest_test['pred_prob'] = forest.predict_proba(model_df.drop(columns = ['hitting_runs','run_first_inning_yes']))[:,1:]

print("corr btwn pred prob and total runs",np.corrcoef(random_forest_test['pred_prob'], random_forest_test['hitting_runs'])[0,1])
print("corr btwen pred prob and binary runs",np.corrcoef(random_forest_test['pred_prob'], random_forest_test['run_first_inning_yes'])[0,1])

new_threshold = 0.292

y_prob = forest.predict_proba(X_test_rf)[:, 1]
# Reclassify the observations based on the new threshold
y_pred = np.where(y_prob > new_threshold, 1, 0)

# Compute the new confusion matrix
new_cm = confusion_matrix(y_test_rf, y_pred)
print(new_cm)

accuracy = accuracy_score(y_test_rf, y_pred)
print("Accuracy:", round(accuracy,3))

explainer = shap.TreeExplainer(forest)
instance = X_test_rf
shap_values = explainer.shap_values(instance)
print("Shapley Values:")
shap.summary_plot(shap_values, features=instance, plot_type='bar', plot_size=(10, 6))


# Show the plot
plt.show()

##
true_prob, pred_prob1 = calibration_curve(y_test_rf, predictions_rf, n_bins=10)
sns.histplot(predictions_rf, kde=True, stat='density', color='C0', alpha=0.5)
sns.lineplot(x=pred_prob1, y=true_prob, color='C1')
# Add labels and title
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration curve with density function')
# Visualize the calibration curve
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax.plot(pred_prob1, true_prob, "s-", label=f"Random forest classifier")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("True probability in each bin")
ax.set_title("Calibration Curve")
ax.legend()
plt

##
# Get the feature importances
importances = forest.feature_importances_
feature_names = X_train_rf.columns

# Visualize the feature importances
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

