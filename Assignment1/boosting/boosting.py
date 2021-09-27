import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from dataset import car_valuation
from dataset import occupancy

# Choose dataset

# dataset = car_valuation
dataset = occupancy

X = dataset.training_data
Y = dataset.training_label

# Uncomment and runs this block of code to figure out optimal parameter

adam_boost = AdaBoostClassifier(random_state=42)
parameters = {'n_estimators': list(range(1, 20)),
              }
gs = GridSearchCV(adam_boost, parameters)
gs.fit(X, Y)
print(gs.best_params_)


# Output of Car valuation
########################
# {'n_estimators': 11} #
########################


# Output of Occupancy
########################
# {'n_estimators': 6} #
########################

val_boosting = AdaBoostClassifier(random_state=42)
estimators = list(range(1, 20))

train_scores, valid_scores = validation_curve(val_boosting, X, Y, "n_estimators", estimators, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(estimators, train_scores_mean, label="train_scores")
plt.plot(estimators, test_scores_mean, label="valid_scores")
plt.xticks(estimators)
plt.legend()
plt.title(f"Validation curve for AdamBoost (Weak learns vs Accuracy)")
plt.savefig('Validation Curve.png')

plt.clf()

learning_boosting = AdaBoostClassifier(n_estimators=6, random_state=42)
train_size = np.linspace(0.1, 1, 10)
train_sizes, train_scores, validation_scores = learning_curve(learning_boosting, X, Y, cv=5, train_sizes=train_size)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(validation_scores, axis=1)


plt.plot(train_sizes, train_scores_mean, label="train_scores")
plt.plot(train_sizes, val_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Learning curve for AdamBoost (Weak learns vs Accuracy)")
plt.savefig('Learning Curve.png')

learning_boosting.fit(X, Y)
predicted_label = learning_boosting.predict(dataset.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, dataset.test_label)}")
