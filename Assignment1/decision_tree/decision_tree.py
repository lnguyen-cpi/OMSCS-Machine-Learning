import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree
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

ds_tree = tree.DecisionTreeClassifier()
parameters = {'max_depth': list(range(1, 20)),
              'criterion': ['gini', 'entropy'],
              'min_samples_split': range(2, 10),
              'min_samples_leaf': range(2, 10),
              }
gs = GridSearchCV(ds_tree, parameters)
gs.fit(X, Y)
print(gs.best_params_)

#                      Output for car valuation dataset
##########################################################################################
# {'criterion': 'entropy', 'max_depth': 9, 'min_samples_leaf': 2, 'min_samples_split': 3}#
##########################################################################################

#                      Output for occupancy dataset
##########################################################################################
# {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 4}#
##########################################################################################


val_decision_tree = tree.DecisionTreeClassifier(
    **{
        'criterion': 'entropy',
        'min_samples_leaf': 5,
        'min_samples_split': 4
    })

max_depth = list(range(1, 20))
train_scores, valid_scores = validation_curve(
    val_decision_tree,
    X,
    Y,
    "max_depth",
    max_depth,
    cv=5
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
plt.plot(max_depth, train_scores_mean, label="train_scores")
plt.plot(max_depth, test_scores_mean, label="valid_scores")
plt.xticks(max_depth)
plt.legend()
plt.title(f"Validation curve for Decision Tree (Max Depth vs Accuracy)")
plt.savefig('Validation Curve.png')

plt.clf()

# 'max_depth' = 11 gives best validation accuracy.
learning_decision_tree = tree.DecisionTreeClassifier(
    **{'criterion': 'entropy',
       'min_samples_leaf': 5,
       'min_samples_split': 4,
       'max_depth': 6
       })

train_size = np.linspace(0.1, 1, 10)
train_sizes, train_scores, validation_scores = learning_curve(
    learning_decision_tree,
    X,
    Y,
    cv=5,
    train_sizes=train_size
)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(validation_scores, axis=1)


plt.plot(train_sizes, train_scores_mean, label="train_scores")
plt.plot(train_sizes, val_scores_mean, label="valid_scores")
plt.legend()
plt.title(f"Learning curve for Decision Tree (Training Size vs Accuracy)")
plt.savefig('Learning Curve.png')

learning_decision_tree.fit(X, Y)
predicted_label = learning_decision_tree.predict(dataset.test_data)
print(f"Accuracy of test data {accuracy_score(predicted_label, dataset.test_label)}")
