import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# from Classifiers.DecisionTree import DecisionTree
# from Classifiers.LogisticRegression import LogisticRegression
from Classifiers.RandomForest import RandomForest

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target  # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# clf = LogisticRegression(lr=0.001)
# clf = DecisionTree(max_depth=150)
clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


acc = accuracy(y_test, y_pred) * 100
print(f"Test Accuracy: ({acc:.2f}/100)")
