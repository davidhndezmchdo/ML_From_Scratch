import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

# from Classifiers.KNN import KNN
from Classifiers.PCA import PCA

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target  # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
)

# plt.figure()
# plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors="k", s=20)
# plt.show()

# # KNN
# clf = KNN(k=3)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# #
# print(predictions)
# #
# acc = np.sum(predictions == y_test) / len(y_test)
# print(acc)


# PCA
pca = PCA(2)
pca.fit(X_train)
X_projected = pca.transform(X)
#
print("Shape of X: ", X.shape)
print("Shape of transformed X: ", X_projected.shape)
#
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
#
plt.scatter(x1, x2, c=y, edgecolor=None, alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
#
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
