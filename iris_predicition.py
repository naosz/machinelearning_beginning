from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


iris = load_iris()

#training feature
print(iris.data)

#training target / results
print(iris.target)


X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

#define the model
knn = KNeighborsClassifier(n_neighbors=1)

#print the model parameters
print(knn)


#train
knn.fit(X,y)

#create new test list
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

#predict 2nd sample
second_test = knn.predict(X_new)
print(second_test)

#logistic regression
logreg = LogisticRegression()

logreg.fit(X,y)

y_pred = logreg.predict(X)
print(len(y_pred))

