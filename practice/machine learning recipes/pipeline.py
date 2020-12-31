from sklearn import datasets

# Load the training dataset
iris = datasets.load_iris()

# X = features, y = labels
X = iris.data
y = iris.target

# Partition the dataset into 50% training data and 50% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Create Decision Tree classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# Train the classifier with the training data
my_classifier.fit(X_train, y_train)

# Predict using the test data
predictions = my_classifier.predict(X_test)
print(predictions)

# Check accuracy of predictions by comparing them to test labels
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

# Create K Neighbours classifier
from sklearn import neighbors
my_classifier = neighbors.KNeighborsClassifier()

# Train the classifier with the training data
my_classifier.fit(X_train, y_train)

# Predict using the test data
predictions = my_classifier.predict(X_test)
print(predictions)

# Check accuracy of predictions by comparing them to test labels
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

