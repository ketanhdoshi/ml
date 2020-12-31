# Import the iris dataset to classify flowers into three varieties of iris based on their
# sepal width/length and petal width/length
from sklearn.datasets import load_iris

from sklearn import tree # Decision Tree
import numpy as np  # Import numpy library

# Load the training dataset, with 150 features
iris = load_iris()

# Print the metadata about the features ie. list of features and labels
print(iris.feature_names)
print(iris.target_names)

# Print the first row of feature data and label data
print(iris.data[0])
print(iris.target[0])

# Index of the first occurrence of each of the three variants, in the training data.
# Below we will set aside those three rows of data as our testing data
test_idx = [0, 50, 100]

# Prepare the training data by removing three rows of data. We will use those three rows
# as our testing data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Now prepare our testing data with those three rows of data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Create and train Decision Tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# Predict our testing data
print(test_target)
print(clf.predict(test_data))

# Visualise the Decision Tree that was created
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, 
    out_file=dot_data, 
    feature_names=iris.feature_names, 
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf ('iris.pdf')