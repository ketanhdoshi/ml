# Import Decision Tree algorithm
from sklearn import tree

# Machine learning 'Hello World' example to classify a fruit as an apple or orange
# First feature is weight, second feature is texture (0 for 'bumpy', 1 for 'smooth')
# Labels are 0 for 'apple', 1 for 'orange'
features = [[140, 1], [130,1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

# Create the classifier
clf = tree.DecisionTreeClassifier()
# and train it
clf = clf.fit(features, labels)

# Now use the trained classifier to predict the label for a new input
print (clf.predict([[160, 0]]))