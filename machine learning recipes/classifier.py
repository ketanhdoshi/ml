from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

# Write our own K Neighbours classifier
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

from sklearn import datasets

# Load the training dataset
iris = datasets.load_iris()

# X = features, y = labels
X = iris.data
y = iris.target

# Partition the dataset into 50% training data and 50% test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Create K Neighbours classifier
my_classifier = ScrappyKNN()

# Train the classifier with the training data
my_classifier.fit(X_train, y_train)

# Predict using the test data
predictions = my_classifier.predict(X_test)
print(predictions)

# Check accuracy of predictions by comparing them to test labels
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

