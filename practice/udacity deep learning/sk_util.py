from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from dl_util import showConfusionMatrix, showMispredicted

# --------------------------------------------------------------------------------------------
# Flatten the image for consumption by scikit
# The dataset has image data in format (imageSize x imageSize) eg (28, 28)
# It needs to be flattened as scikit expects one column per feature eg. (784)
# --------------------------------------------------------------------------------------------
def flattenImage (images):
    flatImage = images.reshape(len(images), -1)
    return flatImage

# --------------------------------------------------------------------------------------------
# Use a Logistic Regression model for predictions
# --------------------------------------------------------------------------------------------
def logReg (train_dataset, train_labels, test_dataset, test_labels):

    # Flatten the images
    flatTrain = flattenImage (train_dataset)
    flatTest = flattenImage (test_dataset)

    # Train the logistic regression classifier and get predictions for the test set
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='multinomial').fit(flatTrain, train_labels)
    pred = clf.predict(flatTest)
    
    # Evaluate the accuracy of the model
    print (clf.score (flatTest, test_labels))

    # Evaluate the model using a confusion matrix
    cm = metrics.confusion_matrix(test_labels, pred)
    showConfusionMatrix(cm)

    # Display the incorrectly predicted images
    showMispredicted(flatTest, test_labels, pred)