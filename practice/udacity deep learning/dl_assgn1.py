from dl_util import downloadBinFileWithProgress, extractTarGz, pickleImageFolders, unpickleImageDataSets, merge_datasets, randomizeDataLabels, saveFinalData
from sk_util import logReg

# Download the training and test image data in tar.gz format
# Data consists of images of letters 'A' thru 'J' in separate folders
# NOTE - this is a large 245M file !!!
trainFile = downloadBinFileWithProgress('https://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz')
testFile = downloadBinFileWithProgress('https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz')

# Extract the downloaded file into folders. One folder per letter (or class)
trainFolders = extractTarGz(trainFile, 10)
testFolders = extractTarGz(testFile, 10)

# Load each letter into a separate dataset, store them on disk and curate them 
# independently. That way the data is more manageable, or else it may not all fit 
# in memory. A few images might not be readable, we'll just skip them
#
# Convert the data into a 3D array (image index, x, y) of floating point values, 
# normalized to have approximately zero mean and standard deviation ~0.5
imageSize = 28  # Pixel width and height.
pixelDepth = 255.0  # Number of levels per pixel.
trainDatasets = pickleImageFolders(trainFolders, imageSize, pixelDepth, 45000)
testDatasets = pickleImageFolders(testFolders, imageSize, pixelDepth, 1800)

#unpickleImageDataSets(trainDatasets, imageSize)
unpickleImageDataSets(testDatasets, imageSize)

# We now have separate data files for each letter. Merge these into three 
# datasets of manageable size for training, validation and testing respectively.
# Each dataset has a mix of all the letters
numTrainDataPerLetter = 10
numValidationDataPerLetter = 2
numTestDataPerLetter = 4
mergedTrainData, mergedTrainLabels, mergedValidationData, mergedValidationLabels = merge_datasets(
    trainDatasets, imageSize, numTrainDataPerLetter, numValidationDataPerLetter)
mergedTestData, mergedTestLabels, _, _  = merge_datasets(testDatasets, imageSize, numTestDataPerLetter)

print('Training Data: %s, Training Labels: %s' % (mergedTrainData.shape, mergedTrainLabels.shape))
print('Validation Data: %s, Validation Labels: %s' % (mergedValidationData.shape, mergedValidationLabels.shape))
print('Testing Data: %s, Testing Labels: %s' % (mergedTestData.shape, mergedTestLabels.shape))

train_dataset, train_labels = randomizeDataLabels(mergedTrainData, mergedTrainLabels)
test_dataset, test_labels = randomizeDataLabels(mergedTestData, mergedTestLabels)
valid_dataset, valid_labels = randomizeDataLabels(mergedValidationData, mergedValidationLabels)

# Save all the final data for later processing
saveFinalData(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels)

# Run a standard model using LogisticRegression
logReg(train_dataset, train_labels, test_dataset, test_labels)