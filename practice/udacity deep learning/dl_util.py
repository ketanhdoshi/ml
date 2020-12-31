import sys
import os
import string
import requests
import tarfile
import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# Download Internet file while showing percent progress
# NB: The file size is taken from the Content-Length HTTP header but that is not returned
# from all download URLs, so check that first
# NB: Bizarre behaviour - if I put a breakpoint in the function then the iter_content loop
# misses the last chunk. If there is no breakpoint, it works fine
# --------------------------------------------------------------------------------------------
def downloadBinFileWithProgress(url):
    # Local file is same as URL filename
    filename = url.split('/')[-1]
    
    # Skip download if file exists
    if os.path.exists(filename):
        statinfo = os.stat(filename)
        print('Found file %s, size = %d bytes' % (filename, statinfo.st_size))
        return filename
  
    # Get the file
    print("Downloading %s" % filename)
    r = requests.get(url, stream=True, headers={'Accept-Encoding': None})
    # Use Head just to get the headers without the file itself
    #r = requests.head(url, stream=True, headers={'Accept-Encoding': None})

    # File size
    filesize = int(r.headers['Content-Length'])
    print('File size = ', r.headers.get('content-length'))

    # Write to the file in chunks
    dl = 0
    with open(filename, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=4096):
            dl += len(chunk)
            fd.write(chunk)

            # Show the progress indicator with each '=' representing a 2 percent piece of the file
            # In each iteration it prints a full 100 percent bar, with a sequence of '=' for the
            # downloaded pieces followed by a sequence of ' ' for the yet undownloaded pieces.
            # The "\r" is a carriage return which brings the cursor back to the beginning of the same
            # line without doing a line feed ie. "\n". Because of this the progress indicator just
            # overwrites what it wrote in each iteration.
            numTwoPercentPieces = int((50 * dl) / filesize)
            sys.stdout.write("\r[%s%s]" % ('=' * numTwoPercentPieces, ' ' * (50-numTwoPercentPieces)) )    
            sys.stdout.flush()

    print('\nDownload Complete!')
    return filename

# --------------------------------------------------------------------------------------------
# Extract tar file
# --------------------------------------------------------------------------------------------
def extractTarGz (filename, expectedFolders, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz

    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
    
        # Extract the tar file into current directory
        dataRoot = '.' 
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(dataRoot)
        tar.close()
  
    # Check that we go the expected number of folders after extraction
    dataFolders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(dataFolders) != expectedFolders:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
            expectedFolders, len(dataFolders)))
 
    print('Folders are ', dataFolders)
    return dataFolders

# --------------------------------------------------------------------------------------------
# Load data for all the images in the given folder
# --------------------------------------------------------------------------------------------
def loadLetter(folder, imageSize, pixelDepth, minNumImages):
    imageFiles = os.listdir(folder)
    dataset = np.ndarray(shape=(len(imageFiles), imageSize, imageSize),
                            dtype=np.float32)

    # Process each image file in the folder
    numImages = 0
    for image in imageFiles:
        imageFile = os.path.join(folder, image)
        try:
            # Each pixel in the image data ranges between 0 and 255. Normalise this to
            # range between -0.5 to 0.5, so that it has a mean of 0, and standard deviation
            # of 0.5
            imageData = (imageio.imread(imageFile).astype(float) - 
                    pixelDepth / 2) / pixelDepth

            # Check that the dimensions of the image that we read is as expected
            if imageData.shape != (imageSize, imageSize):
                raise Exception('Unexpected image shape: %s' % str(imageData.shape))

            # Save the image data in our dataset, using slicing syntax
            dataset[numImages, :, :] = imageData
            numImages = numImages + 1
        except (IOError, ValueError) as e:
            print('Could not read:', imageFile, ':', e, '- it\'s ok, skipping.')
    
    # Use slicing syntax to get a subset of the dataset upto the number of images 
    # successfully processed
    dataset = dataset[0:numImages, :, :]
    if numImages < minNumImages:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (numImages, minNumImages))
    
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

# --------------------------------------------------------------------------------------------
# Load data for all the images in the given folder, and serialise it to a file with pickle
# --------------------------------------------------------------------------------------------
def pickleImageFolders(dataFolders, imageSize, pixelDepth, minNumImagesPerClass, force=False):
    datasetNames = []
    for folder in dataFolders:
        setFilename = folder + '.pickle'
        datasetNames.append(setFilename)

        if os.path.exists(setFilename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % setFilename)
        else:
            print('Pickling %s.' % setFilename)
            dataset = loadLetter(folder, imageSize, pixelDepth, minNumImagesPerClass)
            try:
                with open(setFilename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', setFilename, ':', e)
  
    return datasetNames

# --------------------------------------------------------------------------------------------
# Deserialise the previously saved data files and load it back using pickle
# --------------------------------------------------------------------------------------------
def unpickleImageDataSets(dataFiles, imageSize, visualiseImages=False):
    countImagesList = []  # List of image counts in each data file

    for fileName in dataFiles:
        if os.path.exists(fileName):
            with open(fileName, 'rb') as f:
                dataset = pickle.load(f)

                # Check the dimensions of the loaded data
                print('Unpickling file %s, shape is %s' % (fileName, dataset.shape))
                numImages, imageWidth, imageHeight = dataset.shape
                if (imageWidth != imageSize or imageHeight != imageSize):
                    print('Image size is incorrect. Width = %d, Height = %d' % (imageWidth, imageHeight))
                
                # Count of images in this file
                countImagesList.append(numImages)

                # Visualise a few sample images in a grid to make sure the data looks good
                if (visualiseImages):
                    gridSize = 8
                    showImages(dataset, gridSize)
        else:
            print ('%s file missing - skipping unpickling.' % fileName)

    # Print stats on the image counts
    showStats(countImagesList)

# --------------------------------------------------------------------------------------------
# Display a sample of the images in a grid
# --------------------------------------------------------------------------------------------
def showImages(images, gridSize):
    # Figure size (width, height) in inches
    fig = plt.figure(figsize=(15, 9))

    # Adjust layouts for the subplots. Set the left side of the suplots to 0, the right side to 1, and
    # The bottom to 0 and the top to 1. The height of the blank space between the suplots is set 
    # at 0.005 and the width is set at 0.05
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # Iterate through each image to create a grid of images
    for i in range(gridSize * gridSize):
        # Initialize the subplots: add a subplot in the grid of (gridSize x gridSize), at the i+1-th position
        ax = fig.add_subplot(gridSize, gridSize, i + 1, xticks=[], yticks=[])

        # Display an image at the i-th position. Take a binary colourmap which results in 
        # black, gray values and white colors. Use 'nearest' interpolation method so that your 
        # data is interpolated in such a way that it isn’t smooth
        ax.imshow(images[i], cmap=plt.cm.binary, interpolation='nearest')

        # label the image with the target value, in the bottom-left at coordinates (0,7) of 
        # each subplot 
        ax.text(0, 7, str(i))

    plt.show()


# --------------------------------------------------------------------------------------------
# Display stats based on a list of counts
# --------------------------------------------------------------------------------------------
def showStats(countList):
    numData = np.array(countList, dtype=int)
    print('Stats on Counts: Min = %d, Max = %d, Mean = %d, Std Dev = %d, Variance = %d' % 
        (np.min(countList), np.max(countList), np.mean(countList), np.std(countList), np.var(countList)))

    # Get the first N letters of the alphabet
    lenList=len(numData)
    s = list(string.ascii_uppercase)[:lenList]

    # Show a bar graph of the counts
    index = np.arange(lenList)
    plt.bar(index, countList)
    plt.xlabel('Letters', fontsize=14)
    plt.ylabel('No of Images', fontsize=14)
    plt.xticks(index, s, fontsize=12, rotation=30)
    plt.title('Images for each alphabet letter')
    plt.show()

# --------------------------------------------------------------------------------------------
# Create nparrays for the image data and labels
# --------------------------------------------------------------------------------------------
def makeDataLabelArray(numData, imageSize):
    if numData:
        dataset = np.ndarray((numData, imageSize, imageSize), dtype=np.float32)
        labels = np.ndarray(numData, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

# --------------------------------------------------------------------------------------------
# Extract the given number of data items from the input data and fill it at the right
# location in the output data and labels
# --------------------------------------------------------------------------------------------
def pickData(inData, outData, outLabels, letterIndex, inStart, numData):
    if (numData == 0):
        return
    
    inEnd = inStart + numData
    pickedSet = inData[inStart:inEnd, :, :]
    outStart = letterIndex * numData
    outEnd = outStart + numData
    outData[outStart:outEnd, :, :] = pickedSet
    outLabels[outStart:outEnd] = letterIndex

# --------------------------------------------------------------------------------------------
# Load saved image data files for each alphabet letter using pickle. Extract the given number
# of items from each letter and merge them into a single data set to be used for 
# training/testing or validation
# --------------------------------------------------------------------------------------------
def merge_datasets(pickle_files, imageSize, numTDataPerLetter, numVDataPerLetter=0):
    
    # Prepare empty arrays to hold the training and validation data and labels
    numLetters = len(pickle_files)
    validationData, validationLabels = makeDataLabelArray(numVDataPerLetter * numLetters, imageSize)
    trainData, trainLabels = makeDataLabelArray(numTDataPerLetter * numLetters, imageSize)
    
    # Process image data file for each letter
    for letterIndex, pickle_file in enumerate(pickle_files):       
        try:
            with open(pickle_file, 'rb') as f:
                # Load the image data
                letterData = pickle.load(f)
                
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letterData)
                
                # Extract the required number of data items from the loaded data and fill that
                # into the training and validation data and labels
                pickData(letterData, validationData, validationLabels, letterIndex, 0, numVDataPerLetter)
                pickData(letterData, trainData, trainLabels, letterIndex, numVDataPerLetter, numTDataPerLetter)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    #showImages(trainData, 4)
    return trainData, trainLabels, validationData, validationLabels 

# --------------------------------------------------------------------------------------------
# Randomly shuffle the data and labels
# --------------------------------------------------------------------------------------------
def randomizeDataLabels(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])

    # Use fancy indexing, by using the permuted array as an index into the data and label
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# --------------------------------------------------------------------------------------------
# Save all the final data sets
# --------------------------------------------------------------------------------------------
def saveFinalData(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
    dataRoot = '.' 
    pickle_file = os.path.join(dataRoot, 'notMNIST.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Saved merged pickle to ', pickle_file, ', Compressed size:', statinfo.st_size)

# --------------------------------------------------------------------------------------------
# Show a Confusion Matrix to visualise the difference between expected and predicted labels
# --------------------------------------------------------------------------------------------
def showConfusionMatrix(cm):

    plt.figure(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 15)
    plt.colorbar()

    # cm is a square, so width and height should be the same
    width, height = cm.shape
    tick_marks = np.arange(width)
    tick_labels = ["%d" % i for i in range(width)]
    plt.xticks(tick_marks, tick_labels, rotation=45, size = 10)
    plt.yticks(tick_marks, tick_labels, size = 10)

    plt.tight_layout()
    plt.ylabel('Actual label', size = 15)
    plt.xlabel('Predicted label', size = 15)
    
    # Annotate each cell of the matrix with the count for that cell
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
            horizontalalignment='center',
            verticalalignment='center')

    plt.show()

# --------------------------------------------------------------------------------------------
# Show the images which were incorrectly predicted, annotated with the expected 
# and predicted values
# --------------------------------------------------------------------------------------------
def showMispredicted (flatTest, test_labels, pred):
    # Find the indexes which are incorrectly predicted
    misPredictedIndexes = []
    size = len(test_labels)
    for index in range(size):
        if test_labels[index] != pred[index]: 
            misPredictedIndexes.append(index)

    plt.figure(figsize=(15,9))
    gridSize = 6
    maxShow = min (len(misPredictedIndexes), gridSize * gridSize)
    for plotIndex, badIndex in enumerate(misPredictedIndexes[0:maxShow]):
        plt.subplot(gridSize, gridSize, plotIndex + 1)
        plt.imshow(np.reshape(flatTest[badIndex], (28,28)), cmap=plt.cm.gray)
        plt.title('Predicted: {}, Actual: {}'.format(pred[badIndex], test_labels[badIndex]))
    plt.show()

# --------------------------------------------------------------------------------------------
# Convert these into test cases
# --------------------------------------------------------------------------------------------
if (False):
    #These did not work due to not getting the right content-length
    #downloadBinFileWithProgress('https://github.com/tensorflow/tensorflow/archive/master.zip')
    #downloadBinFileWithProgress('https://static.googleusercontent.com/media/www.google.com/en//googleblogs/pdfs/google_predicting_the_present.pdf')

    #Use this to test the function
    downloadBinFileWithProgress('https://portableapps.duckduckgo.com/On-ScreenKeyboardPortable_2.1.paf.exe')

    testDatasets = ['notMNIST_small\\A.pickle', 'notMNIST_small\\B.pickle']
    unpickleFiles(testDatasets)
