import os
import sys

import urllib.request   # Use urllib library

# --------------------------------------------------------------------------------------------
# fetch a URL
# --------------------------------------------------------------------------------------------
def fetchURL (url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
        print(html)

# --------------------------------------------------------------------------------------------
# retrieve a resource
# --------------------------------------------------------------------------------------------
def fetchResource(url):
    filename, headers = urllib.request.urlretrieve('http://python.org/')
    html = open(filename)

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def fetchImage(url, filename):
    # Copy a network object to a local file
    # Note that urlretrieve has been deprecated
    urllib.request.urlretrieve(url, filename)

# --------------------------------------------------------------------------------------------
# A hook to report the progress of a download. Reports every 5% change in download progress
# --------------------------------------------------------------------------------------------
last_percent_reported = 0
def fetchProgressHook(count, blockSize, totalSize):
    # Older FTP download servers don't return the retrieval file size
    if (totalSize < 0):
        return

    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
      
    last_percent_reported = percent

# --------------------------------------------------------------------------------------------
# Download a file if not present, and make sure it's the right size
# --------------------------------------------------------------------------------------------
def fetchFileWithProgress(url, filename, expectedBytes, force=False):
    # Store file in current directory
    dataRoot = '.' 
    destFilename = os.path.join(dataRoot, filename)

    # Download file if not present locally
    if force or not os.path.exists(destFilename):
        print('Attempting to download:', filename) 
        filename, _ = urllib.request.urlretrieve(url + filename, destFilename, reporthook=fetchProgressHook)
        print('\nDownload Complete!')
    
    # Check if file size matches the expected size
    statinfo = os.stat(destFilename)
    if statinfo.st_size == expectedBytes:
        print('Found and verified', destFilename)
    else:
        raise Exception('Failed to verify ' + destFilename + '. Can you get to it with a browser?')
    
    return destFilename

# --------------------------------------------------------------------------------------------
# Call the functions
# --------------------------------------------------------------------------------------------
# NB: None of these return total file size in the download hook, so no progress indicator gets printed
#tryFilename = fetchFileWithProgress('https://static.googleusercontent.com/media/www.google.com/en//googleblogs/pdfs/', 'google_predicting_the_present.pdf', 311671)
#tryFilename = fetchFileWithProgress('https://github.com/tensorflow/tensorflow/archive/', 'master.zip', 311671)

tryFilename = fetchFileWithProgress('https://portableapps.duckduckgo.com/', 'On-ScreenKeyboardPortable_2.1.paf.exe', 550640)

#trainFilename = fetchFileWithProgress('https://commondatastorage.googleapis.com/books1000/', 'notMNIST_large.tar.gz', 247336696)
#testFilename = fetchFileWithProgress('https://commondatastorage.googleapis.com/books1000/', 'notMNIST_small.tar.gz', 8458043)


fetchURL('http://www.google.com/')
fetchResource('http://www.google.com/')
fetchImage('https://www.python.org/static/opengraph-icon-200x200.png', "myPythonImage.png")