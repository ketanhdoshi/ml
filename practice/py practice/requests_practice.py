import requests
import sys

# --------------------------------------------------------------------------------------------
# Make a GET request
# --------------------------------------------------------------------------------------------
def doGet (url):
    req = requests.get(url)
    print(req.status_code, req.encoding)
    #print(req.text)

# --------------------------------------------------------------------------------------------
# Make a POST request
# --------------------------------------------------------------------------------------------
def doPost(url, data):
    req = requests.post(url, params = data)
 
# Similarly can make PUT and DELETE requests

# --------------------------------------------------------------------------------------------
# Download Internet images or files to local file
# --------------------------------------------------------------------------------------------
def downloadBinFile(url, filename):
    # download the url contents in binary format
    r = requests.get(url)
 
    # open method to open a file on your system and write the contents
    with open(filename, "wb") as code:
        code.write(r.content)
            
# --------------------------------------------------------------------------------------------
# Call the functions
# --------------------------------------------------------------------------------------------

doGet('http://www.google.com/')
data = {"email":"info@tutsplus.com",
            "password":"12345"}
doPost('http://www.google.com/', data)

# Download image
downloadBinFile('https://www.python.org/static/opengraph-icon-200x200.png', 'myPythonImage.png')

# Download PDF file
downloadBinFile('https://static.googleusercontent.com/media/www.google.com/en//googleblogs/pdfs/google_predicting_the_present.pdf', 'myPdfFile.pdf')

# Download ZIP file
downloadBinFile('https://codeload.github.com/fogleman/Minecraft/zip/master', 'myZIPFile.zip')

# Similarly can download a video file



