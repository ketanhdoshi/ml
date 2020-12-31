# Here is how serializing with file I/O is used with pickle library
import pickle

mylist = ["This", "is", 4, 13327]
# Open the file binary.dat for writing. The letter r before the
# filename string is used to prevent backslash escaping.
myfile = open(r"binary.dat", "wb")
pickle.dump(mylist, myfile)
myfile.close()

# Open the file for reading.
myfile = open(r"binary.dat", "rb")
loadedlist = pickle.load(myfile)
myfile.close()
print(loadedlist)

myfile = open(r"text.txt", "w")
myfile.write("This is a sample string")
myfile.close()

myfile = open(r"text.txt")
print(myfile.read())
myfile.close()