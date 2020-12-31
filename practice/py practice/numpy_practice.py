import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

if (0):
    x = np.arange(10)               # Create a list of sequential numbers over the range
    print (x, np.exp(x))
    plt.plot(x, np.exp(x))          # Plot the exp of each x point
    plt.show()                      # Display the plot

    y = np.arange(15).reshape(3,5)  # Create a (3,5) array of sequential numbers over the range
    print (y, np.exp(y))
    plt.plot(y, np.exp(y))          # Plot the exp of each x point
    plt.show()                      # Display the plot

    # np.sum works by summing the elements of an array. For a 2D array, you can sum
    # or by columns. When axis=0 for instance, you sum up all the rows and collapse
    # down to a single row. When axis=1, you sum up the columns and collapse down to
    # a single column. In other words, you are collapsing the axis parameter.
    print(np.sum(y, axis=0))        # Sum the values by collapsing the 0th dimension
    print(np.sum(y, axis=1))        # Sum the values by collapsing the 1st dimension