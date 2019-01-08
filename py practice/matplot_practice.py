#%%
msg = "Hello World"
print(msg)

#%%
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

#%%
rand = np.random.RandomState(42)
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

#%%
plt.scatter(X[:, 0], X[:, 1])

#%%
indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]  # fancy indexing here
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='none', s=200)

#%%
w1 = 4
b1 = 1
w2 = 5
b2 = 2
plt.plot(X[:,0], X[:,1], 'go', label='Real data')
plt.plot(X[:,0], X[:,0] * w1 + b1, 'y', label='Prediction 1')
plt.plot(X[:,0], X[:,0] * w2 + b2, 'b', label='Prediction 2')
plt.legend()
plt.show()

#%%
plt.semilogx(np.arange(X.shape[0]), X[:,1])
plt.grid(True)
plt.title('Test accuracy by regularization (logistic)')
plt.show()

#%%
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))

#%%
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported

#%%
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted

#%%
# linestyle and color codes can be combined into a single non-keyword argument
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red

#%%
# adjust axis limits 
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

#%%
# set the x and y limits with a single call
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])

#%%
plt.plot(x, np.sin(x))
plt.axis('tight')

#%%
plt.plot(x, np.sin(x))
plt.axis('equal')

#%%
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

#%%
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend()

#%%
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot');