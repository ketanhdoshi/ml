#-----------------------------------------------------------
# Simple program to validate that tensorflow is working
#-----------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input1 = tf.ones((2, 3))
input2 = tf.reshape(tf.range(1, 7, dtype=tf.float32), (2, 3))
output = input1 + input2

with tf.Session():
  result = output.eval()
print(result)