import numpy as np

# A calculator function that performs an operation on two numbers

def calculator(x, y, op):
  if op == "+":
    return x + y
  elif op == "-":
    return x - y
  elif op == "*":
    return x * y
  elif op == "/":
    return x/y
  elif op == "**":
    return x ** y
  
# A function that does point-wise addition (vector-vector addition)

def vectorAdd(vector1, vector2):
  vector = []
  if len(vector1) != len(vector2):
    print("Vector lengths must be equal")
    return []
    
  else:
    for i in range(len(vector1)):
      vector.append(vector1[i] + vector2[i])
    return vector

vectorAdd([1, 3], [2, 5])

# A function that adds all elements in a list

def addList(x):
  sum = 0
  for i in range(len(x)):
    sum += x[i]
  return sum