import numpy as np

#step 1 : Define input features i.e X

#                  [x1 , x2 , x3]
input = np.array([2.0 , 3.0 ,4.0])
print("X : ",input)




#step 2 :  Define weights ie w

#                  [W1 , W2 , W3]
weights = np.array([0.5,0.3,0.2])
print("W : ", weights)


#step 3  : Define bias ie b
#       b
bias = 1.0

# step 4 : Calculate weighted sum ie Z.
# z = x1w1 + x2w2 + x3w3 + b
# z = (2.0*0.5) + (3.0*0.3) + (4.0*0.2) + 1.0

z = np.dot(input,weights) + bias
print("z : ", z)

#step 5 : Activation function (ReLU)
def ReLU(X):
    return max(0,X)

#step 6 : Final output

Y = ReLU(z)
print("Y :", Y)


#ANN , FNN all chain till gen ai llm

# Dr. gokhale

#activation funcrtions , resume