import numpy as np
import math

#-------------------------

# step 1 : Input layer 

#------------------------


x1 = 2.0
x2 = 3.0

print("step 1 : Input layer ")
print("input features : X(This are the features)")
print(f"X1 = {x1}")
print(f"x2 = {x2}")




#-------------------------

# step 2 : Hidden layer 

#------------------------


print("step 2 : Hidden layer (2 neurons)")

print("Hidden neuron 1")

W11 = 0.5
w12 = -0.2
b1 = 0.1


print("weights : ")
print(f"w11 : {W11}")
print(f"w12 : {w12}")

print("Bias :  ")
print(f"b1 : {b1}")

print("Weighted sum ")
print("z1 = (x1*w11 + x2 * w12) + b1")

z1 = (x1 * W11) + (x2 * w12) + b1

print(" Weighted sum : z1 ", z1)

h1 = max(0 , z1)

print("Output of hidden neuron : ", h1)


#########################  neuron 2   ####################33


print("Hidden neuron 2")

W21 = 0.8
w22 = 0.4
b2 = -0.1


print("weights : ")
print(f"w21 : {W21}")
print(f"w22 : {w22}")

print("Bias :  ")
print(f"b2 : {b2}")

print("Weighted sum ")
print("z2 = (x1*w21 + x2 * w22) + b2")

z2 = (x1 * W21) + (x2 * w22) + b2

print(" Weighted sum : z2 ", z2)

h2 = max(0 , z2)

print("Output of hidden neuron 2 : ", h2)

 
 
 #-------------------------

# step 3 : Output layer 

#------------------------





w_out1 = 1.0
w_out2 = -1.5
bias_out = 0.2

print("step 3 : Output layer")

print("Weights :")

print(f"w_out 1 : {w_out1}")
print(f"w_out 2 : {w_out2}")


print("Bias : ")
print(f"Bias_Out  : {bias_out}")

z_out =  h1 * w_out1 + h2 * w_out2 + bias_out

print("Weighted sum : ", z_out)

#sigmoid

z = 1 / (1 + math.exp(-z_out))

print("_______________________________")
print("___Neural network summary____")
print("____________________________________")

print("Input layer")

print(f"x1 : {x1}")
print(f"x2 : {x2}")

print("Hidden layer")

print(f"h1 : {h1}")

print(f"h2 : {h2}")

print("Output layer")
print(f"z : {z}")

print(" Predictions of Neural network")

if(z >= 0.5):
    print("Predicted as POSITIVE CLASS")
    
else:
    print("Predicted as NEGATIVE CLASS")
    