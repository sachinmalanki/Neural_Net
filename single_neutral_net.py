
print("Single Neural Netowrk")
import numpy as np
'''1x4 input'''
x=np.array([[1,0,1],[0,0,0],[1,1,1],[1,1,0]])
'''4x1 outputs'''
y=np.array([[1,0,1,0]]).T

weights = np.random.random((3,1))

for i in range(1000): #epochs
    z=np.dot(x,weights)
    sigmoid = 1/(1+np.exp(-z))
    error =(y-sigmoid)
    sig_der=sigmoid*(1-sigmoid)
    weights+=np.dot(x.T,error*sig_der)
    print("Weights",weights)

print("Considering new situation [0,1,1]")
newZ=np.dot(np.array([1,0,0]),weights)
activationout=1/(1+np.exp(-newZ))
print(activationout)
#print(np.array([1,1,1]))