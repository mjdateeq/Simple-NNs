
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
   return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
   return sigmoid(x)*(1 - sigmoid(x))



def forward(x, w1, w2, predict=False):
   a1 = np.matmul(x, w1) #hidden layer activation =XW(1)
   z1 = sigmoid (a1) # hidden layer output

   bias = np.ones((len(z1), 1))
   z1 = np.concatenate((bias, z1), axis=1) ## add bias to hidden layer output

   a2 = np.matmul(z1, w2) # output layer activation
   z2 = sigmoid (a2) # predicted output
   if predict:
      return z2
   return a1, z1, a2, z2

def backprop(a2, z0, z1, z2, y):
   delta2 = z2 - y   #(Error = expected output - predicted output)
   Delta2 = np.matmul(z1.T, delta2) # Error * hidden layer output
   delta1 = (delta2.dot(w2[1:,:].T))*sigmoid_deriv(a2) #d hidden layer
   Delta1 = np.matmul(z0.T, delta1)
   return delta2, Delta1, Delta2


#input and output
X=np.array ([[1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]])
y=np.array ([[0],[1],[1],[1]])

# initial weights
w1 = np.random.randn(3, 3) 
w2 = np.random.randn(4,1)  


lr = 0.001

costs = []

epochs = 30000
m=len(X)



for i in range (epochs):
   a1, z1, a2, z2 = forward (X, w1, w2)
   delta2, Delta1, Delta2 = backprop(a2, X, z1, z2, y)


   w1 -= lr*Delta1
   w2 -= lr*Delta2

   c = np.mean(np.abs(delta2))
   costs.append(c)

#   if i % 100 == 0:
#     print('iteration: {} Error {}'.format (i, c))
#     print('Error: {}'.format (c))



print ("Training Complete.")

z3=forward (X, w1, w2, True)


print ("Precentage: ")
print (z3)
print ("Prediction: ")
print (np.round(z3))

print ("Weights: ")
print (w1)

print ("Weights: ")
print (w2)

print ("Costs: ")
print (c)

plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Error')
#plt.savefig('plotxor.jpg')
plt.show()
