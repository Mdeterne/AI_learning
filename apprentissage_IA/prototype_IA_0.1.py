import numpy as np
import matplotlib.pyplot as plt

x_input = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1]),dtype= float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype = float) # 1-rouge 2-bleu

x_input = x_input/np.amax(x_input, axis=0)

X = np.split(x_input,[8])[0]
x_prediction = np.split(x_input,[8])[1]

class Neural_Network():
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.w1 = np.random.randn(self.inputSize, self.hiddenSize) # Matrice 2x3
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) # Matrice 3x1

    def forward(self,X):

        self.z = np.dot(X,self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.w2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self,s):
        return s * (1-s)

    def backward(self,X,y,o):

        self.o_error = y-o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)
        return self.w1, self.w2

    def train(self,X,y):

        o = self.forward(X)
        w1,w2 = self.backward(X,y,o)
        return w1,w2

    def predict(self):
        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(x_prediction))
        print("Sortie : \n" + str(self.forward(x_prediction)))

        if(self.forward(x_prediction) < 0.5):
            print("La fleur est BLEU ! \n")
        else:
            print("La fleur est ROUGE ! \n")

NN = Neural_Network()

for i in range(30000):
    #print("#" + str(i) + "\n")
    #print("Valeurs d'entrées: \n" + str(X))
    #print("Sortie actuelle: \n" + str(y))
    #print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    #print("\n")
    w1,w2 = NN.train(X,y)
p = [[0.5,0.5],[1,0.75],[1.5,1],[2,1.25],[2.5,1.5],[3,1.75],[3.5,2],[4,2.25]]
p = p/np.amax(p, axis=0)
p_pred = []

for i in range(8):
    p_pred.append(NN.forward(p[i]))

plt.figure(figsize=(12,8))
plt.scatter(x_input[:,0],x_input[:,1], c=y)
plt.plot(p_pred,p, c = "red")
plt.show()
print("affichage p\n")
print(p)
print("affichage p_pred\n")
print(p_pred)

NN.predict()