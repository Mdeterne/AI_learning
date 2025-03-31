import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()

x = iris.data
y = iris.target
names = list(iris.target_names)

print(f"x contient {x.shape[0]} exemples et {x.shape[1]} varianble")
print(f"il y a {np.unique(y).size} classes")


# affichage des 4 variable via scatter
plt.scatter(x[:,0], x[:,1], c=y, alpha = x[:,2] / max(x[:,2]), s = x[:,3]*100)
plt.xlabel("longueur sépal")
plt.ylabel("largeur sépal")
plt.show()


# affichage 3D du dataset
ax = plt.axes(projection = "3d")
ax.scatter(x[:,0], x[:,1], x[:,2], c=y, s = x[:,3]*100)
plt.show()


# affichage 3D d'une surface
f = lambda x, y: np.sin(x) + np.cos(x+y)
X = np.linspace(0, 5, 100)
Y = np.linspace(0, 5, 100)

X, Y = np.meshgrid(X, Y)
Z = f(X,Y)
ax = plt.axes(projection = "3d")
ax.plot_surface(X, Y, Z, cmap="plasma")
plt.show()

#exercice
n = x.shape[1]
plt.figure(figsize=(17,12))
    
for i in range(n):
    plt.subplot(n//2, n//2, i+1)
    plt.scatter(x[:,0], x[:, i] ,c=y)
    plt.xlabel("0")
    plt.ylabel(i)
    plt.colorbar(ticks=list(np.unique(y)))
    
plt.show()