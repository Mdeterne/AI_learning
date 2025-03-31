import numpy as np
import matplotlib.pyplot as plt

dataset = {f"experience{i}": np.random.randn(100)for i in range(4)}
dataset2 = {f"experience{i}": np.random.randn(100,3)for i in range(5)}

plt.figure()
i = []

for j in range(1,100):
    i.append(j)
    
# affichage basique    
plt.subplot(4,1,1)
plt.plot(i,dataset["experience0"][i], lw=1, c="red")
plt.title("experience 1")

plt.subplot(4,1,2)
plt.plot(i,dataset["experience1"][i], lw=1, c="red")
plt.title("experience 2")

plt.subplot(4,1,3)
plt.plot(i,dataset["experience2"][i], lw=1, c="red")
plt.title("experience 3")

plt.subplot(4,1,4)
plt.plot(i,dataset["experience3"][i], lw=1, c="red")
plt.title("experience 4")

plt.show()
    
# affichage orienté objet
fig, ax = plt.subplots(4,1, sharex=True)
ax[0].plot(i,dataset["experience0"][i], lw=1, c="red")
ax[0].set_title("experience 1")
    
ax[1].plot(i,dataset["experience1"][i], lw=1, c="red")
ax[1].set_title("experience 2")
    
ax[2].plot(i,dataset["experience2"][i], lw=1, c="red")
ax[2].set_title("experience 3")
    
ax[3].plot(i,dataset["experience3"][i], lw=1, c="red")
ax[3].set_title("experience 4")
plt.show()
    
    
# fonction flexible
def graphique(data):
    n = len(data)
    plt.figure(figsize=(15,12))
    
    for k, i in zip(data.keys(), range(1, n+1)):
        plt.subplot(n, 1, i)
        plt.plot(data[k])
        plt.title(k)
        
    plt.show()
    
graphique(dataset)
graphique(dataset2)