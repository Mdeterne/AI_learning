import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataset = {f"experience{i}": np.random.randn(100)for i in range(4)}

    plt.figure()
    i = []

    for j in range(1,100):
        i.append(j)
        
    plt.subplot(4,1,1, sharex=True)
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