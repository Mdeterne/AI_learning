import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

def initialisation(X):
    w = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return(w,b)


def model(X, w, b):
    Z = X.dot(w) + b
    A = 1/(1+ np.exp(-Z))
    return(A)


def log_loss(A, y):
    return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(A,X,y):
    dw = 1 / len(y) * np.dot(X.T, A - y)
    db = 1/len(y) * np.sum(A-y)
    return(dw,db)


def update(dw, db , w, b, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return (w,b)

def predict(X, w, b):
    A = model(X,w,b)
    return A >= 0.5

def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):    
    #initialisation
    w,b = initialisation(X)
    
    history = []
    loss = []
    
    for i in range(n_iter):
        A = model(X,w,b)
        loss.append(log_loss(A,y))
        dw, db = gradients(A,X,y)
        w,b = update(dw, db, w, b, learning_rate)
        history.append([w,b,loss,i])
    
    y_pred = predict(X,w,b)
    print(accuracy_score(y,y_pred))
    
    plt.plot(loss)
    plt.show()

    return w,b,loss,history

if __name__ == "__main__":
    
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0],1))    
    
    history = []
    loss = []   
       
    w,b,loss,history = artificial_neuron(X,y)
    
    
    new_plant = np.array([2,1])
    
    x0 = np.linspace(-1, 4, 100)
    x1 = (-w[0] * x0 - b) /w[1]
    
    plt.scatter(X[:,0], X[:,1], c=y, cmap = 'summer')
    plt.scatter(new_plant[0], new_plant[1], c='r')
    plt.plot(x0, x1, c="orange", lw = 3)
    plt.show()
    predict(new_plant,w,b) 
    
    # affichage 3D
    fig = go.Figure(data=[go.Scatter3d( 
        x=X[:, 0].flatten(),
        y=X[:, 1].flatten(),
        z=y.flatten(),
        mode='markers',
        marker=dict(
            size=5,
            color=y.flatten(),                
            colorscale='YlGn',  
            opacity=0.8,
            reversescale=True
        )
    )])

    fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
    fig.layout.scene.camera.projection.type = "orthographic"
    
    X0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    X1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx0, xx1 = np.meshgrid(X0, X1)
    Z = w[0] * xx0 + w[1] * xx1 + b
    A = 1 / (1 + np.exp(-Z))

    fig = (go.Figure(data=[go.Surface(z=A, x=xx0, y=xx1, colorscale='YlGn', opacity = 0.7, reversescale=True)]))

    fig.add_scatter3d(x=X[:, 0].flatten(), y=X[:, 1].flatten(), z=y.flatten(), mode='markers', marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity = 0.9, reversescale=True))


    fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()
    
    
    def animate(params):
        w = params[0]
        b = params[1]
        loss = params[2]
        i = params[3]
        
        ax[0].clear() # frontière de décision
        ax[1].clear() # sigmoide
        ax[2].clear() # fonction cout
        
        s = 300
        # frontière de décision
        ax[0].scatter(X[:, 0], X[:, 1], c=y, s=s, cmap="summer", edgecolors = 'k',linewidths=3)
        
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()
        
        x1 = np.linspace(-3,6,100)
        x2 = (-w[0] * x1 - b)/ w[1]
        ax[0].plot(x1, x2, c="orange", lw=4)
        
        ax[0].set_xlim(X[:,0].min(), X[:,0].max())
        ax[0].set_xlim(X[:,1].min(), X[:,1].max())
        ax[0].set_tittle("Frontière de décision")
        ax[0].set_xlabel("x1")
        ax[0].set_ylabel("x2")
        
        # sigmoide
        z = X.dot(w) + b
        z_new = np.linsp(z.min(), z.max(), 100)
        A = 1/(1 + np.exp(-z_new))
        ax[1].plot(z_new, A, c="orange", lw=4)
        ax[1].scatter(z[y==0], np.zeros(z[y==0].shape), c="#008066", edgecolors='k', linewidths=3, s=s)
        ax[1].scatter(z[y==0], np.ones(z[y==0].shape), c="#ffff66", edgecolors='k', linewidths=3, s=s)
        #ax[1].vlines(x=0, ymin=0, ymax=1, colors="red") # frontière de décision
        ax[1].set_xlim(z.min(), z.max())
        ax[1].set_title("sigmoid")
        ax[1].set_xlable('Z')
        ax[1].set_ylable("A(Z)")
        
        for j in range(len(A[y.flatten()==0])):
            ax[1].vlines(z[y==0][j], ymin=0, ymax = 1/ (1 + np.exp(-z[y==0][j])), color="red", alpha=0.5, zorder=-1)
            
        for j in range(len(A[y.flatten()==1])):
            ax[1].vlines(z[y==1][j], ymax=1, ymin = 1/ (1 + np.exp(-z[y==1][j])), color="red", alpha=0.5, zorder=-1)
            
        ax[2].plot(range(i), loss[:,i], color="red", lw=4)
        ax[2].set_xlim(loss[-1] * 0.8, len(loss))
        ax[2].set_ylim(0,loss[0] * 1.1)
        ax[2].set_title("Fonction Cout")
        ax[2].set_xlabel("itteration")
        ax[2].se_ylabel("loss")
        
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40, 10))
    ani = animation.FuncAnimation(fig, animate, frames=history, interval=200, repeat=False)
    
    Writer = animation.writers["pillow"]
    writer = Writer(fps=15, metadata=dict(artist="Me"))
    ani.save("animation.gif", writer= writer)