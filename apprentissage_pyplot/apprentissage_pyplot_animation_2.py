import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

m = 1 #kg
k = 1 # N/m
d = 0.2 # unit of d

t = np.linspace(0,40,501)
w_d = np.sqrt((4*m*k - d**2)/(4*m**2))
x = np.exp(-d/(2*m) * t) * np.cos(w_d * t)

fig, axis = plt.subplots(1,2)

animated_curv, = axis[1].plot([],[])
animated_str, = axis[0].plot([],[], color = "blue")
animated_mass, = axis[0].plot([],[], 'o',markersize = 20, color = "red")


axis[1].set_xlim([min(t),max(t)])
axis[1].set_ylim([-1,1])
axis[1].grid()

axis[0].set_xlim([-2,2])
axis[0].set_ylim([-2,2])
axis[0].grid()


def update(frame):
    
    animated_curv.set_data(t[:frame],x[:frame])
    animated_mass.set_data([x[frame]],[0])
    animated_str.set_data([-2,x[frame]],[0,0])
    
    animated_str.set_linewidth(int(abs(x[frame]-2)*2))
    
    return animated_curv,animated_mass, animated_str

animation = anim.FuncAnimation(fig = fig, func = update, frames = len(t), interval = 25,blit = True,)
animation.save("animation_poid.gif")
plt.show()