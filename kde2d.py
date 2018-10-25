import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#construct a Gaussian distribtion of temp, imbalance
nsamp=1000
dT = np.random.normal(loc=0.91,scale=0.11/1.96,size=nsamp)
dQ = np.random.normal(0.61,0.14/1.96,size=nsamp)
#This is probably wrong for the forcing, but it's what was in Pier's code in July (need to update)
dF2xCO2_dist =np.random.normal(3.8,0.74/1.96,size=nsamp)
dF = dF2xCO2_dist - np.random.normal(1.46,0.77/1.96,size=nsamp) 
values = np.zeros((3,nsamp))
values[2]=dT
values[0]=dF
values[1]=dQ
#Calculate a KDE from samples
kde = stats.gaussian_kde(values)
#evaluate it at the samples for plotting
density = kde(values)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#x = dF, y=dQ, z=dT
ax.scatter(dF, dQ, dT, c=density)
ax.set_xlabel(r'$\Delta$F')
ax.set_ylabel(r'$\Delta$Q')
ax.set_zlabel(r'$\Delta$T')




likelihood = []
#These aren't samples; they just define a plane in xyz space
Fs = np.linspace(np.min(dF),np.max(dF),nsamp)
Qs = np.linspace(.3,.9,nsamp)
Ses = np.arange(.1,10,.1)
for Shist in Ses:
    #the plane is given by
    Ts = Shist * (Fs-Qs)/dF2xCO2_dist
    #now evaluate the kernel density on the plane using kde
    Svals=np.vstack((Fs,Qs,Ts))
    likelihood += [np.sum(kde(Svals))/float(nsamp)]
                       
    
plt.figure()
plt.plot(Ses,likelihood)

def plot_plane(Shist,ax=plt.gca()):
    Fs = np.linspace(np.min(dF),np.max(dF),nsamp)
    Qs = np.linspace(.3,.9,nsamp)
    xx,yy=np.meshgrid(Fs,Qs)
    zz = Shist/3.8*(xx-yy)
    ax.plot_surface(xx,yy,zz,alpha=.3,label="Shist = "+str(Shist))
