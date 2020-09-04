import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import filtfilt
import scipy.io as spio
import numpy as np
from numpy import diff 

def returnRpos(L):
    maxe = L.max()
    a = []
    d = []
    v = []
    g = []
    u = []
    xdiff = diff(np.array(L))
    for i in range(0,len(L)):
        if L[i] > 0.35 * maxe and abs(xdiff[i]) > 0.1:
            a.append(i)
    for i in range(0,len(a)-1):
        if a[i+1] - a[i] < 20:
            d.append(a[i])
            v.append(L[a[i]])
        else:
            d = np.array(d)    
            v = np.array(v)    
            for k in range(0,len(d)):
                if L[d[k]] == v.max() :
                    g.append(d[k])
                    u.append(L[d[k]])
            d = []
            v = []
    return[g,u]
spf1 =spio.loadmat('data_ecg_noisy.mat')
spf1 = np.array(spf1['ecg_noisy'][0])

spf =np.loadtxt('ecg.dat')
#filtering 1st transfer funtion
num = np.array([1,0,0,0,0,0,-1])                                                  
den = np.array([1,-1,0,0,0,0,0])
x1 = signal.lfilter(num,den,spf1)
x1 = x1/32
#filtering 2nd transfer funtion
num1 = np.array([1] + list(np.zeros(30))+ [-1]) 
de1 = np.array([1,-1])
x2 = signal.lfilter(num,den,x1)
xdiff = diff(x2)
max1 = np.max(x2)
mean = np.mean(x2)
x3 = np.arange(len(x2))
print((returnRpos(x2)))
x = np.array(returnRpos(x2)[0])
y = np.array(returnRpos(x2)[1])
#plt.subplot(3,1,1)
plt.plot(x2)
plt.scatter(x,y,s = 6 , color = 'black')
#plt.subplot(3,1,2)
#plt.plot(xdiff)
plt.show()
#print(xdiff[227],xdiff[273])
