import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter,filtfilt
import scipy.io as spio
import numpy as np
from numpy import diff 
from readmit import b
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def index(L,v):
    b =[]
    for i in range(0,len(L)):
        if L[i]>v:
            b.append(i)
    return(b)
def maxvalinlist(L):
    b=[]
    c=[]
    for i in range(0,len(L)-1):
        if L[i+1]-L[i]<10  :
            b.append(L[i])
        else:
            c.append(b)
            b=[]
    return(c)
def imax(L,x2):
    a =[]
    b =[]
    if len(L)!=0:
        for i in L:
            a.append(x2[i])
        for j in L:
            if max(a)==x2[j]:
                b.append(j)
        return(b[0])

def removee(L):
    for i in L:
        if i ==[]:
            L.remove(i)
    return(L)

spf1 =np.loadtxt('ecg.dat')

spf =spio.loadmat('data_ecg_noisy.mat')
spf = np.array(spf['ecg_noisy'][0])
#filtering 1st transfer funtion
spf = butter_highpass_filter(spf,0.5, 360, order=5)
num = np.array([1,0,0,0,0,0,-1])                                                  
den = np.array([1,-1,0,0,0,0,0])
x1 = signal.lfilter(num,den,spf)
x1 = x1/32
#filtering 2nd transfer funtion
num1 = np.array([1] + list(np.zeros(30))+ [-1]) 
de1 = np.array([1,-1])
x2 = signal.lfilter(num,den,x1)
#plt.subplot(5,1,3)
plt.plot(x2)
max1 = np.max(x2[0:len(x2)-100])
mean = np.mean(x2[0:len(x2)-100])
#thresholding the ecg value with respect to the mean 
for i in range(0,len(x2)):
    if x2[i] < 0.45 * max1:
        x2[i]= 2*mean
    else:
        x2[i] = x2[i]
dx = 0.001
x4 = diff(x2)
x4 =x4**2
x5 =np.zeros(len(x4))
"""
Thresholding part
45% threshold

"""
thres = np.ones(len(x2))
thres *= 0.45*max1
plt.plot(thres,label = 'Thresholding')
#plt.subplot(5,1,1)
#plt.plot(spf[0:1000])
#plt.plot(x4[0:1000])
#plt.subplot(5,1,2)
#plt.plot(x1[0:1000])
#plt.subplot(4,1,4)
#plt.plot(x5[0:1000])
#plt.subplot(5,1,4)
#plt.plot(x2[0:1000])
#print(np.max(x2))
mider = index(list(x2),0.6)
#print(mider)
ab = maxvalinlist(mider)
ab = removee(ab)
#print(ab)
aka = []
for i in ab:
    cu = imax(i,x2)
    aka.append(cu)
aka = np.array(aka)
kito=[]
for i in aka:
    a = x2[i]
    kito.append(a)
#x = np.array([82,82,82,82,82,82,82])
#y = np.zeros(len(x))
kito = np.array(kito)
#print(kito)
#plt.subplot(5,1,3)
plt.scatter(aka,kito,label = 'R peak',color = 'red',s=6)
#print(type(aka),type(kito))
#print(aka)
c = []
minpoint=[]
minvalue=[]
for i in aka:
    a = []
    d = []
    for j in range(i-80,i):
        a.append(j)
        #print(a)
        d.append(x2[j])
    for k in a:
        if x2[k] == min(d):
            minpoint.append(k)
            minvalue.append(min(d))
            break
minpoint=np.array(minpoint)
minvalue=np.array(minvalue)
            #print(k)
        #print(min(d))
        #print(d)
print(minpoint)
#plt.subplot(5,1,3)
plt.scatter(minpoint,minvalue,color = 'black',alpha =1,s=6, label = 'P start')
minpoint1=[]
minvalue1=[]
for i in aka:
    a = []
    d = []
    for j in range(i-40,i):
        a.append(j)
        #print(a)
        d.append(x2[j])
    for k in a:
        if x2[k] == min(d):
            minpoint1.append(k)
            minvalue1.append(min(d))
            break
minpoint1=np.array(minpoint1)
minvalue1=np.array(minvalue1)
#plt.subplot(5,1,3)
plt.scatter(minpoint1,minvalue1,color = 'green',alpha =1,s=6,label = 'P end')
minpoint2=[]
minvalue2=[]
for i in aka:
    a = []
    d = []
    for j in range(i-20,i):
        a.append(j)
        #print(a)
        d.append(x2[j])
    for k in a:
        if x2[k] == min(d):
            minpoint2.append(k)
            minvalue2.append(min(d))
            break
minpoint2=np.array(minpoint2)
minvalue2=np.array(minvalue2)
#plt.subplot(5,1,3)
#plt.scatter(minpoint2,minvalue2,color = 'blue',alpha =1,s=6)        
x2 += 2 
cak = []
vcak = []
for i in aka:
    a = []
    b = []
    for ju in range(i+9,i+15):
        a.append(ju)
        b.append(x2[ju])
    for iu in a:
        if x2[iu] == min(b):
            cak.append(iu)
            vcak.append(x2[iu])
            break
cak = np.array(cak)
vcak = np.array(vcak)
vcak -= 2 
vcak *= -1*(vcak)

#print(cak)
#    print(b)
print(len(cak))
print(len(minpoint))
#print(aka)
#print(len(minvalue))
#plt.subplot(5,1,3)
plt.scatter(cak,-1*vcak,color = 'purple',alpha =1,s=6,label = 'S wave')     
#plt.legend((kito),("haha"),scatterpoints=1,loc='lower left',ncol=3)
plt.legend()
plt.show()
qrs = []
rr = []
pr = []
for i in range(0,len(cak)-1):
    qrs.append(cak[i]-minpoint2[i])
    rr.append(aka[i+1]-aka[i])
    pr.append(minpoint1[i]-minpoint[i])

print(min(rr))
qrs1 = (np.array(qrs).mean())
rr1 = (np.array(rr).mean())
pr1 = (np.array(pr).mean())

print("qrs:",(1000*qrs1/360),"mS")
print("rr:",(1000*rr1/360),"mS")
print("pr:",(1000*pr1/360),"mS")
print(np.array(rr).std())
