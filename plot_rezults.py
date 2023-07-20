import numpy as np
import matplotlib.pyplot as plt


dims = np.load('./rez/dims_normal.npy')
rezults = np.load('./rez/rezults_normal.npy')



rezults_kb = []
rezults_hz = []
stds_kb = []
stds_hz = []
for d,rez in zip(dims, rezults):
    #print("{} {}".format(d, np.mean(rez, axis=0)))
    mn = np.mean(rez, axis=0)    
    std = np.std(rez, axis=0)
    rezults_kb.append(mn[1])
    rezults_hz.append(mn[3])
    stds_kb.append(std[1])
    stds_hz.append(std[3])
    #print("{} {} {}".format(d, mn[0], mn[2]))

dims = np.load('./rez/dims_5000.npy')
rezults = np.load('./rez/rezults_5000.npy')
rezults_kb1 = []
rezults_hz1 = []
stds_kb1 = []
stds_hz1 = []
for d,rez in zip(dims, rezults):
    #print("{} {}".format(d, np.mean(rez, axis=0)))
    mn = np.mean(rez, axis=0)    
    std = np.std(rez, axis=0)
    rezults_kb1.append(mn[0])
    rezults_hz1.append(mn[2])
    stds_kb1.append(std[0])
    stds_hz1.append(std[2])
    #print("{} {} {}".format(d, mn[0], mn[2]))
plt.subplot(1, 2, 1)
plt.errorbar(dims, rezults_kb, stds_kb) #, 'b--')
plt.errorbar(dims, rezults_hz, stds_hz) #, 'r--')
plt.subplot(1, 2, 2)
plt.errorbar(dims, rezults_kb1, stds_kb1) #, 'b--')
plt.errorbar(dims, rezults_hz1, stds_hz1) #, 'r--')
plt.show()

#breakpoint()    
