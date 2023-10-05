import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

def nice_plot(data_1, data_2, fname = ''):
    # plot
    plt.figure()
    # only errorbar
    #plt.subplot(211)
    #for data in [data_1, data_2]:
    #    plt.errorbar(**data, fmt='o')

    # errorbar + fill_between
    #plt.subplot(212)
    for data in [data_1, data_2]:
        plt.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
        data = {
            'x': data['x'],
            'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
            'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
        plt.fill_between(**data, alpha=.25)
    if len(fname) > 0:
        plt.savefig('./figures/{}.png'.format(fname))
    plt.show()    


distributions = ['chisquare1', 'chisquare2', 'uniform01', 'uniform-11', 'laplace', 'logistic01', 'logistic02', 'power2', 'beta82', 'beta55', 'beta28', 'cauchy', 'mixture']
#uniform01
#uniform-11
#
distributions = [distributions[0]]
#distributions = ['g0','g1','g2','g3']
#distributions = ['g3']

dims = range(50,160,10)
num_samples = 1000

rez = []
for distribution in distributions:
    #file_name = './rez_g/rezults_{}_{}.npy'.format(distribution, num_samples)
    file_name = './rez/rezults_{}_{}.npy'.format(distribution, num_samples)
    x = np.load(file_name)
    if rez == []:
        rez = x
    else:
        rez = np.vstack((rez, x))   

#breakpoint()
rezults_kb1 = []
rezults_hz1 = []
stds_kb1 = []
stds_hz1 = []
for dim in dims:
    ind = np.where(rez[:,5] == dim)
    mn = np.mean(rez[ind,...][0],axis=0)
    std = np.std(rez[ind,...][0],axis=0)   

    rezults_kb1.append(mn[0])
    rezults_hz1.append(mn[2])
    stds_kb1.append(std[0])
    stds_hz1.append(std[2])

"""
fig, ax = plt.subplots()

trans1 = Affine2D().translate(-0.3, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.3, 0.0) + ax.transData
er1 = ax.errorbar(dims, rezults_kb1, yerr=stds_kb1, marker="o", linestyle="--", color='b', transform=trans1)
er2 = ax.errorbar(dims, rezults_hz1, yerr=stds_hz1, marker="o", linestyle="--", color='r', transform=trans2)

plt.show()
"""

data_1 = {
    'x': list(dims),
    'y': rezults_kb1,
    'yerr': stds_kb1}
data_2 = {
    'x': list(dims),
    'y': rezults_hz1,
    'yerr': stds_hz1}

nice_plot(data_1,data_2, 'individual_nongaussian/{}_{}'.format(distributions[0], num_samples))
