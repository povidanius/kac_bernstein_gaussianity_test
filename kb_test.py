import dcor
import numpy as np
import matplotlib.pyplot as plt
from HSIC import hsic_gam
from pingouin import multivariate_normality
from sklearn.metrics import confusion_matrix

class Output():
    def __init__(self):
        self.type = 'none'
        self.normal = None
        self.delta = 0.0
    
def kb_test(data):
    n = len(data)
    n_samples = int(len(data)/2)
    independence = True
    #data = (data - data.mean())/data.std()
    dim = data.shape[1]
    out = Output()
    out.type = 'kacim'
    n_iter = 3

    for i in range(n_iter):
        data = np.random.permutation(data)

        #breakpoint()

        x = data[:n_samples,...]
        y = data[n_samples:,...]
        #result0 = dcor.independence.distance_correlation_t_test(x, y)
        #result = dcor.independence.distance_correlation_t_test(x-y,x+y)
        #testStat, thresh = hsic_gam(x, y, 0.05)
        testStat1, thresh1 = hsic_gam(x-y, x+y, 0.05)
        delta = testStat1 - thresh1

        if delta >= 0:
                print("Non-gaussian")
                out.normal = False
                out.delta = delta
                return out
    
    print("Gaussian")
    out.normal = True
    out.delta = delta
    return out


   
def test(data):
   rez = [] 
   output0 = kb_test(data) # p_value_threshold = 0.05, n_iter = 2)
   output1 = multivariate_normality(data, alpha=.05)
   return output0, output1



rez = []
n_samples = 1500
p_value_threshold = 0.05
dim = 200


rez.append(test(np.random.normal(0, 1, size=(2*n_samples, dim))))
rez.append(test(np.random.uniform(0, 1, size=(2*n_samples, dim))))
rez.append(test(np.random.laplace(0, 1, size=(2*n_samples, dim))))
rez.append(test(np.random.gumbel(0, 1, size=(2*n_samples, dim))))
rez.append(test(np.random.lognormal(0, 1, size=(2*n_samples, dim))))
gt = np.zeros(len(rez))
gt[0] = True

stats = []

for x in rez:
    stats.append([x[0].normal, x[1].normal])

stats = np.array(stats)

cm0 = confusion_matrix(gt, stats[:,0])
cm1 = confusion_matrix(gt, stats[:,0])

num_gaussian = np.count_nonzero(gt == True)
num_nongaussian = np.count_nonzero(gt == False)

num_correctly_rejected = 0
num_correctly_accepted = 0


for i in range(len(gt)):
    if gt[i] == False and stats[i,0] == False:
        num_correctly_rejected += 1

    if gt[i] == True and stats[i,0] == True:
        num_correctly_accepted += 1

a = num_correctly_rejected/num_nongaussian
b = num_correctly_accepted/num_gaussian

print (a)
print (b)
num_correctly_rejected = 0
num_correctly_accepted = 0


for i in range(len(gt)):
    if gt[i] == False and stats[i,1] == False:
        num_correctly_rejected += 1

    if gt[i] == True and stats[i,1] == True:
        num_correctly_accepted += 1

a = num_correctly_rejected/num_nongaussian
b = num_correctly_accepted/num_gaussian


print (a)
print (b)
#stats[:,0] == True 
#gt[0] == True
breakpoint()  

#o = kb_test(np.random.normal(0, 1, size=(2*n_samples, dim)), p_value_threshold)
#o = kb_test(np.random.uniform(0, 1, size=(2*n_samples, dim)), p_value_threshold)
#o = kb_test(np.random.laplace(0, 1, size=(2*n_samples, dim)), p_value_threshold)
#o = kb_test(np.random.gumbel(0, 1, size=(2*n_samples, dim)), p_value_threshold)
#o = kb_test(np.random.lognormal(0, 1, size=(2*n_samples, dim)),p_value_threshold)


#mean = [100, 0]
#cov = [[1, 0], [0, 1]]  # diagonal covariance
#val, stat, th = kb_test(np.random.multivariate_normal(mean, cov, size=2*n_samples), p_value_threshold)

