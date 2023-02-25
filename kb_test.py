import dcor
import numpy as np
import matplotlib.pyplot as plt
from HSIC import hsic_gam
from pingouin import multivariate_normality
from sklearn.metrics import confusion_matrix
import seaborn


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
        data1 = np.random.permutation(data)

        #breakpoint()

        x = data1[:n_samples,...]
        y = data1[n_samples:,...]
        #result0 = dcor.independence.distance_correlation_t_test(x, y)
        #result = dcor.independence.distance_correlation_t_test(x-y,x+y)
        #testStat, thresh = hsic_gam(x, y, 0.05)
        testStat1, thresh1 = hsic_gam(x-y, x+y, 0.05)
        delta = testStat1 - thresh1

        if delta >= 0:
                #print("Non-gaussian")
                out.normal = False
                out.delta = delta
                return out
    
    #print("Gaussian")
    out.normal = True
    out.delta = delta
    return out


   
def test(data):
   rez = [] 
   output0 = kb_test(data) 
   output1 = multivariate_normality(data, alpha=.05)
   print("G: {} {}".format(output0.normal, output1.normal))
   return output0, output1


def monte_carlo_iteration(n_samples = 2000, dim = 100, gaussian_data = True):
    rez = []

    A = np.random.rand(dim,dim)
    cov = np.dot(A, A.transpose())
    mean = np.zeros(dim)

    #rez.append(test(np.random.multivariate_normal(mu, C, 2*n_samples)))
    #rez.append(test(np.random.normal(0, 1, size=(2*n_samples, dim))))
    #
    # ez.append(test(np.random.uniform(0, 1, size=(2*n_samples, dim))))
    #rez.append(test(np.random.laplace(0, 1, size=(2*n_samples, dim))))
    #rez.append(test(np.random.gumbel(0, 1, size=(2*n_samples, dim))))
    #rez.append(test(np.random.lognormal(0, 1, size=(2*n_samples, dim))))
    #rez.append(test(np.random.beta(5, 5, size=(2*n_samples, dim))))

    if gaussian_data == True:
        rez.append(test(np.random.multivariate_normal(mean, cov, 2*n_samples)))
        rez.append(test(np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 2*n_samples)))
        rez.append(test(np.random.multivariate_normal(2*np.random.uniform(size=dim)-1.0, np.eye(dim), 2*n_samples)))
        rez.append(test(np.random.multivariate_normal(2*np.random.uniform(size=dim)-1.0, cov, 2*n_samples)))
        gt = np.ones(len(rez))

    # wishart
    if gaussian_data == False:
        rez.append(test(np.random.chisquare(1, size=(2*n_samples, dim))))
        rez.append(test(np.random.chisquare(2, size=(2*n_samples, dim))))
        rez.append(test(np.random.uniform(0, 1, size=(2*n_samples, dim)))) #-1,1
        rez.append(test(np.random.laplace(0, 1, size=(2*n_samples, dim))))
        rez.append(test(np.random.logistic(loc=0.0, scale=1.0, size=(2*n_samples, dim)))) #
        rez.append(test(np.random.logistic(loc=0.0, scale=2.0, size=(2*n_samples, dim)))) #
        rez.append(test(np.random.power(2, size=(2*n_samples, dim))))
        rez.append(test(np.random.beta(5,5, size=(2*n_samples, dim)))) #
        rez.append(test(np.random.beta(8,2, size=(2*n_samples, dim))))
        rez.append(test(np.random.beta(2,8, size=(2*n_samples, dim))))
        rez.append(test(np.random.standard_cauchy(size=(2*n_samples, dim))))    
        #rez.append(test(0.1*np.random.multivariate_normal(np.zeros(dim), C, 2*n_samples) + 0.7*np.random.multivariate_normal(1.0+np.zeros(dim), np.eye(dim,dim), 2*n_samples) ))
        gt = np.zeros(len(rez))


    
    #gt[0] = True
    #gt[1] = True
    #gt[2] = True
    #gt[3] = True

    stats = []

    for x in rez:
        stats.append([x[0].normal, x[1].normal])

    stats = np.array(stats)
    #breakpoint()

    #cm0 = confusion_matrix(gt, stats[:,0])
    #cm1 = confusion_matrix(gt, stats[:,0])

    num_gaussian = np.count_nonzero(gt == True)
    num_nongaussian = np.count_nonzero(gt == False)

    num_correctly_rejected = 0
    num_correctly_accepted = 0
    
    #print("{} {}".format(num_gaussian, num_nongaussian))


    for i in range(len(gt)):
        if gt[i] == False and stats[i,0] == False:
            num_correctly_rejected += 1

        if gt[i] == True and stats[i,0] == True:
            num_correctly_accepted += 1
    
    p11, p12, p21, p22 = 0.0,0.0,0.0,0.0

    if num_nongaussian > 0:
        p11 = num_correctly_rejected/num_nongaussian

    if num_gaussian > 0:
        p12 = num_correctly_accepted/num_gaussian


    num_correctly_rejected = 0
    num_correctly_accepted = 0


    for i in range(len(gt)):
        if gt[i] == False and stats[i,1] == False:
            num_correctly_rejected += 1

        if gt[i] == True and stats[i,1] == True:
            num_correctly_accepted += 1
    
    if num_nongaussian > 0:
        p21 = num_correctly_rejected/num_nongaussian

    if num_gaussian > 0:
        p22 = num_correctly_accepted/num_gaussian

    return np.array([p11,p12,p21,p22])


if __name__ == "__main__":
    num_monte_carlo = 30

    dim = 100
    num_samples = 2000
    gaussian_data = True
    dims = []
    rezults = []
    for dim in range(50, 160, 10):
        print('Analysing dimension {}'.format(dim))
        dims.append(dim)
        rez = []
        for i in range(num_monte_carlo):
            rez.append(monte_carlo_iteration(num_samples,dim, gaussian_data))
            print("---")
        rez = np.array(rez)
        if gaussian_data == True:
            mean_rez = np.mean(rez,axis=0)[[1,3]]
            std_rez = np.std(rez,axis=0)[[1,3]]
        else:
            mean_rez = np.mean(rez,axis=0)[[0,2]]
            std_rez = np.std(rez,axis=0)[[0,2]]

        rezults.append(rez)
        print(mean_rez)

    np.save('./dims_normal.npy', dims)
    np.save('./rezults_normal.npy', rezults)
    """
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(rez, showfliers=False)
    #seaborn.violinplot(rez)
    plt.show()
    """

breakpoint()  

