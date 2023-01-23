import dcor
import numpy as np
import matplotlib.pyplot as plt


def kb_test(data):
    n_samples = int(len(data)/2)
    x = data[:n_samples]
    y = data[n_samples:]
 

    result0 = dcor.independence.distance_correlation_t_test(x, y)
    result = dcor.independence.distance_correlation_t_test(x-y,x+y)

    if result.p_value > 0.05:
        print("Gaussian ".format(result.p_value))
    else:
        print("Non-gaussian")    

n_samples = 500


kb_test(np.random.normal(0, 1, size=2*n_samples))
kb_test(np.random.uniform(0, 1, size=2*n_samples))
kb_test(np.random.laplace(0, 1, size=2*n_samples))
