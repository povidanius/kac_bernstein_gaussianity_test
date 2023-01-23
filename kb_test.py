import dcor
import numpy as np
import matplotlib.pyplot as plt


def kb_test(data, p_value_threshold = 0.05):
    n_samples = int(len(data)/2)
    x = data[:n_samples]
    y = data[n_samples:]
 

    result0 = dcor.independence.distance_correlation_t_test(x, y)
    result = dcor.independence.distance_correlation_t_test(x-y,x+y)

    if result.p_value > p_value_threshold:
        print("Gaussian ".format(result.p_value))
    else:
        print("Non-gaussian")    



n_samples = 500
p_value_threshold = 0.05


kb_test(np.random.normal(0, 1, size=2*n_samples), p_value_threshold)
kb_test(np.random.uniform(0, 1, size=2*n_samples), p_value_threshold)
kb_test(np.random.laplace(0, 1, size=2*n_samples), p_value_threshold)
kb_test(np.random.gumbel(0, 1, size=2*n_samples), p_value_threshold)
kb_test(np.random.lognormal(0, 1, size=2*n_samples), p_value_threshold)

