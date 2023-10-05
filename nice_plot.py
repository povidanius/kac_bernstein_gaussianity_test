import random
import matplotlib.pyplot as plt

# create sample data
N = 8
data_1 = {
    'x': list(range(N)),
    'y': [10. + random.random() for dummy in range(N)],
    'yerr': [.25 + random.random() for dummy in range(N)]}
data_2 = {
    'x': list(range(N)),
    'y': [10.25 + .5 * random.random() for dummy in range(N)],
    'yerr': [.5 * random.random() for dummy in range(N)]}

# plot
plt.figure()
# only errorbar
plt.subplot(211)
for data in [data_1, data_2]:
    plt.errorbar(**data, fmt='o')
# errorbar + fill_between
plt.subplot(212)
for data in [data_1, data_2]:
    plt.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25)

plt.show()    