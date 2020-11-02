import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('Salut, monde!')

array = np.random.rand(1000, 2)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(array[:,0], array[:,1], '+')
plt.title('Tout va bien, vous pouvez fermer cette fenêtre.')

plt.tight_layout()
plt.show()