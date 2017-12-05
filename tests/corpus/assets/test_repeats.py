import operator
import numpy as np
import matplotlib.pyplot as plt

srf1 = np.load("/Volumes/projects/kws/data/repetition/sample/srf1_posteriors.npy")
srf3 = np.load("/Volumes/projects/kws/data/repetition/sample/srf3_posteriors.npy")

for idx, val in enumerate(srf1):
     x,i = max(enumerate(val),key=operator.itemgetter(1))
     y,j = max(enumerate(srf3[idx]),key=operator.itemgetter(1))
     print(x,i,y,j)



data = srf1

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])

plt.show()