from OfflineADP import ADP
import os
import matplotlib.pyplot as plt
import numpy
import scipy.io
import scipy.spatial.distance
import numpy.matlib

dirpath = os.getcwd()
dr =dirpath+'\\data1.mat'
mat_contents = scipy.io.loadmat(dr)
data=mat_contents['data']

##
centre,IDX=ADP(data)


for ii in range(0,max(IDX)+1):
    plt.plot(data[IDX==ii,0],data[IDX==ii,1],'o')
    plt.hold(True)
plt.plot(centre[:,0],centre[:,1],'k.')
plt.hold(False)
plt.show()
