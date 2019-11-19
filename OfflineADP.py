# Copyright 2018, Xiaowei Gu and Plamen P. Angelov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.






# Offline Autonomous Data Partitioning (ADP) algorithm with changeable distance type and granularity
# References:
# [1] X. Gu, P. Angelov, J. Principe, A method for autonomous data partitioning, Information Sciences, vol.460–461, pp. 65-82, 2018.
# [2] X. Gu, P. Angelov, Self-organising fuzzy logic classifier, Information Sciences, vol.447, pp. 36-51, 2018.
# Programmed by Xiaowei Gu

import numpy
import scipy.io
import scipy.spatial.distance
import numpy.matlib

distancetype='chebyshev' #changeable, i.e. 'euclidean','cityblock','sqeuclidean','cosine'.
granularity=11 #changeable, any positive integer. The larger granularity is, the more details the partitioning result gives.

def ADP(data):
    data=numpy.float32(numpy.matrix(data))
    L0,W0=data.shape
    udata,frequency = numpy.unique(data,return_counts=True, axis=0)
    frequency=numpy.matrix(frequency)
    L,W=udata.shape
    dist=(scipy.spatial.distance.cdist(udata,udata,metric=distancetype))**2
    unidata_pi=numpy.sum(numpy.multiply(dist,numpy.matlib.repmat(frequency,L,1)), axis=1)
    unidata_density=numpy.transpose(unidata_pi)*numpy.transpose(frequency)/(unidata_pi*2*L0)
    unidata_Gdensity=numpy.multiply(unidata_density,numpy.transpose(frequency))

    pos=numpy.zeros(L0)
    pos[0]=int(numpy.argmax(unidata_Gdensity))
    seq=numpy.array(range(0,L0))
    seq=numpy.delete(seq,pos[0])

    for ii in range(1,L):
        p1=numpy.argmin(dist[int(pos[ii-1]),seq])
        pos[ii]=seq[p1]
        seq=numpy.delete(seq,p1)

    udata2=numpy.zeros([L,W])
    uGD=numpy.zeros(L)
    for ii in range(0,L):
        udata2[ii,:]=udata[int(pos[ii]),:]
        uGD[ii]=unidata_Gdensity[int(pos[ii]),0]

    uGD1=uGD[range(0,L-2)]-uGD[range(1,L-1)]
    uGD2=uGD[range(1,L-1)]-uGD[range(2,L)]
    seq2=numpy.array(range(1,L-1))

    seq3=numpy.array([i for i in range(len(uGD2)) if uGD1[i]<0 and uGD2[i]>0])

    seq4=numpy.array([0])
    if uGD2[L-3]<0:
        seq4=numpy.append(seq4,seq2[seq3])
        seq4=numpy.append(seq4,numpy.array([int(L-1)]))
    else:
        seq4=numpy.append(seq4,seq2[seq3])

    L2, =seq4.shape
    centre0=numpy.zeros([L2,W])
    for ii in range(0,L2):
        centre0[ii]=udata2[int(seq4[ii]),:]
    
    dist1=scipy.spatial.distance.cdist(data,centre0,metric=distancetype)
    seq5=dist1.argmin(1)
    centre1=numpy.zeros([L2,W])
    Mnum=numpy.zeros(L2)

    for ii in range(0,L2):
        seq6=[i for i in range(len(seq5)) if seq5[i] == ii]
        Mnum[ii]=len(seq6)
        centre1[ii,:]=numpy.mean(data[seq6,:],axis=0)

    seq7=[i for i in range(len(Mnum)) if Mnum[i] > 1]
    seq8=[i for i in range(len(Mnum)) if Mnum[i] <= 1]

    L3=len(seq7)
    L4=len(seq8)

    centre2=numpy.zeros([L3,W])
    centre3=numpy.zeros([L4,W])
    Mnum1=numpy.zeros(L3)
    for ii in range(0,L3):
        centre2[ii,:]=centre1[seq7[ii],:]
        Mnum1[ii]=Mnum[seq7[ii]]

    for ii in range(0,L4):
        centre3[ii,:]=centre1[seq8[ii],:]

    dist2=scipy.spatial.distance.cdist(centre3,centre2,distancetype)
    seq9=dist2.argmin(1)

    for ii in range(0,L4):
        centre2[seq9[ii],:]=centre2[seq9[ii],:]*Mnum1[seq9[ii]]/(Mnum1[seq9[ii]]+1)+centre3[ii,:]/(Mnum1[seq9[ii]]+1)
        Mnum1[seq9[ii]]=Mnum1[seq9[ii]]+1

    UD2=centre2
    L5=0
    Count=0
    while L5 != L3 and L3>2:
        Count=Count+1
        L5=L3
        dist3=scipy.spatial.distance.cdist(data,UD2,distancetype)
        seq10=dist3.argmin(1)
        centre3=numpy.zeros([L3,W])
        Mnum3=numpy.zeros(L3)
        Sigma3=numpy.zeros(L3)
        seq12=[]
        for ii in range(0,L3):
            seq11=[i for i in range(len(seq10)) if seq10[i] == ii]
            if len(seq11)>=2:
                data1=data[seq11,:]
                Mnum3[ii]=len(seq11)
                centre3[ii,:]=numpy.sum(data1,axis=0)/Mnum3[ii]
                Sigma3[ii]=numpy.sum(numpy.sum(numpy.multiply(data1,data1)))/Mnum3[ii]-numpy.sum(numpy.multiply(centre3[ii,:],centre3[ii,:]))
                if Sigma3[ii]>0:
                    seq12.append(ii)
        L3=len(seq12)
        Mnum3=numpy.matrix(Mnum3[seq12])
        centre3=centre3[seq12,:]
        dist=(scipy.spatial.distance.cdist(centre3,centre3,distancetype))**2
        unidata_pi=numpy.sum(numpy.multiply(dist,numpy.matlib.repmat(Mnum3,L3,1)), axis=1)
        unidata_density=numpy.transpose(unidata_pi)*numpy.transpose(Mnum3)/(unidata_pi*2*L0)
        unidata_Gdensity=numpy.multiply(unidata_density,numpy.transpose(Mnum3))
        dist2=(scipy.spatial.distance.pdist(centre3,distancetype))
        dist3=scipy.spatial.distance.squareform(dist2)
        Aver1=numpy.mean(dist2)
        for ii in range(granularity):
            Aver1=numpy.mean(dist2[dist2<=Aver1])
        Sigma=Aver1/2
        dist3=dist3-numpy.ones([L3,L3])*Sigma
        seq15=[]
        for i in range(0,L3):
            seq13=numpy.array(list(range(0,i))+list(range(i+1,L3)))
            seq14=seq13[dist3[i,seq13]<0]
            if len(seq14)>0:
                if unidata_Gdensity[i]>max(unidata_Gdensity[seq14]):
                    seq15.append(i)
            else:
                seq15.append(i)
        L3=len(seq15)
        UD2=centre3[numpy.array(seq15),:]

    centre=UD2

    dist1=scipy.spatial.distance.cdist(data,centre,distancetype)
    IDX=dist1.argmin(1)
    Mnum=numpy.zeros(L3)

    for ii in range(0,L3):
        seq6=[i for i in range(len(IDX)) if IDX[i] == ii]
        Mnum[ii]=len(seq6)
        centre[ii,:]=numpy.sum(data[seq6,:],axis=0)/Mnum[ii]
    return centre,IDX



