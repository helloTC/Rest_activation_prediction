# Cluster components using hierarchical clustering method
# Method followed Smith et al., 2013, TICS.
# The correlation matrix is the group-averaged netmats, using 'full' normalized temporal correlation between every node timeseries and every other.

import numpy as np
from os.path import join as pjoin
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import cifti

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
ncomp = '100'
taskname = 'relation'
netmat_all = np.loadtxt(pjoin(parpath, 'node_timeseries', 'netmats', '3T_HCP1200_MSMAll_d'+ncomp+'_ts2', 'netmats1.txt'))
netmat = np.mean(netmat_all,axis=0).reshape(int(ncomp), int(ncomp))

Z = linkage(netmat, 'ward')
m = dendrogram(Z, labels=np.arange(1,int(ncomp)+1))
labelnum = m['leaves']
plt.close()


# Load mask
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
masklabel = np.unique(mask[mask!=0])

# Load beta data
beta_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_avgbeta', ncomp+'comp', 'avgbeta_'+taskname+'_'+ncomp+'comp.dscalar.nii'))
beta_map = beta_map[:,np.newaxis,:]

beta_mat = np.zeros((int(ncomp), len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(int(ncomp)):
        beta_mat[j,i] = beta_map[j, (mask==masklbl)][0]

beta_mat_compsort = beta_mat[labelnum, :]

# Reliability map
reb_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'reliability_region_LGL_100Parcels.dscalar.nii'))
reb_map = reb_map[:,np.newaxis,:]

rb_mat = np.zeros((47, len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(47):
        rb_mat[j,i] = reb_map[j, (mask==masklbl)][0]
task_idx = np.array([2,5,8,17,18,19,20,21,24,27,38])
rb_mat = rb_mat[task_idx,:]

# Sort regions by reliability values
rgsort = np.argsort(rb_mat[-3,:])[::-1]
beta_mat_rgsort = beta_mat[:,rgsort]

beta_mat_sort = beta_mat_rgsort[labelnum,:]
