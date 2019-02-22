from os.path import join as pjoin
import cifti
import numpy as np
from scipy import stats

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
ncomp = '100'
taskname = 'emotion'

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

# Reliability map
reb_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'reliability_region_LGL_100Parcels.dscalar.nii'))
reb_map = reb_map[:,np.newaxis,:]

rb_mat = np.zeros((47, len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(47):
        rb_mat[j,i] = reb_map[j, (mask==masklbl)][0]
task_idx = np.array([2,5,8,17,18,19,20,21,24,27,38])
taskname = np.array(['emotion', 'gambling', 'language', 'LF', 'LH', 'RF', 'RH', 'tongue', 'relation', 'tom', 'wm'])
rb_mat = rb_mat[task_idx,:]

# Correlation between beta values and reliability
r_reb_beta = []
for i in range(int(ncomp)):
    r_tmp, _ = stats.pearsonr(rb_mat[1,:], beta_mat[i,:])
    r_reb_beta.append(r_tmp)
r_reb_beta = np.array(r_reb_beta)

numsort = np.argsort(r_reb_beta)[::-1]
beta_mat_sort = beta_mat[numsort, :]



