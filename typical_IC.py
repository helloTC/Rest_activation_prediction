# Attempt to generate typical ICs

import numpy as np
from os.path import join as pjoin
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from ATT.iofunc import iofiles
import cifti

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
ncomp = '100'
netmat_all = np.loadtxt(pjoin(parpath, 'node_timeseries', 'netmats', '3T_HCP1200_MSMAll_d'+ncomp+'_ts2', 'netmats1.txt'))
netmat = np.mean(netmat_all,axis=0).reshape(int(ncomp), int(ncomp))

Z = linkage(netmat, 'ward')
cluster_type = fcluster(Z,140,criterion='distance')

icmap, header = cifti.read(pjoin(parpath, 'rest_comp', 'normalize', 'melodic_IC_d'+ncomp+'.dscalar.nii'))

cluster_ic = np.zeros((len(np.unique(cluster_type)), icmap.shape[-1]))
for i, cluster in enumerate(np.sort(np.unique(cluster_type))):
    idx = np.where(cluster_type==cluster)[0]
    tmpic = np.mean(icmap[idx, :],axis=0)
    cluster_ic[i, :] = tmpic
ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'test_data', 'hiecluster_comp'+ncomp+'.dscalar.nii'))
ioscalar.save_from_existed_header(header, cluster_ic)







