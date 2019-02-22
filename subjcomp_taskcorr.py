# Correlation in specific components and subjects between reliability and prediction accuracy.

from os.path import join as pjoin
import numpy as np
from ATT.iofunc import iofiles
import cifti
from scipy import stats

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/'
reb_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'reliability_region_LGL_100Parcels.dscalar.nii'))
dist_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'discrimination_region_LGL_100Parcels.dscalar.nii'))
reb_map = reb_map[:,np.newaxis,:]
dist_map = dist_map[:,np.newaxis,:]
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
masklabel = np.unique(mask[mask!=0])

rb_mat = np.zeros((47, len(masklabel)))
dist_mat = np.zeros((47, len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(47):
        rb_mat[j,i] = reb_map[j, (mask==masklbl)][0]
        dist_mat[j,i] = dist_map[j, (mask==masklbl)][0]
task_idx = np.array([2,5,8,17,18,19,20,21,24,27,38])
rb_mat = rb_mat[task_idx, :]
dist_mat = dist_mat[task_idx, :]

comp = ['15', '50', '100']
taskname = ['emotion', 'gambling', 'language', 'LF', 'LH', 'RF', 'RH', 'tongue', 'relation', 'tom', 'wm']
subjnum = [30,50,70,90,110,130,150,200,300,500,841]

predacc = np.zeros((len(subjnum), len(comp), len(taskname), len(masklabel), 100))
discrimination = np.zeros((len(subjnum), len(comp), len(taskname), len(masklabel)))
for i, tn in enumerate(taskname):
    for j, cp in enumerate(comp):
        iopkl_predacc = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'predacc', 'predacc', 'predacc_'+tn+'_'+cp+'comp.pkl'))
        iopkl_discrimination = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'predacc', 'discrimination', 'discrimination_'+tn+'_'+cp+'comp.pkl'))
        predacc[:,j,i,...] = iopkl_predacc.load()
        discrimination[:,j,i,...] = iopkl_discrimination.load()
mean_predacc = np.mean(predacc,axis=-1)
mean_predacc_flatten = mean_predacc.reshape(len(subjnum), len(comp), len(taskname)*len(masklabel))
rb_mat_flatten = rb_mat.reshape(len(taskname)*len(masklabel),)

rcorr_rel = np.zeros((len(subjnum), len(comp)))
rcorr_dist = np.zeros((len(subjnum), len(comp)))
for i, subj in enumerate(subjnum):
    for j, cp in enumerate(comp):
        rtmp, _ = stats.pearsonr(mean_predacc_flatten[i,j], rb_mat_flatten)
        rcorr_rel[i,j] = rtmp
        rtmp, _ = stats.pearsonr(discrimination[i,j,...].flatten(), rb_mat_flatten)
        rcorr_dist[i,j] = rtmp




