# Code to evaluate subject and component effects

from os.path import join as pjoin
import numpy as np
import cifti
from ATT.iofunc import iofiles

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'

reb_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'reliability_region_LGL_100Parcels.dscalar.nii'))
reb_map = reb_map[:,np.newaxis,:]
dis_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'discrimination_region_LGL_100Parcels.dscalar.nii'))
dis_map = dis_map[:,np.newaxis,:]
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
masklabel = np.unique(mask[mask!=0])

rb_mat = np.zeros((47, len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(47):
        rb_mat[j, i] = reb_map[j, (mask==masklbl)][0]

dis_mat = np.zeros((47, len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(47):
        dis_mat[j, i] = dis_map[j, (mask==masklbl)][0]


thr_reb = 0.6

task_idx = np.array([2,5,8,17,18,19,20,21,24,27,38])
rb_mat = rb_mat[task_idx, :]
dis_mat = dis_mat[task_idx, :]
taskname = np.array(['emotion', 'gambling', 'language', 'LF', 'LH', 'RF', 'RH', 'tongue', 'relation', 'tom', 'wm'])


predacc_path_100comp = [pjoin(parpath, 'program', 'framework', 'predacc', 'predacc_LGL', 'predacc_'+tn+'_100comp.pkl') for tn in taskname]
predacc_path_50comp = [pjoin(parpath, 'program', 'framework', 'predacc', 'predacc', 'predacc_LGL'+tn+'_50comp.pkl') for tn in taskname]
predacc_path_15comp = [pjoin(parpath, 'program', 'framework', 'predacc', 'predacc', 'predacc_LGL'+tn+'_15comp.pkl') for tn in taskname]
discrimination_path_100comp = [pjoin(parpath, 'program', 'framework', 'predacc', 'discrimination_LGL', 'discrimination_'+tn+'_100comp.pkl') for tn in taskname]
discrimination_path_50comp = [pjoin(parpath, 'program', 'framework', 'predacc', 'discrimination_LGL', 'discrimination_'+tn+'_50comp.pkl') for tn in taskname]
discrimination_path_15comp = [pjoin(parpath, 'program', 'framework', 'predacc', 'discrimination_LGL', 'discrimination_'+tn+'_15comp.pkl') for tn in taskname]

subjnum = [30,50,70,90,110,130,150,200,300,500,841]
predacc_value_15comp = np.zeros((len(subjnum), len(task_idx), len(masklabel)))
predacc_value_50comp = np.zeros((len(subjnum), len(task_idx), len(masklabel)))
predacc_value_100comp = np.zeros((len(subjnum), len(task_idx), len(masklabel)))
discrimination_value_15comp = np.zeros((len(subjnum), len(task_idx), len(masklabel)))
discrimination_value_50comp = np.zeros((len(subjnum), len(task_idx), len(masklabel)))
discrimination_value_100comp = np.zeros((len(subjnum), len(task_idx), len(masklabel)))

for i, ti in enumerate(task_idx):
    iopkl = iofiles.make_ioinstance(predacc_path_15comp[i])
    avgpred_tmp = np.mean(iopkl.load(), axis=-1)
    predacc_value_15comp[:,i,:] = avgpred_tmp

    iopkl = iofiles.make_ioinstance(predacc_path_50comp[i])
    avgpred_tmp = np.mean(iopkl.load(), axis=-1)
    predacc_value_50comp[:,i,:] = avgpred_tmp

    iopkl = iofiles.make_ioinstance(predacc_path_100comp[i])
    avgpred_tmp = np.mean(iopkl.load(), axis=-1)
    predacc_value_100comp[:,i,:] = avgpred_tmp

    iopkl = iofiles.make_ioinstance(discrimination_path_15comp[i])
    avgdist_tmp = iopkl.load()
    discrimination_value_15comp[:,i,:] = avgdist_tmp

    iopkl = iofiles.make_ioinstance(discrimination_path_50comp[i])
    avgdist_tmp = iopkl.load()
    discrimination_value_50comp[:,i,:] = avgdist_tmp

    iopkl = iofiles.make_ioinstance(discrimination_path_100comp[i])
    avgdist_tmp = iopkl.load()
    discrimination_value_100comp[:,i,:] = avgdist_tmp

rb_mat_mask = (rb_mat>thr_reb)
rb_mat_mask = np.tile(rb_mat_mask[np.newaxis, ...], (len(subjnum), 1, 1))

# 100 component, prediction accuracy
predacc_100comp_highreb = predacc_value_100comp*rb_mat_mask
predacc_100comp_highreb[predacc_100comp_highreb==0] = np.nan
predacc_100comp_highreb = predacc_100comp_highreb.reshape((len(subjnum), len(task_idx)*len(masklabel)))

mean_predacc_100comp_hr = np.nanmean(predacc_100comp_highreb,axis=-1)
ste_predacc_100comp_hr = np.nanstd(predacc_100comp_highreb,axis=-1)/np.sqrt(len(predacc_100comp_highreb[~np.isnan(predacc_100comp_highreb)]))

# 50 component, prediction accuracy
predacc_50comp_highreb = predacc_value_50comp*rb_mat_mask
predacc_50comp_highreb[predacc_50comp_highreb==0] = np.nan
predacc_50comp_highreb = predacc_50comp_highreb.reshape((len(subjnum), len(task_idx)*len(masklabel)))

mean_predacc_50comp_hr = np.nanmean(predacc_50comp_highreb,axis=-1)
ste_predacc_50comp_hr = np.nanstd(predacc_50comp_highreb,axis=-1)/np.sqrt(len(predacc_50comp_highreb[~np.isnan(predacc_50comp_highreb)]))

# 15 component, prediction accuracy
predacc_15comp_highreb = predacc_value_15comp*rb_mat_mask
predacc_15comp_highreb[predacc_15comp_highreb==0] = np.nan
predacc_15comp_highreb = predacc_15comp_highreb.reshape((len(subjnum), len(task_idx)*len(masklabel)))

mean_predacc_15comp_hr = np.nanmean(predacc_15comp_highreb,axis=-1)
ste_predacc_15comp_hr = np.nanstd(predacc_15comp_highreb,axis=-1)/np.sqrt(len(predacc_15comp_highreb[~np.isnan(predacc_15comp_highreb)]))


# 100 component, discrimination
discrimination_100comp_highreb = discrimination_value_100comp*rb_mat_mask
discrimination_100comp_highreb[discrimination_100comp_highreb==0] = np.nan
discrimination_100comp_highreb = discrimination_100comp_highreb.reshape((len(subjnum), len(task_idx)*len(masklabel)))

mean_discrimination_100comp_hr = np.nanmean(discrimination_100comp_highreb,axis=-1)
ste_discrimination_100comp_hr = np.nanstd(discrimination_100comp_highreb,axis=-1)/np.sqrt(len(discrimination_100comp_highreb[~np.isnan(discrimination_100comp_highreb)]))

# 50 component, discrimination
discrimination_50comp_highreb = discrimination_value_50comp*rb_mat_mask
discrimination_50comp_highreb[discrimination_50comp_highreb==0] = np.nan
discrimination_50comp_highreb = discrimination_50comp_highreb.reshape((len(subjnum), len(task_idx)*len(masklabel)))

mean_discrimination_50comp_hr = np.nanmean(discrimination_50comp_highreb,axis=-1)
ste_discrimination_50comp_hr = np.nanstd(discrimination_50comp_highreb,axis=-1)/np.sqrt(len(discrimination_50comp_highreb[~np.isnan(discrimination_50comp_highreb)]))


# 15 component, discrimination
discrimination_15comp_highreb = discrimination_value_15comp*rb_mat_mask
discrimination_15comp_highreb[discrimination_15comp_highreb==0] = np.nan
discrimination_15comp_highreb = discrimination_15comp_highreb.reshape((len(subjnum), len(task_idx)*len(masklabel)))

mean_discrimination_15comp_hr = np.nanmean(discrimination_15comp_highreb,axis=-1)
ste_discrimination_15comp_hr = np.nanstd(discrimination_15comp_highreb,axis=-1)/np.sqrt(len(discrimination_15comp_highreb[~np.isnan(discrimination_15comp_highreb)]))


 
