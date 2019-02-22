# Calculate test-retest reliability in these 40 subjects across global brain

from os.path import join as pjoin
from ATT.algorithm import tools
import framework_rt as fr
import cifti
import numpy as np

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'

with open(pjoin(parpath, 'tables', 'sessid_trt'), 'r') as f:
    sessid_trt = f.read().splitlines()

# mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'mmp_subcortex_mask.dscalar.nii'))
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
masklabel = np.unique(mask[mask!=0])

actmap_path = [pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore', sid+'_cohend_zscore.dtseries.nii') for sid in sessid_trt]
actmap_trt_path = [pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore_trt', sid+'_cohend_zscore.dscalar.nii') for sid in sessid_trt]

actmap = fr.cifti_read(actmap_path, np.arange(47), 'all')
actmap_trt = fr.cifti_read(actmap_trt_path, np.arange(47), 'all')

trt_map = np.zeros((47,1,91282))
for task in range(47):
    reliability, discrimination = fr.pred_partsim(actmap[:,task,:], actmap_trt[:,task,:], mask[0,:]) 
    mean_reliability = np.mean(reliability,axis=1)
    for i, masklbl in enumerate(masklabel):
        trt_map[task, (mask==masklbl)] = mean_reliability[i]







