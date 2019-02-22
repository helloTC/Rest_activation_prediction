# Calculate test-retest reliability in these 40 subjects across global brain

from os.path import join as pjoin
from ATT.algorithm import tools
import framework_rt as fr
import cifti
import numpy as np

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'

with open(pjoin(parpath, 'tables', 'sessid_trt'), 'r') as f:
    sessid_trt = f.read().splitlines()

mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'mmp_subcortex_mask.dscalar.nii'))

actmap_path = [pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore', sid+'_cohend_zscore.dtseries.nii') for sid in sessid_trt]
actmap_trt_path = [pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore_trt', sid+'_cohend_zscore.dscalar.nii') for sid in sessid_trt]

actmap = fr.cifti_read(actmap_path, 21, 'all')
actmap_trt = fr.cifti_read(actmap_trt_path, 21, 'all')

trt_map = np.zeros((1,91282))
for i in range(91282):
    print('ICC in voxel {}'.format(i+1))
    data1 = actmap[:,0,i]
    data2 = actmap_trt[:,0,i]
    con_data = np.vstack((data1, data2)).T
    icc_val, _ = tools.icc(con_data, '(3,1)')
    trt_map[0,i] = icc_val








