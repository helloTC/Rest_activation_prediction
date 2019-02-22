# Extract the 100 subjects' activation map from the whole dataset to make the comparision between actmap and reconstructed actmap easily.

from os.path import join as pjoin
import framework_rt as fr
import cifti
import numpy as np
from ATT.iofunc import iofiles

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
# Sessid
with open('test_sessid', 'r') as f:
    sessid = f.read().splitlines()

tasklbl = np.array([2, 5, 8, 17, 18, 19, 20, 21, 24, 27, 38])
taskname = np.array(['emotion', 'gambling', 'language', 'LF', 'LH', 'RF', 'RH', 'tongue', 'relation', 'tom', 'wm'])
assert len(tasklbl) == len(taskname), 'Length not matched.'

# Read mask map for header
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))

# Read activation map
actmap_path = [pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast', sid+'_cohend.dtseries.nii') for sid in sessid]
actmap = fr.cifti_read(actmap_path, tasklbl, hemi='all')

neg_tasklbl = np.array([1, 2, 8, 9])
actmap[:,neg_tasklbl] = -1.0*actmap[:,neg_tasklbl]

# Save files
for i, tn in enumerate(taskname):
    print('Now handle task {}'.format(tn))
    ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'actmap', tn, 'actmap.dscalar.nii'))
    ioscalar.save_from_existed_header(header, actmap[:,i,:])





