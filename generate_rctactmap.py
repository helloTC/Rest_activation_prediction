# Generation of reconstructed map
from os.path import join as pjoin
import cifti
import framework_rt as fr
import numpy as np
from ATT.iofunc import iofiles

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
ncomp = '15'
task = 'wm'

print('task {}'.format(task))

# Session id
with open(pjoin(parpath, 'program', 'framework', 'test_sessid'), 'r') as f:
    sessid = f.read().splitlines()
# sessid = sessid[:3]

# Mask for header
mask, header = cifti.read(pjoin(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii')))

# Read Subject IC
subjIC_path = [pjoin('/nfs/p1/atlases/subjIC','IC'+ncomp,'IC'+ncomp+'_'+sid+'.dscalar.nii') for sid in sessid]
subjic = fr.cifti_read(subjIC_path, np.arange(int(ncomp)), hemi='all')

# Read the averaged beta
avgbeta_path = [pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_avgbeta', ncomp+'comp', 'avgbeta_'+task+'_'+ncomp+'comp.dscalar.nii')]
avgbeta = fr.cifti_read(avgbeta_path, np.arange(int(ncomp)), hemi='all')
avgbeta = np.tile(avgbeta, (len(sessid),1,1))

# Reconstructed activation map
rctactmap = np.sum(avgbeta*subjic,axis=1)

ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'rctactmap', 'LGL_global_'+ncomp+'comp_'+task, 'rctactmap.dscalar.nii'))
ioscalar.save_from_existed_header(header, rctactmap)

