# Generate the averaged beta map
from os.path import join as pjoin
import numpy as np
import nibabel as nib
import cifti
from ATT.iofunc import iofiles

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
_, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))

with open('train_sessid', 'r') as f:
    train_sessid = f.read().splitlines()

ncomp = 100
task = ['tom']
for tk in task:
    print('Task {}'.format(tk))
    betamap_path = np.array([pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_'+str(ncomp)+'comp_'+tk, 'beta_'+sid+'.nii.gz') for sid in train_sessid])

    train_betamap = np.zeros((ncomp, 91282))
    for i, bt in enumerate(betamap_path):
        # print('Beta {}'.format(i+1))
        betamap = nib.load(bt).get_data()
        train_betamap += betamap
    train_betamap = train_betamap/len(train_sessid)

    ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_avgbeta', str(ncomp)+'comp', 'avgbeta_'+tk+'_'+str(ncomp)+'comp.dscalar.nii'))
    ioscalar.save_from_existed_header(header, train_betamap)






