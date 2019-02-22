# Select specific subject number from path

from os.path import join as pjoin
import numpy as np
import framework_rt as fr
from ATT.algorithm import tools
import cifti
from ATT.iofunc import iofiles
import nibabel as nib

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'

# with open(pjoin(parpath, 'tables', 'subjIC_sessid'), 'r') as f:
#     sessid = f.read().splitlines()
# sessid = sessid[:750]

# with open(pjoin(parpath, 'tables', 'sessid_trt'), 'r') as f:
#     sessid_trt = f.read().splitlines()

with open('train_sessid', 'r') as f:
    train_sessid = f.read().splitlines()
with open('test_sessid', 'r') as f:
    test_sessid = f.read().splitlines()

ncomp = 100
task = 'language'
actmap_path = np.array([pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore', sid+'_cohend_zscore.dtseries.nii') for sid in test_sessid])
icmap_path = np.array([pjoin('/nfs/p1/atlases', 'subjIC', 'IC'+str(ncomp), 'IC'+str(ncomp)+'_'+sid+'.dscalar.nii') for sid in test_sessid])
# actmap_trt_path = np.array([pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore_trt', sid+'_cohend_zscore.dscalar.nii') for sid in test_sessid])

print('Now loading connectivity maps and activation maps')
icmap = fr.cifti_read(icmap_path, np.arange(ncomp), 'all')
actmap = fr.cifti_read(actmap_path, 8, 'all')
# actmap_trt = fr.cifti_read(actmap_trt_path, 8, 'all')
actmap = -1.0*actmap

betamap_path = np.array([pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_'+str(ncomp)+'comp_'+task, 'beta_'+sid+'.nii.gz') for sid in train_sessid])

mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
# mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'mmp_subcortex_mask.dscalar.nii'))

# reliability, _ = fr.pred_partsim(actmap, actmap_trt, mask)
# sort_idx = np.argsort(reliability, axis=0)

# nsubj = [10, 20, 30, 50, 100, 150, 200, 300]

# nsubj = [len(betamap_name)]
# nsubj = [30, 50, 70, 90, 110, 130, 150, 200, 300, 500, 814]
nsubj = [814]
# r_test_all = []
raccpred_all = []
discrimination_all = []
for i, nj in enumerate(nsubj):
    print('Subject number {}, task {}'.format(nj, task))
    n_train_subj = np.random.choice(range(len(betamap_path)), nj, replace=False)

    betapath_train = betamap_path[n_train_subj]

    print('Now loading and calculating the averaged beta maps')
    train_betamap = np.zeros((ncomp,91282))
    for bt in betapath_train:
        betamap = nib.load(bt).get_data()
        train_betamap+=betamap
    train_betamap = train_betamap/len(betapath_train)
    train_betamap = train_betamap[np.newaxis,...]

    train_betamap = np.tile(train_betamap, (len(test_sessid),1,1))
    rctactmap = np.sum(icmap*train_betamap,axis=1)
    rctactmap = rctactmap[:,np.newaxis,:]

    r_accpred, discrimination = fr.pred_partsim(actmap, rctactmap, mask)
    
    raccpred_all.append(r_accpred)
    discrimination_all.append(discrimination)
raccpred_all = np.array(raccpred_all)
discrimination_all = np.array(discrimination_all)
# iopkl_predacc = iofiles.make_ioinstance('predacc/predacc_'+task+'_'+str(ncomp)+'comp.pkl')
# iopkl_disc = iofiles.make_ioinstance('predacc/discrimination_'+task+'_'+str(ncomp)+'comp.pkl')
# iopkl_predacc.save(raccpred_all)
# iopkl_disc.save(discrimination_all)
