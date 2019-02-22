import framework_rt as fr
from os.path import join as pjoin
import cifti
from ATT.iofunc import iofiles
import numpy as np
from scipy import stats

with open('/nfs/p1/atlases/subjIC/subjIC_sessid', 'r') as f:
    subjID = f.read().splitlines()

# subjID = subjID[770:]

nsubj = len(subjID)

# mask, header = cifti.read('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/rest_comp/mmp_subcortex_mask.dscalar.nii')
mask, header = cifti.read('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/rest_comp/mmp_subcortex_mask.dscalar.nii')
neighbor_table = fr.mask_dictdata(mask)

ncomp = '15'
actmap_path = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/task_merge_cohend/cohend_47contrast_zscore/'+sid+'_cohend_zscore.dtseries.nii' for sid in subjID]
icmap_path = ['/nfs/p1/atlases/subjIC/IC'+ncomp+'/IC'+ncomp+'_'+sid+'.dscalar.nii' for sid in subjID]

for i,sid in enumerate(subjID):
    print('Now calculating beta for subject {0} with component number {1}'.format(i+1, int(ncomp)))
    actmap_zscore, _ = cifti.read(actmap_path[i])
    actmap_zscore = actmap_zscore[18,...]
    # actmap_zscore = -1.0*actmap_zscore
    actmap_zscore = actmap_zscore[np.newaxis, ...]

    icmap_zscore, _ = cifti.read(icmap_path[i])

    betamap, _ = fr.linear_estimate_model_spatial(actmap_zscore, icmap_zscore, neighbor_table)
    fr.save_maps_to_nifti(betamap, pjoin('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/program/framework/betamap/MMP_global_'+ncomp+'comp_LH', 'beta_'+sid+'.nii.gz'))
    


