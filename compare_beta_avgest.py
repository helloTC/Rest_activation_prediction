import numpy as np
import cifti
from scipy import stats
import framework_rt as fr
from os.path import join as pjoin

parpath = '/nfs/h1/workingshop/huangtaicheng/hcp_test/'

with open('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/tables/subjIC_sessid', 'r') as f:
    subjID = f.read().splitlines()
subjID = subjID[:203]
nsubj = len(subjID)

mask, header = cifti.read('/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')
neighbor_table = fr.mask_dictdata(mask)
actmap_path = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/task_merge_cohend/cohend_47contrast_zscore/'+sid+'_cohend_zscore.dtseries.nii' for sid in subjID]
icmap_path = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/rest_comp/subjIC_itr/IC50_'+sid+'.dscalar.nii' for sid in subjID]
actmap = fr.cifti_read(actmap_path, 8, 'both')
actmap_zscore = stats.zscore(actmap,axis=-1)
mean_actmap = np.mean(actmap_zscore,axis=0)

subcortex_component = np.array([27,28,31,34,36,38,39,40,41,42,43,44,45,46,47,48,49])
cortex_component = np.delete(np.arange(50), subcortex_component)
icmap = fr.cifti_read(icmap_path, cortex_component, 'both')
icmap_zscore = stats.zscore(icmap,axis=-1)
mean_icmap = np.mean(icmap_zscore,axis=0)

subjnum = np.arange(1,nsubj+1,1)
betamap_path = [pjoin(parpath, 'program', 'framework', 'betamap', 'MMP_cortex_33comp_language', 'beta_'+str(sid)+'.nii.gz') for sid in subjnum]
betamap = fr.nifti_read(betamap_path, np.arange(33))
mean_betamap_fromavg = np.mean(betamap,axis=0)

mean_betamap_fromcal, scoremap = fr.linear_estimate_model_spatial(mean_actmap, mean_icmap, neighbor_table)
