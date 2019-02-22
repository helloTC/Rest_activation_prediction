import framework_rt as fr
from os.path import join as pjoin
import cifti
from ATT.iofunc import iofiles
from sklearn import linear_model
import numpy as np
from scipy import stats
from ATT.algorithm import tools

with open('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/tables/subjIC_sessid', 'r') as f:
    subjID = f.read().splitlines()
subjID = subjID[:203]
nsubj = len(subjID)

mask, header = cifti.read('/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')
neighbor_table = fr.mask_dictdata(mask)

actmap_path = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/task_merge_cohend/cohend_24contrast_zscore/'+sid+'_cohend_zscore.dtseries.nii' for sid in subjID]
icmap_subj_itr_path = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/rest_comp/subjIC_itr/IC50_'+sid+'.dscalar.nii' for sid in subjID]

glm = linear_model.LinearRegression(fit_intercept=False)

actmap = fr.cifti_read(actmap_path, 0, 'both')
actmap_zscore = stats.zscore(actmap,axis=-1)
mean_actmap = np.mean(actmap_zscore,axis=0)

for i in range(nsubj):
    glm.fit(mean_actmap.T, actmap_zscore[i,...].T)
    actmap_zscore[i,...] = actmap_zscore[i,...] - np.dot(glm.coef_, mean_actmap)

subcortex_component = np.array([27,28,31,34,36,38,39,40,41,42,43,44,45,46,47,48,49])
# subcortex_component = np.array([])
cortex_component = np.delete(np.arange(50), subcortex_component)

icmap = fr.cifti_read(icmap_subj_itr_path, cortex_component, 'both')
icmap_zscore = stats.zscore(icmap,axis=-1)
# mean_icmap = np.mean(icmap_zscore,axis=0)
# for i in range(nsubj):
#     glm.fit(mean_icmap.T, icmap_zscore[i,...].T)
#     icmap_zscore[i,...] = icmap_zscore[i,...] - np.dot(glm.coef_, mean_icmap)

for i in range(nsubj):
    print('Now calculating beta for subject {0}'.format(i+1))
    betamap, scoremap = fr.linear_estimate_model(actmap_zscore[i,...], icmap_zscore[i,...], neighbor_table)
    fr.save_maps_to_nifti(betamap, pjoin('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/program/framework/betamap/MMP_cortex_33comp_residuecognitive', 'beta_'+str(i+1)+'.nii.gz'))
    


