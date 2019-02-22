import framework_rt as fr
from os.path import join as pjoin
import cifti
from ATT.iofunc import iofiles
import numpy as np
from scipy import stats

with open('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/tables/subjIC_sessid', 'r') as f:
    subjID = f.read().splitlines()
subjID = subjID[:203]
nsubj = len(subjID)

mask, header = cifti.read('/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii')
neighbor_table = fr.mask_dictdata(mask)

icmap_path = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/rest_comp/subjIC_itr/IC50_'+sid+'.dscalar.nii' for sid in subjID]

myelinmap, _ = cifti.read('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/rest_comp/structure/thickness_203.dscalar.nii')
myelinmap = myelinmap[:,np.newaxis,:]
myelinmap_zscore = stats.zscore(myelinmap,axis=-1)

subcortex_component = np.array([27,28,31,34,36,38,39,40,41,42,43,44,45,46,47,48,49])
cortex_component = np.delete(np.arange(50), subcortex_component)

icmap = fr.cifti_read(icmap_path, cortex_component, 'both')
icmap_zscore = stats.zscore(icmap,axis=-1)
mean_icmap = np.mean(icmap_zscore,axis=0)

for i in range(nsubj):
    print('Now calculating beta for subject {0}'.format(i+1))
    betamap, scoremap = fr.linear_estimate_model_spatial(myelinmap_zscore[i,...], icmap_zscore[i,...], neighbor_table)
    fr.save_maps_to_nifti(betamap, pjoin('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/program/framework/betamap/MMP_cortex_33comp_thickness', 'beta_'+str(i+1)+'.nii.gz'))
    


