# Generate the randomized beta map for further testing changes of prediction accuracy and discrimination

from os.path import join as pjoin
import cifti
import numpy as np
from ATT.iofunc import iofiles

def randomize_ROI(rawdata, mask=None):
    """
    Randomize the averaged data in ROI that defined by the mask
    if mask is None, randomized original data in global brain (but not cross the ROIs)

    Parameters:
    ------------
    rawdata: original activation data
    mask: mask to define ROIs

    Return:
    -------
    rand_data: randomized data
    """
    masklabel = np.unique(mask[mask!=0])
    rawshape = rawdata.shape
    if mask is None:
        rawdata_flatten = rawdata.flatten()
        rddata_flatten = np.random.choice(rawdata_flatten, len(rawdata_flatten), replace=False)
        rand_data = rddata_flatten.reshape(rawshape)
    else:
        rawdata = rawdata[:,np.newaxis,:]
        rand_data = np.zeros_like(rawdata)
        randomized_masklabel = np.random.choice(masklabel, len(masklabel), replace=False)
        for i, masklbl in enumerate(masklabel):
            avg_rdroi = np.mean(rawdata[:,(mask==randomized_masklabel[i])],axis=1)
            rand_data[:,(mask==masklbl)] = np.tile(avg_rdroi[:,np.newaxis],(len(mask[mask==masklbl])))
        rand_data = rand_data[:,0,:]
    return rand_data 

def simple_surface_by_ROI(rawdata, mask):
    """
    Simple surface using ROI, extract the averaged value of each ROI

    Parameters:
    -----------
    rawdata: the original data, [contrasts]*[spatial vertex]
    mask: mask to define ROIs

    Returns:
    --------
    sim_mat: the simplified matrix from rawdata
    """
    masklabel = np.unique(mask[mask!=0])
    rawdata = rawdata[:,np.newaxis,:]
    sim_mat = np.zeros((rawdata.shape[0], len(masklabel)))
    for i, masklbl in enumerate(masklabel):
        for j in range(rawdata.shape[0]):
            sim_mat[j,i] = np.mean(rawdata[j,(mask==masklbl)])
    return sim_mat




parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
ncomp = '100'
task = 'wm'

# Read Mask
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))

# Load avgbeta
avgbeta, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_avgbeta', ncomp+'comp', 'avgbeta_'+task+'_'+ncomp+'comp.dscalar.nii'))
rdbeta = randomize_ROI(avgbeta, mask=mask)

# Save radomized ROI
ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_rdavgbeta', ncomp+'comp', 'rdavgbeta_'+task+'_'+ncomp+'comp.dscalar.nii'))
ioscalar.save_from_existed_header(header, rdbeta)



