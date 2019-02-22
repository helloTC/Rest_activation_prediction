# Method to evaluate component contribution to the prediction activation.
# We built model:
# M1: y = b1x1 + b2x2 + b3x3
# M2: y = b1x1 + b2x2
# And calculate the PRE(Proportional Reductional of Error) between these two models with the formula as:
# PRE = (Residual Sum of Squares of M2 - RSS of M1)/(RSS of M2)
# The PRE represents the effect size of the predictor. It represents its unique contribution in percentage in explaining the variance of the dependent variable.

from os.path import join as pjoin
import numpy as np
import cifti
from ATT.iofunc import iofiles

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test'
ncomp = '100'
task = 'emotion'

# Average IC
print('Preparing averaged IC')
avg_subjIC = np.zeros((int(ncomp), 91282))
with open(pjoin(parpath, 'program', 'framework', 'test_sessid'), 'r') as f:
    sessid = f.read().splitlines()
for i, sid in enumerate(sessid):
    subjIC, _ = cifti.read(pjoin('/nfs/p1/atlases/subjIC/IC'+ncomp, 'IC'+ncomp+'_'+sid+'.dscalar.nii'))
    avg_subjIC += subjIC
avg_subjIC /= len(sessid)

# Load average beta
print('Loading beta map')
avg_beta, header = cifti.read(pjoin(parpath, 'program', 'framework', 'betamap', 'LGL_global_avgbeta', ncomp+'comp', 'avgbeta_'+task+'_'+ncomp+'comp.dscalar.nii'))

# Average activation map
print('Preparing the averaged activation map')
taskidx = np.array([2,5,8,17,18,19,20,21,24,27,38])
avg_actdata = np.zeros((len(taskidx), 91282))
for i, sid in enumerate(sessid):
    actvalue, _ = cifti.read(pjoin(parpath, 'task_merge_cohend', 'cohend_47contrast_zscore', sid+'_cohend_zscore.dtseries.nii'))
    avg_actdata += actvalue[taskidx, :]
avg_actdata /= len(sessid)

ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'avgACTIC', 'avg_cohend_100subj.dscalar.nii'))
ioscalar.save_from_existed_header(header, avg_actdata)

ioscalar = iofiles.make_ioinstance(pjoin(parpath, 'program', 'framework', 'avgACTIC', 'avg_IC_100subj.dscalar.nii'))
ioscalar.save_from_existed_header(header, avg_subjIC)













