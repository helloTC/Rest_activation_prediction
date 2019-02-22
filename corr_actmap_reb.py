# Correlation between activation and reliability

from os.path import join as pjoin
import numpy as np
import cifti
import matplotlib.pyplot as plt
import matplotlib

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/'

taskidx = np.array([2,5,8,17,18,19,20,21,24,27,38])

# Load reliability map
reb_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'reliability_region_LGL_100Parcels.dscalar.nii'))
reb_map = reb_map[taskidx, np.newaxis, :]

# Load mask
mask, _ = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
masklabel = np.unique(mask[mask!=0])

# Transfer reb_map to reb_mat
rb_mat = np.zeros((len(taskidx), len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(len(taskidx)):
        rb_mat[j,i] = reb_map[j, (mask==masklbl)][0]

# Load activation map
tasklbl = np.array(['emotion', 'gambling', 'language', 'LF', 'LH', 'RF', 'RH', 'tongue', 'relation', 'tom', 'wm'])
act_map = np.zeros((len(tasklbl), 91282))
for i, tl in enumerate(tasklbl):
    act_tmp, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'actmap', tl, 'actmap.dscalar.nii'))
    act_map[i,:] = (np.mean(np.abs(act_tmp),axis=0))
act_map = act_map[:,np.newaxis,:]
act_mat = np.zeros((len(tasklbl), len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(len(tasklbl)):
        act_mat[j,i] = np.mean(act_map[j, (mask==masklbl)])

# Plot figures
colors = matplotlib.cm.rainbow(np.linspace(0,1,12))
motor_idx = np.array([3,4,5,6,7])
motor_act = act_mat[motor_idx,:].flatten()
motor_rb = rb_mat[motor_idx,:].flatten()

fig, ax = plt.subplots()
ax.scatter(rb_mat[0,:], act_mat[0,:], color=colors[0], label='emotion')
ax.scatter(rb_mat[1,:], act_mat[1,:], color=colors[1], label='gambling')
ax.scatter(rb_mat[2,:], act_mat[2,:], color=colors[2], label='language')
ax.scatter(motor_rb, motor_act, color=colors[5], label='motor')
ax.scatter(rb_mat[8,:], act_mat[8,:], color=colors[9], label='relation')
ax.scatter(rb_mat[9,:], act_mat[9,:], color=colors[10], label='social')
ax.scatter(rb_mat[10,:], act_mat[10,:], color=colors[11], label='working memory')
plt.legend()
plt.show()










