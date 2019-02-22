# Distinguish regions by its reliability.

import cifti
from os.path import join as pjoin
import numpy as np
from ATT.iofunc import iofiles
import matplotlib
import matplotlib.pyplot as plt

parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/'

reb_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'reliability_region_LGL_100Parcels.dscalar.nii'))
reb_map = reb_map[:,np.newaxis,:]
dist_map, _ = cifti.read(pjoin(parpath, 'program', 'framework', 'predacc', 'reliability', 'discrimination_region_LGL_100Parcels.dscalar.nii'))
dist_map = dist_map[:,np.newaxis,:]
mask, header = cifti.read(pjoin(parpath, 'rest_comp', 'LGL_100Parcels_7Network_subregion.dscalar.nii'))
masklabel = np.unique(mask[mask!=0])

rb_mat = np.zeros((47,len(masklabel)))
dist_mat = np.zeros((47,len(masklabel)))
for i, masklbl in enumerate(masklabel):
    for j in range(47):
        rb_mat[j, i] = reb_map[j, (mask==masklbl)][0]
        dist_mat[j, i] = dist_map[j, (mask==masklbl)][0]

ncomp = '50'
# Multiple tasks
# Tasks contain emotion, gambling, language, motor, tongue, relation, tom, wm
task_idx = np.array([2, 5, 8, 17, 18, 19, 20, 21, 24, 27, 38])
taskname = np.array(['emotion', 'gambling', 'language', 'LF', 'LH', 'RF', 'RH', 'tongue', 'relation', 'tom','wm'])
predacc_path = [pjoin(parpath, 'program', 'framework', 'predacc', 'predacc', 'predacc_'+tn+'_'+ncomp+'comp.pkl') for tn in taskname]
discrimination_path = [pjoin(parpath, 'program', 'framework', 'predacc', 'discrimination', 'discrimination_'+tn+'_'+ncomp+'comp.pkl') for tn in taskname]
# 100, 100 test subj
predacc_value = np.zeros((len(task_idx), len(masklabel), 100))
discrimination_value = np.zeros((len(task_idx), len(masklabel)))
for i, ti in enumerate(task_idx):
    iopkl = iofiles.make_ioinstance(predacc_path[i])
    predacc_value[i, ...] = iopkl.load()[-1,...]
    iopkl = iofiles.make_ioinstance(discrimination_path[i])
    discrimination_value[i,...] = iopkl.load()[-1,...]
mean_predacc_value = np.mean(predacc_value,axis=-1)
rb_mat_task = rb_mat[task_idx,:]
dist_mat_task = dist_mat[task_idx,:]

# design data for figure plotting.
# Here we merged Several motor conditions together.
motor_idx = np.array([3,4,5,6,7])
motor_predacc = mean_predacc_value[motor_idx, :].flatten()
motor_rb = rb_mat_task[motor_idx, :].flatten()
motor_preddist = discrimination_value[motor_idx, :].flatten()
motor_dist = dist_mat_task[motor_idx,:].flatten()
colors = matplotlib.cm.rainbow(np.linspace(0,1,12))

# Relabelled region by network
with open('/nfs/p1/atlases/Schaefer_localglobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_100Parcels_7Networks_order_info.txt', 'r') as f:
    rglabel_name = f.read().splitlines()
rglabel_name = rglabel_name[::2]
network_name = np.array([rn.split('_')[2] for rn in rglabel_name])
rglabel_dict = {}
for i, nn in enumerate(network_name):
    if nn not in rglabel_dict.keys():
        rglabel_dict[nn] = np.array([int(i)])
    else:
        rglabel_dict[nn] = np.append(rglabel_dict[nn],int(i))


# plot relationship between prediction accuracy and reliability

# fig, ax = plt.subplots()
# ax.scatter(rb_mat_task[0,:], mean_predacc_value[0,:], color=colors[0], label='emotion')
# ax.scatter(rb_mat_task[1,:], mean_predacc_value[1,:], color=colors[1], label='gambling')
# ax.scatter(rb_mat_task[2,:], mean_predacc_value[2,:], color=colors[2], label='language')
# ax.scatter(motor_rb, motor_predacc, color=colors[5], label='motor')
# ax.scatter(rb_mat_task[8,:], mean_predacc_value[8,:], color=colors[9], label='relation')
# ax.scatter(rb_mat_task[9,:], mean_predacc_value[9,:], color=colors[10], label='social')
# ax.scatter(rb_mat_task[10,:], mean_predacc_value[10,:], color=colors[11], label='working memory')
# ax.set_xlim(-0.15,0.85)
# ax.set_ylim(-0.25,0.85)
# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(rb_mat_task[3,:], mean_predacc_value[3,:], color=colors[3], label='LF')
# ax.scatter(rb_mat_task[4,:], mean_predacc_value[4,:], color=colors[4], label='LH')
# ax.scatter(rb_mat_task[5,:], mean_predacc_value[5,:], color=colors[6], label='RF')
# ax.scatter(rb_mat_task[6,:], mean_predacc_value[6,:], color=colors[7], label='RH')
# ax.scatter(rb_mat_task[7,:], mean_predacc_value[7,:], color=colors[8], label='tongue')
# ax.set_xlim(-0.15,0.85)
# ax.set_ylim(-0.25,0.85)
# plt.legend()
# plt.show()

# plot relationship between discrimination and reliability

# fig, ax = plt.subplots()
# ax.scatter(rb_mat_task[0,:], discrimination_value[0,:], color=colors[0], label='emotion')
# ax.scatter(rb_mat_task[1,:], discrimination_value[1,:], color=colors[1], label='gambling')
# ax.scatter(rb_mat_task[2,:], discrimination_value[2,:], color=colors[2], label='language')
# ax.scatter(motor_rb, motor_preddist, color=colors[5], label='motor')
# ax.scatter(rb_mat_task[8,:], discrimination_value[8,:], color=colors[9], label='relation')
# ax.scatter(rb_mat_task[9,:], discrimination_value[9,:], color=colors[10], label='social')
# ax.scatter(rb_mat_task[10,:], discrimination_value[10,:], color=colors[11], label='working memory')
# ax.set_xlim(-0.15,0.85)
# ax.set_ylim(-0.05,0.90)
# plt.legend()
# plt.show()

colors_network = matplotlib.cm.rainbow(np.linspace(0,1,len(rglabel_dict.keys())))
fig,ax = plt.subplots()
ax.scatter(rb_mat_task[:,rglabel_dict['Cont']].flatten(), discrimination_value[:,rglabel_dict['Cont']].flatten(), color=colors_network[0], label='Control')
ax.scatter(rb_mat_task[:,rglabel_dict['Default']].flatten(), discrimination_value[:,rglabel_dict['Default']].flatten(), color=colors_network[1], label='Default')
ax.scatter(rb_mat_task[:,rglabel_dict['Limbic']].flatten(), discrimination_value[:,rglabel_dict['Limbic']].flatten(), color=colors_network[2], label='Limbic')
ax.scatter(rb_mat_task[:,rglabel_dict['DorsAttn']].flatten(), discrimination_value[:,rglabel_dict['DorsAttn']].flatten(), color=colors_network[3], label='Dorsal Attention')
ax.scatter(rb_mat_task[:,rglabel_dict['SalVentAttn']].flatten(), discrimination_value[:,rglabel_dict['SalVentAttn']].flatten(), color=colors_network[4], label='Ventral Attention')
ax.scatter(rb_mat_task[:,rglabel_dict['Vis']].flatten(), discrimination_value[:,rglabel_dict['Vis']].flatten(), color=colors_network[5], label='Visual')
ax.scatter(rb_mat_task[:,rglabel_dict['SomMot']].flatten(), discrimination_value[:,rglabel_dict['SomMot']].flatten(), color=colors_network[6], label='Somatomotor')
ax.set_xlim(-0.15,0.85)
ax.set_ylim(-0.05,0.90)
plt.legend()
plt.show()


