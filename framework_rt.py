# Rewrite framework for predicting task activation maps from rest components

import numpy as np
import cifti
import nibabel as nib
from scipy import stats, signal
from sklearn import linear_model
from ATT.algorithm import surf_tools, tools


def cifti_read(path, ntk_comp, hemi='both'):
    """
    Read cifti data
    """
    if type(ntk_comp) is int:
        ntk_comp = np.array([ntk_comp])
    if type(ntk_comp) is list:
        ntk_comp = np.array(ntk_comp)

    if hemi == 'left':
        vxrgn = np.arange(29696)
    elif hemi == 'right':
        vxrgn = np.arange(29696,59412)
    elif hemi == 'subcortex':
        vxrgn = np.arange(59412,91282)
    elif hemi == 'both':
        vxrgn = np.arange(59412)
    elif hemi == 'all':
        vxrgn = np.arange(91282)
    else:
        raise Exception('')
 
    data = np.zeros((len(path), len(ntk_comp), len(vxrgn)))
    for i, pt in enumerate(path):
        print('Now reading subject {0}'.format(i+1))
        data_tmp, _ = cifti.read(pt) 
        data[i,...] = data_tmp[ntk_comp[:,None], vxrgn]
    return data


def nifti_read(path, ntk_comp, hemi='both'):
    """
    Read imperfect nifti-data (some intermediate data, e.g. beta maps)
    """
    if type(ntk_comp) is int:
        ntk_comp = np.array([ntk_comp])
    if type(ntk_comp) is list:
        ntk_comp = np.array(ntk_comp)

    if hemi == 'left':
        vxrgn = np.arange(29696)
    elif hemi == 'right':
        vxrgn = np.arange(29696,59412)
    elif hemi == 'subcortex':
        vxrgn = np.arange(59412,91282)
    elif hemi == 'both':
        vxrgn = np.arange(59412)
    elif hemi == 'all':
        vxrgn = np.arange(91282)
    else:
        raise Exception('')

    data = np.zeros((len(path), len(ntk_comp), len(vxrgn)))
    for i, pt in enumerate(path):
        print('Now reading subject {0}'.format(i+1))
        data_tmp = nib.load(pt).get_data()
        data[i,...] = data_tmp[ntk_comp[:,None], vxrgn]
    return data


def mask_dictdata(mask):
    """
    Transfer mask data into the designed mask dictionary format.
    The designed mask dictionary: [label]:[vertices]

    Parameters:
    -----------
    mask: mask in cifti format

    Returns:
    --------
    mask_dict
    """
    print('Transfer mask data into the designed mask dictionary format.')
    mask_dict = {}
    masklbl = np.unique(mask[mask!=0]).astype('int')
    for lbl in masklbl:
        mask_dict[lbl] = np.where(mask==lbl)[-1]
    return mask_dict


def searchlight_dictdata(faces, nrings, vertex_list):
    """
    Function to generate neighbor vertex relationship for searchlight analysis
    The format of dictdata is [label]:[vertices]

    Parameters:
    -----------
    faces:
    nrings:
    vertex_list: vertex-index relationship, e.g. vertex_list[29696] = 32492

    Returns:
    --------
    output_vx
    """
    output_vx = {}
    vertex_list = list(vertex_list)
    index_dict = dict((value, idx) for idx,value in enumerate(vertex_list))
    for i, vl in enumerate(vertex_list):
        print('{0}:{1}'.format(i+1, vl))
        neighbor_vxidx = surf_tools.get_n_ring_neighbor(int(vl), faces, n=nrings)[0]
        neighbor_vxidx.intersection_update(set(index_dict.keys()))
        neighbor_vx = [index_dict[nv] for nv in neighbor_vxidx]
        output_vx[i] = neighbor_vx
    return output_vx


def linear_estimate_model_spatial(actmap, icmap, mask_dict, zscore = False, fit_intercept = False, used_model = 'GLM', upsample = True):
    """
    Estimate linear model in one subject using spatial pattern information

    Parameters:
    -----------
    actmap: activation map
    icmap: subject connectivity map
    mask_dict: mask dictionary
    zscore: whether do zscore in each roi, by default is False
    used_model: 'GLM', perform general linear model.
            'Lasso', Lasso, with alpha as 1.0 (by-default value)
    upsample: When sample number is smaller than feature number, upsample sample points by using FFT transformations. By default is True.

    Returns:
    --------
    beta_map:
    score_map: 
    """
    assert actmap.ndim == 2, "Only support prediction in one participant."
    assert icmap.ndim == 2, "Only support prediction in one participant."
    assert actmap.shape[-1] == icmap.shape[-1], "Actmap and icmap need to be matched."
    score_map = np.zeros_like(actmap)
    beta_map = np.zeros_like(icmap)
    if used_model == 'GLM': 
        model = linear_model.LinearRegression(fit_intercept)
    elif used_model == 'Lasso':
        model = linear_model.Lasso(fit_intercept=fit_intercept)
    else:
        raise Exception('We only support GLM or Lasso now.')
    for vl in mask_dict.keys():
        X_array = icmap[:,mask_dict[vl]].T
        Y_array = actmap[:,mask_dict[vl]].T
        if zscore is True:
            X_array = stats.zscore(X_array, axis=0)
            Y_array = stats.zscore(Y_array, axis=0)
        if Y_array.shape[0] < X_array.shape[1]:
           print('ROI {} does not have enough samples'.format(vl))
           Y_array = signal.resample(Y_array, X_array.shape[1]) 
           X_array = signal.resample(X_array, X_array.shape[1])
        model.fit(X_array, Y_array)
        beta_map[:,mask_dict[vl]] = np.tile(model.coef_[0,:], (len(mask_dict[vl]),1)).T
        score = model.score(X_array, Y_array)
        score_map[:,mask_dict[vl]] = score
    return beta_map, score_map
   

def linear_estimate_model_individual(actmap, icmap, mask_dict, zscore=False, fit_intercept=False, concate_option='average'):
    """
    Estimate linear model across subjects

    Parameters:
    ------------
    actmap: activation map
    icmap: connectivity map
    mask_dict: mask dictionary
    zscore: whether to do zscore in a ROI before performing linear estimation
    fit_intercept: whether to fit intercept when doing linear estimation
    concate_option: in each roi, set strategy to concatenate voxels, use 'average' or 'concate'. 'average' means average values in each roi across its spatial pattern, 'concate' means concatenate values in each roi across its spatial pattern.

    Returns:
    --------
    beta_map:
    score_map:
    """
    beta_map = np.zeros((icmap.shape[1], icmap.shape[2]))
    score_map = np.zeros((actmap.shape[1], actmap.shape[2]))
    model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    for vl in mask_dict.keys():
        # print('Now perform ROI {}'.format(vl))
        if type(mask_dict[vl]) is int:
            vx_select = np.array([mask_dict[vl]])
        else:
            vx_select = mask_dict[vl]
        X_array = icmap[...,vx_select]
        Y_array = actmap[...,vx_select]
        if concate_option == 'average':
            X_array = np.mean(X_array,axis=-1)
            Y_array = np.mean(Y_array, axis=-1)
        elif concate_option == 'concate':
            X_tmp = np.swapaxes(X_array, 1, 2)
            Y_tmp = np.swapaxes(Y_array, 1, 2)
            X_array = X_tmp.reshape((X_tmp.shape[0]*X_tmp.shape[1], X_tmp.shape[2]))
            Y_array = Y_tmp.reshape((Y_tmp.shape[0]*Y_tmp.shape[1], Y_tmp.shape[2]))
        else:
            raise Exception('Unsupport option expect for average or concate')
        if zscore is True:
            X_array = stats.zscore(X_array, axis=0)
            Y_array = stats.zscore(Y_array, axis=0)
        model.fit(X_array, Y_array)
        beta_map[:,vx_select] = np.tile(model.coef_[0,:], (len(vx_select),1)).T
        score_map[:,vx_select] = model.score(X_array, Y_array)
    return beta_map, score_map


def seperate_idx_cv(index_array, nfold=2):
    """
    A function to seperate index array with n-fold cross validation way.
    """
    n_idx = len(index_array)
    test_dict = {}
    train_dict = {}
    for fld in range(nfold):
        test_idx = np.arange(fld*(n_idx/nfold),(fld+1)*(n_idx/nfold))
        test_dict[fld] = index_array[test_idx]
        train_dict[fld] = np.delete(index_array, test_idx)
    return test_dict, train_dict
    

def execute_cross_validation_spatial(train_dict, test_dict, subjic_all, beta_all, method_option='average'):
    """
    """
    assert subjic_all.shape[0] == beta_all.shape[0], "Unmatched subject number."
    assert train_dict.keys() == test_dict.keys(), "Keys need to be matched between train_dict and test_dict."
    nsubj = len(subjic_all)
    rctactmap_all = np.zeros((nsubj, 1, beta_all.shape[-1]))
    for td_idx, td_key in enumerate(test_dict.keys()):
        print('Now perform cross validation in fold {}'.format(td_idx+1)) 
        icmap_test = subjic_all[test_dict[td_key], ...]
        betamap_train = beta_all[train_dict[td_key], ...]
        if method_option == 'average':
            for i,ttd_sid in enumerate(test_dict[td_key]):
                mean_betamap = np.mean(betamap_train,axis=0)
                rctactmap_all[ttd_sid,...] = np.sum(mean_betamap*icmap_test[i,...],axis=0)
    return rctactmap_all
                

def execute_cv_spatial_avginput(actmap, icmap, neighbor_table, train_dict, test_dict, zscore=False, fit_intercept=False):
    """
    """
    nsubj = actmap.shape[0]
    rctactmap_all = np.zeros_like(actmap)
    for td_idx, td_key in enumerate(test_dict.keys()):
        print('Now perform cross validation in fold {}'.format(td_idx+1))
        icmap_test = icmap[test_dict[td_key], ...]
        icmap_train = icmap[train_dict[td_key], ...]
        actmap_test = actmap[test_dict[td_key], ...]
        actmap_train = actmap[train_dict[td_key], ...]
        avgicmap_train = np.mean(icmap_train,axis=0)
        avgactmap_train = np.mean(actmap_train,axis=0)
        avgbetamap, _ = linear_estimate_model_spatial(avgactmap_train, avgicmap_train, neighbor_table, zscore=zscore, fit_intercept=fit_intercept)
        rctactmap_all[test_dict[td_key],0,:] = np.sum(icmap_test*np.tile(avgbetamap[np.newaxis,...], (icmap_test.shape[0],1,1)), axis=1)
    return rctactmap_all 

    
def execute_cross_validation_individual(actmap, icmap, neighbor_table, train_dict, test_dict, zscore=False, fit_intercept=False, concate_option='average'):
    """
    """
    nsubj = actmap.shape[0]
    rctactmap_all = np.zeros_like(actmap)
    for td_idx, td_key in enumerate(test_dict.keys()):
        print('Now perform cross validation in fold {}'.format(td_idx+1))
        icmap_test = icmap[test_dict[td_key], ...]
        icmap_train = icmap[train_dict[td_key], ...]
        actmap_test = actmap[test_dict[td_key], ...]
        actmap_train = actmap[train_dict[td_key], ...]
        betamap_train, _ = linear_estimate_model_individual(actmap_train, icmap_train, neighbor_table, zscore=zscore, fit_intercept=fit_intercept, concate_option=concate_option)
        rctactmap_tmp = np.sum(icmap_test*np.tile(betamap_train[np.newaxis,...],(icmap_test.shape[0],1,1)),axis=1)
        rctactmap_all[test_dict[td_key],0,:] = rctactmap_tmp
    return rctactmap_all 


def local_correlation_bynumber(rctactmap, actmap, thr, option='descend', isabs=True):
    """
    Do local correlation between the activation map and the reconstructed activation map.
    
    Parameter:
    ----------
    rctactmap: reconstructed(predicted) activation map
    actmap: actual activation map
    thr: reserved voxel number
    option: default, 'decend', filter from the highest values
                     'ascend', filter from the lowest values
    isabs: threshold numbers after considering absoluted value.
    """
    nsubj = actmap.shape[0]
    assert rctactmap.shape[0] == nsubj, "Unmatched subject number between rctactmap and actmap."
    if rctactmap.ndim == 3:
        rctactmap = rctactmap.reshape((rctactmap.shape[0], rctactmap.shape[-1]))
    if actmap.ndim == 3:
        actmap = actmap.reshape((actmap.shape[0], actmap.shape[-1]))
    actmap_part = np.zeros((nsubj, thr))
    rctactmap_part = np.zeros((nsubj, thr))
    for i in range(nsubj):
        print('Sorting the largest/smallest {} voxels in subject {}'.format(thr, i+1))
        if isabs is True:
            actmap_tmp = tools.threshold_by_number(np.abs(actmap[i,:]), thr, option=option)
            rctactmap_tmp = tools.threshold_by_number(np.abs(rctactmap[i,:]), thr, option=option)
        else:
            actmap_tmp = tools.threshold_by_number(actmap[i,:], thr, option=option)
            rctactmap_tmp = tools.threshold_by_number(rctactmap[i,:], thr, option=option)   
        actmap_part[i,:] = actmap[i,actmap_tmp!=0]
        rctactmap_part[i,:] = rctactmap[i,rctactmap_tmp!=0]
    r, p = tools.pearsonr(actmap_part, rctactmap_part)
    return r, p
    

def pred_partsim(actmap, rctactmap, mask, method='pearson'):
    """
    Calculate prediction accuracy between activation map and reconstructed activation map in each roi from mask
    """
    assert actmap.shape == rctactmap.shape, "Mismatch between activation map and prediction activation map."
    nsubj = actmap.shape[0]
    masklabel = np.unique(mask[mask!=0])
    r_acc_all = []
    diag_freq = []
    for i, masklbl in enumerate(masklabel):
        # print('Calculating prediction accuracy in region {}'.format(masklbl))
        actmap_rg = actmap[:,mask==masklbl]
        rctactmap_rg = rctactmap[:,mask==masklbl]
        if method == 'pearson':
            r_tmp, _ = tools.pearsonr(actmap_rg, rctactmap_rg)
        r_acc_all.append(np.diag(r_tmp))
        # Outperformed rate in diagonal elements against non-diagonal elements
        argmax_array = np.argmax(r_tmp,axis=0)
        diag_freq.append(1.0*np.sum(argmax_array==np.arange(nsubj))/nsubj)
    r_acc_all = np.array(r_acc_all)
    diag_freq = np.array(diag_freq)
    return r_acc_all, diag_freq


def filter_betamap(betamap_path, meth, thr=[-3,3]):
    """
    filter betamap  
    """
    avgval = _avgbetamap(betamap_path)
    _, beta_nooutlier = tools.removeoutlier(avgval, meth, thr)
    outlier_idx_beta = np.where(np.isnan(beta_nooutlier))[0]
    betamap_path_normal = np.delete(betamap_path, outlier_idx_beta)
    betamap_normal_name = [bpn.split('/')[-1] for bpn in betamap_path_normal]
    return betamap_normal_name


def cal_avgbeta_across_subjects(betamap_path):
    """
    Calculate averaged beta map across subjects
    """
    betamap_tmp = nib.load(betamap_path[0]).get_data()
    avgbetamap = np.zeros_like(betamap_tmp)
    for i, bp in enumerate(betamap_path):
        print('Now reading subject {}'.format(i+1))
        betamap = nib.load(bp).get_data() 
        avgbetamap+=betamap
    avgbetamap = avgbetamap/len(betamap_path)
    return avgbetamap


def _avgbetamap(betamap_path):
    """
    Calculate average value of betamap
    """
    avgval = []
    for i, bp in enumerate(betamap_path):
        print('Loading subject {} for calculating averaged betamap'.format(i+1))
        betamap = nib.load(bp).get_data()
        avgbeta = np.mean(betamap)
        avgval.append(avgbeta)
    avgval = np.array(avgval)
    return avgval


def save_maps_to_nifti(data, filename):
    """
    Save maps to nifti files

    Parameters:
    -----------
    data: Numpy array
    filename: file name 
    """
    outimg = nib.Nifti2Image(data, header=None, affine=None)
    nib.save(outimg, filename)





