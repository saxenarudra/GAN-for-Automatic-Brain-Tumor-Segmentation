
import numpy as np

from functools import reduce
#for Hausdorff
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure

from data_processing import graph_io

'''
A collection of functions to evaluate the model's predictions.
Largely focused on the Sorenson Dice coefficient and 95th percentile of the Hausdorff distance,
which are the metrics used by BraTS.
'''

HEALTHY=0
EDEMA=1
NET=2
ET=3


def count_node_labels(preds_or_label):
    count = np.zeros(4)
    for i,j in zip(*np.unique(preds_or_label,return_counts=True)):
        count[i]=j
    return count

#calculate nodewise Dice score for WT CT and ET for a single brain.
#Expects two 1D vectors of integers
def calculate_node_dices(pred,label):
    p,l=pred,label
    wt_pred = np.where(p==HEALTHY,0,1)
    wt_lab = np.where(l==HEALTHY,0,1)
    wt_dice = calculate_dice_from_logical_array(wt_pred,wt_lab)

    ct_pred = np.isin(p,[NET,ET]).astype(int)
    ct_lab = np.isin(l,[NET,ET]).astype(int)
    ct_dice = calculate_dice_from_logical_array(ct_pred,ct_lab)

    at_pred = np.where(p==ET,1,0)
    at_lab = np.where(l==ET,1,0)
    at_dice = calculate_dice_from_logical_array(at_pred,at_lab)

    return [wt_dice,ct_dice,at_dice]

#Computes the percentage of voxels that are correctly predicted, optionally excluding tissue labelled as healthy in the ground truth
#Useful for computing Achievable segmentation accuracy, for example.
def compute_accuracy(supervoxel_labels,grd_truth,include_healthy=True):
    assert(grd_truth.shape==supervoxel_labels.shape)
    if(include_healthy):
        num_voxel=reduce(lambda x, y: x*y, grd_truth.shape)
        num_correspondenc = np.sum(grd_truth==supervoxel_labels)
    else:
        nonzero_masks = grd_truth!=0
        num_voxel=np.sum(nonzero_masks)
        num_correspondenc=np.sum((grd_truth==supervoxel_labels)*nonzero_masks)
    return num_correspondenc/num_voxel


#calculates voxelwise WT,CT,ET Dice and HD95 for a single brain.
#Expects two n-D (only tested with 2D or 3D) arrays of integers
def calculate_brats_metrics(predicted_voxel,true_voxel):
    wt_pred = np.where(predicted_voxel==HEALTHY,0,1)
    wt_gt = np.where(true_voxel==HEALTHY,0,1)
    wt_dice = calculate_dice_from_logical_array(wt_pred,wt_gt)
    wt_hd = calculate_hd95_from_logical_array(wt_pred,wt_gt)

    ct_pred = np.isin(predicted_voxel,[NET,ET]).astype(int)
    ct_gt = np.isin(true_voxel,[NET,ET]).astype(int)
    ct_dice = calculate_dice_from_logical_array(ct_pred,ct_gt)
    ct_hd = calculate_hd95_from_logical_array(ct_pred,ct_gt)

    at_pred = np.where(predicted_voxel==ET,1,0)
    at_gt = np.where(true_voxel==ET,1,0)
    at_dice = calculate_dice_from_logical_array(at_pred,at_gt)
    at_hd = calculate_hd95_from_logical_array(at_pred,at_gt)

    return [wt_dice,ct_dice,at_dice,wt_hd,ct_hd,at_hd]

#wrapper around hd95 function that handles the case where one or more labels are missing from the ground truth or prediction.
def calculate_hd95_from_logical_array(preds,gts):
    try:
        hd_95 = hd95(preds,gts)
    #no positive (1) voxels present in one of the inputs
    except RuntimeError as e:
        #then this label isnt present in either the prediction or gt, so assign a distance of zero since the pred was correct
        if(not 1 in preds and not 1 in gts):
            hd_95 = 0
        #return maximal distance
        else:
            hd_95 = 300
    finally:
        return hd_95

#Each tumor region (WT,CT,ET) is binarized for both the prediction and ground truth and then the overlapping volume is calculated.
def calculate_dice_from_logical_array(binary_prediction,binary_grd_truth):
    true_positive = np.logical_and(binary_prediction==1, binary_grd_truth==1)
    false_positive = np.logical_and(binary_prediction==1, binary_grd_truth==0)
    false_negative = np.logical_and(binary_prediction==0, binary_grd_truth==1)
    tp,fp,fn=np.count_nonzero(true_positive),np.count_nonzero(false_positive),np.count_nonzero(false_negative)
    #the case where no such labels exist (only really relevant for ET case)
    if(tp+fp+fn)==0:
        return 1
    return (2*tp)/(2*tp+fp+fn)



#Copied from medpy package

def hd95(results, references, voxel_spacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd_1 = __surface_distances(results, references, voxel_spacing, connectivity)
    hd_2 = __surface_distances(references, results, voxel_spacing, connectivity)
    hd_95 = np.percentile(np.hstack((hd_1, hd_2)), 95)
    return hd_95




def __surface_distances(results, references, voxel_spacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    results = np.atleast_1d(results.astype(np.bool))
    references = np.atleast_1d(references.astype(np.bool))
    if voxel_spacing is not None:
        voxel_spacing = _ni_support._normalize_sequence(voxel_spacing, results.ndim)
        voxel_pacing = np.asarray(voxel_spacing, dtype=np.float64)
        if not voxel_spacing.flags.contiguous:
            voxel_spacing = voxel_spacing.copy()
            
    # binary structure
    footprints = generate_binary_structure(results.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(results):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(references):
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_borders = results ^ binary_erosion(results, structure=footprints, iterations=1)
    reference_borders = references ^ binary_erosion(references, structure=footprints, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_borders, sampling=voxel_spacing)
    sd = dt[result_borders]
    
    return sd


def print_metrics(loss,dsc,hd):
    print(f"Loss : {loss}")
    print (f"WT Dice : {dsc[0]}")
    print (f"CT Dice : {dsc[1]}")
    print (f"AT Dice : {dsc[2]}")
    print (f"WT HD95 : {hd[0]}")
    print (f"CT HD95 : {hd[1]}")
    print (f"AT HD95 : {hd[2]}")


#not used, keeping for reference
#micro average is weighted by prevalence of a given label, macro is a non-weighted mean,
#and None is dont average(return a vector of 3 results)
'''
def calculate_dice_score_by_class(saved_graphs_folder,scan_to_pred_array,average="none"):
    if(average not in ["macro","micro","none"]):
        raise Exception("specified average must be one of macro, micro or none")
    all_dices=[]
    for scan,preds in scan_to_pred_array.items():
        #first get preds for every voxel in image
        predicted_brain_voxels = graphio.convert_supervoxel_labels_to_voxel(f"{saved_graphs_folder}{scan}",preds)
        ground_truth = graphio.read_cropped_labels(f"{saved_graphs_folder}{scan}")
        label_counts = np.unique(ground_truth,return_counts=True)[1][1:]
        dices_per_label = []
        for label in [1,2,4]:
            true_positives = np.logical_and(predicted_brain_voxels==label, ground_truth==label)
            false_positives = np.logical_and(predicted_brain_voxels==label, ground_truth!=label)
            false_negatives = np.logical_and(predicted_brain_voxels!=label, ground_truth==label)
            tp,fp,fn=np.count_nonzero(true_positives),np.count_nonzero(false_positives),np.count_nonzero(false_negatives)
            dices_per_label.append(2*tp/(2*tp+fp+fn))
        if(average=='micro'):
            all_dices.append(np.dot(dices_per_label,label_counts/sum(label_counts)))
        elif(average=='macro'):
            all_dices.append(np.mean(dices_per_label))
        elif(average=='none'):
            all_dices.append(dices_per_label)
    if(average=='none'):
        return np.mean(all_dices,axis=1)
    else:
        return np.mean(all_dices)
'''
