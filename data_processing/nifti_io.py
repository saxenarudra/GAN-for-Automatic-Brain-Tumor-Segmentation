import nibabel as nib
import numpy as np
import os

'''
A collection of helper functions to read and write niftii files
'''

#Given a directory and a list of modality extensions, scans that directory for any files ending in those extensions
#and returns them as a single stacked numpy float array.
#Assumes the modalities all belong to the same patient, have already been co-registered, and have the same resolution.
def read_in_patient_sample(scan_Dir,modality_ext):
    num_modality=len(modality_ext)
    modality_img = []
    for root, _, files in os.walk(scan_Dir):
        for exts in modality_ext:
            for file in files:
                if file.endswith(exts):
                    f_path = os.path.join(root, file)
                    mod_img = nib.load(f_path)
                    #data is actually stored as int16
                    img_data = np.array(mod_img.dataobj,dtype=np.float32)
                    modality_img.append(img_data)
    #check that all the modalities were present in the folder
    assert(len(modality_img)==num_modality)

    patient_samples = np.stack(modality_img,3) if num_modality>1 else modality_img[0]
    return patient_samples


def read_in_labels(scan_Dir, label_exts):
    for file in os.listdir(scan_Dir):
        if file.endswith(label_exts):
            labels_nib = nib.load(scan_Dir + os.sep + file)
            # potentially also return affine if they are different between images (which they are not for brats)
            return np.array(labels_nib.dataobj, dtype=np.int16)
    raise FileNotFoundError(f"Label image not found in folder: {scan_Dir}")


#Uses the BraTS standard affine matrix to save a numpy array into a nifti file.
#This will likely break any data that is not curated by BraTS or TCIA
def save_as_nifti(image,fi):
    affine_mat = np.array([
        [ -1.0,  -0.0,  -0.0,  -0.0],
        [ -0.0,  -1.0,  -0.0, 239.0],
        [  0.0,   0.0,   1.0,   0.0],
        [  0.0,   0.0,   0.0,   1.0],
        ])
    image = nib.nifti1.Nifti1Image(image, affine_mat)
    nib.save(image, fi)

def read_nifti(fi,data_type):
    nib_object = nib.load(fi)
    return np.array(nib_object.dataobj,dtype=data_type)



