import numpy as np
import readmhd

rawdir = "/export2/PET-CT_iso3mm/"
moddir = "/home/kamesawa/data/"

abnormal_lung_idx = [2,3,4,5,6,7,8,10,11,13,17,18,22,24,26,27,32,99,122,147,178,227,
                     244,245,246,247,248,264]

def raw(NL, index, filename):
    """
    param NL: 'N' or 'L'
    param index: index of data, int or list of int
    param filename: kind of filename
    """
    assert NL in ('N', 'L')

    if type(index) is int:
        return readmhd.read(rawdir+"%s%05d/%s" % (NL, index, filename)).vol

    return [readmhd.read(rawdir+"%s%05d/%s" % (NL, i, filename)).vol for i in index]

def raw_PET(NL, index):
    """
    param NL: 'N' or 'L'
    param index: index of data, int or list of int
    """
    return raw(NL, index, "PETiso.mhd")

def raw_CT(NL, index):
    """
    param NL: 'N' or 'L'
    param index: index of data, int or list of int
    """
    return raw(NL, index, "CTiso.mhd")

def raw_lung_mask(NL, index):
    """
    param NL: 'N' or 'L'
    param index: index of data, int or list of int
    """
    return raw(NL, index, "LungAreaIso.mhd")

def raw_lung_lesion_mask(index):
    """
    param NL: 'N' or 'L'
    param index: index of data, int or list of int
    """
    return raw('L', index, "LungLesionMask.mhd")
    

def block_filename(size, NL, index):
    if type(index) is int:
        return moddir+"CTLung%d/block/%s%05d.npy" % (size, NL, index)
    else:
        return [moddir+"CTLung%d/block/%s%05d.npy" % (size, NL, i) for i in index]

def feature_filename(size, NL, index):
    if type(index) is int:
        return moddir+"CTLung%d/feature/%s%05d.npy" % (size, NL, index)
    else:
        return [moddir+"CTLung%d/feature/%s%05d.npy" % (size, NL, i) for i in index]

def pet_filename(NL, index):
    if type(index) is int:
        return moddir+"PETLung/%s%05d.npy" % (NL, index)
    else:
        return [moddir+"PETLung/%s%05d.npy" % (NL, i) for i in index]

def z_mhd_filename(NL, index, modeldir):
    return modeldir+'%s%05d_z.mhd' % (NL, index)

def mean_mhd_filename(NL, index, modeldir):
    return modeldir+'%s%05d_mean.mhd' % (NL, index)

def var_mhd_filename(NL, index, modeldir):
    return modeldir+'%s%05d_var.mhd' % (NL, index)

def z_img(NL, index, modeldir):
    return readmhd.read(z_mhd_filename(NL, index, modeldir)).vol

def random_mask_filename(NL, index, rsize, number):
    filename = moddir+("mask%d/%s%05d_%d.npy" % (rsize, NL, index, number))
    import os
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    return filename

def random_feature_filename(fsize, rsize, number):
    return moddir+("CTLung%d/feature/random%d_%d.npy" %
                   (fsize, rsize, number))

def random_PET_filename(rsize, number):
    return moddir+"PETLung/random%d_%d.npy" % (rsize, number)
