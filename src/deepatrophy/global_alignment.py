import os
import random
import shutil
import time
import warnings
import numpy as np
import csv
import os
import subprocess
import tempfile
from picsl_c3d import Convert3D 
from picsl_greedy import Greedy3D
import SimpleITK as sitk
import io
import re
# from scipy.io import savemat
# from scipy.linalg import sqrtm
import scipy
import scipy.linalg
import pathlib
import h5py

def global_align_rigid(source_image, target_image, mask_file=None, workdir=None, prefix=None, debug=False):
    
    g = Greedy3D()

    img_fixed = sitk.ReadImage(target_image)
    img_moving = sitk.ReadImage(source_image)

    # greedy -d 3 \
    #     -ia-image-centers -a \
    #     -i $ALOHA_BL_MPRAGE $ALOHA_FU_MPRAGE \
    #     -m NCC 4x4x4 -n 500x250x100x0 \
    #     -o $WDINIT/mprage_long_RAS.mat
    
    # greedy -d 3 \
    #     -rf $ALOHA_BL_MPRAGE \
    #     -rm $ALOHA_FU_MPRAGE $WDINIT/mprage_fu_to_bl_resliced.nii.gz \
    #     -r $WDINIT/mprage_long_RAS.mat
    
    # used rigid registration. If use affine, then need to save the affine matrix and use it for later volume calculation.
    g.execute('-d 3 -ia-image-centers -a -dof 6 '
              '-i my_fixed my_moving '
              '-m NCC 4x4x4 -n 500x250x100x0 '
              f'-o {workdir}/{prefix}_mprage_long_RAS.mat ', # 
              my_fixed = img_fixed, my_moving = img_moving) # , my_rigid = None
    
    # Read affine transformation into numpy array
    with open(f'{workdir}/{prefix}_mprage_long_RAS.mat', 'r') as file:
        file_content = file.read()

    data = [float(value) for value in file_content.split()]
    mat_rigid = np.array(data).reshape(4, 4)

    # calculate hw matrix and its inverse
    mat_rigid_hw = scipy.linalg.sqrtm(mat_rigid)

    if np.iscomplexobj(mat_rigid_hw):
        mat_rigid_hw = np.real(mat_rigid_hw)

    mat_rigid_hw_inv = np.linalg.inv(mat_rigid_hw)

    # print the affine transformation matrix
    print('The affine transform matrix hw is ', mat_rigid_hw)
    print('The affine transform matrix hw_inv is ', mat_rigid_hw_inv)

    # save hw matrix and its inverse
    np.savetxt(f'{workdir}/{prefix}_mprage_long_RAS_hw.mat', mat_rigid_hw, fmt='%.8f')
    np.savetxt(f'{workdir}/{prefix}_mprage_long_RAS_hw_inv.mat', mat_rigid_hw_inv, fmt='%.8f')

    # reslice both baseline and followup images to the halfway space
    g.execute('-rf my_fixed -rm my_moving my_resliced_fu '
          f'-r {workdir}/{prefix}_mprage_long_RAS_hw.mat',
          my_resliced_fu = None)
    
    g.execute('-rf my_fixed -rm my_fixed my_resliced_bl '
              f'-r {workdir}/{prefix}_mprage_long_RAS_hw_inv.mat',
              my_resliced_bl = None)

    # save the rigid transformation and resliced image
    sitk.WriteImage(g['my_resliced_fu'], os.path.join(workdir, prefix + '_resliced_fu.nii.gz'))
    sitk.WriteImage(g['my_resliced_bl'], os.path.join(workdir, prefix + '_resliced_bl.nii.gz'))

    return mat_rigid, g['my_resliced_bl'], g['my_resliced_fu']



def get_mask(target_image, template_image, template_mask, workdir=None, prefix=None, debug=False):

    # get bounding box from registration to the template image.
    # source and target images are registered to the halfway space in the previous code.
    # step 1. register the target (baseline) image to the template image, and obtain the translation. Can use rigid registration.
    # step 2. apply the translation of the template segmentation to the target image, and obtain the bounding box.
    # step 3. obtain the segmentation on the target image.

    # source image = followup image
    # target image = baseline image

    # load 
    g = Greedy3D()

    img_fixed = sitk.ReadImage(target_image)
    img_moving = sitk.ReadImage(template_image)
    img_seg = sitk.ReadImage(template_mask)
    
    # used affine registration.
    g.execute('-d 3 -ia-image-centers -a '
              '-i my_fixed my_moving '
              '-m NCC 4x4x4 -n 500x250x100x0 '
              '-o my_affine ', # mprage_long_RAS.mat
              my_fixed = img_fixed, my_moving = img_moving, my_affine = None)
    
    affine_matrix = g['my_affine']

    print('The affine transform matrix is ', affine_matrix)

    # use deformable registration.
    g.execute('-d 3 '
            '-i my_fixed my_moving '
            '-m NCC 4x4x4 -n 40x20x10x0 '
            '-o my_warp ',
            my_fixed = img_fixed, my_moving = img_moving, my_warp = None)
              
    # apply segmentation to the registered image
    g.execute(f'-rf my_fixed -rm my_moving {workdir}/{prefix}_resliced.nii.gz '
              '-ri LABEL 0.2vox '
              f'-rm my_seg {workdir}/{prefix}_resliced_seg.nii.gz '
              '-r my_warp my_affine ',
              my_seg = img_seg)
    
    # crop segmented images and obtain bounding box with c3d
    c = Convert3D()

    resliced_seg = sitk.ReadImage(f'{workdir}/{prefix}_resliced_seg.nii.gz')
    c.push(resliced_seg)                                     # Put image on the stack
    c.execute(f'-trim 10vox -o {workdir}/{prefix}_cropped_seg.nii.gz')  # Make a thumbnail
    c.push(img_fixed)                                     
    c.execute(f'-reslice-identity -o {workdir}/{prefix}_cropped_bl.nii.gz')  

    return None


# The main program launcher
class GlobalAlignmentLauncher:

    def __init__(self, parse):

        # Add the arguments
        parse.add_argument('--baseline-image', metavar='baseline_image', 
                           type=pathlib.Path, 
                           help='filename of the baseline image')
        parse.add_argument('--followup-image', metavar='followup_image', 
                           type=pathlib.Path, 
                           help=' filename of the followup image')
        parse.add_argument('--prefix', metavar='prefix', type=str, 
                           help='Prefix for the output files. Default is None.')
        parse.add_argument('--template-image', metavar='template_image', 
                           type=pathlib.Path, 
                           default="/data/mengjin/DeepAtrophyPackage/DeepAtrophy/template/template.nii.gz",
                           help='Template file of the hippocampus image.')
        parse.add_argument('--template-mask', metavar='template_mask', 
                           type=pathlib.Path, 
                           default="/data/mengjin/DeepAtrophyPackage/DeepAtrophy/template/refspace_meanseg_left.nii.gz",
                           help='Template file of the hippocampus mask.')
        parse.add_argument('--get-ROI', action='store_true',
                           help='Whether to get the region of interest. Default is False.')
        parse.add_argument('--workdir', type=str, 
                           help='Location to store intermediate files. Default is system temp directory.')
        parse.add_argument('--debug', action='store_true', 
                           help='Enable verbose/debug mode. Default is False.')
        parse.add_argument('-s', '--side', type=str, choices=['left', 'right'], 
                           help='Side of the brain', default='left')
        
        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        global_align_rigid(args.followup_image, args.baseline_image, workdir=args.workdir, prefix=args.prefix, debug=args.debug)

        if args.get_ROI:
            get_mask(args.baseline_image, args.template_image, args.template_mask, workdir=args.workdir, prefix=args.prefix, debug=args.debug)

        print("finished runnning GlobalAlignmentLauncher.run")





