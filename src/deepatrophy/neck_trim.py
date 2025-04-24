import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
import SimpleITK as sitk
import io
import re
import numpy as np
import pathlib

def trim_neck_rf(input_image, trimmed_image, head_length=180, clearance=10, mask_file=None, workdir=None, debug=False):
    """
    Brain MRI neck removal tool

    Parameters:
    input_image (str): Path and filename to the input image.
    output_image (str): Path and filename to the output image.
    head_length (int): Length (sup-inf) of the head that should be captured. Default is 180 mm.
    clearance (int): Clearance above the head that should be captured. Default is 10 mm.
    mask_file (str): Location to save the mask used for the computation. Default is None.
    workdir (str): Location to store intermediate files. Default is system temp directory.
    verbose (bool): Enable verbose/debug mode. Default is False.
    """  

    # Set verbose mode
    verbose = "-verbose" if debug else ""

    # Create a temporary directory if not provided
    if workdir is None:
        workdir = tempfile.mkdtemp()
    else:
        os.makedirs(workdir, exist_ok=True)

    print(f"Working directory: {workdir}")

    # Create a landmark file for background/foreground segmentation
    landmarks_content = """40x40x40% 1
60x40x40% 1
40x60x40% 1
40x40x60% 1
60x60x40% 1
60x40x60% 1
40x60x60% 1
60x60x60% 1
3x3x3% 2
97x3x3% 2
3x97x3% 2
3x3x97% 2
97x97x3% 2
97x3x97% 2
3x97x97% 2
97x97x97% 2"""
    
    with open(os.path.join(workdir, "landmarks.txt"), 'w') as f:
        f.write(landmarks_content)

    # Intermediate file paths
    ras_trimmed = os.path.join(workdir, "trimmed_ras.nii.gz")
    ras_result = os.path.join(workdir, "result_ras.nii.gz")
    rfmap = os.path.join(workdir, "rfmap.nii.gz")
    levelset = os.path.join(workdir, "levelset.nii.gz")
    mask = os.path.join(workdir, "mask.nii.gz")
    slab = os.path.join(workdir, "slab.nii.gz")

    c = Convert3D()

    img = sitk.ReadImage(input_image)       # Load image with SimpleITK
    c.push(img)                                     # Put image on the stack
    c.execute('-swapdim RAS -smooth-fast 1vox -resample-mm 2x2x2mm')  # Make a thumbnail
    dsample_ras = c.peek(-1)                          # Get the last image on the stack
    c.push(dsample_ras)                              # Put the image on the stack
    c.push(dsample_ras)                              # duplicate the image on stack
    c.execute('-steig 2.0 4x4x4')
    c.push(dsample_ras)
    c.execute(f'-dup -scale 0 -lts %s 15' % os.path.join(workdir, "landmarks.txt")) 
    samples = c.peek(-1)
    c.execute('-rf-param-patch 1x1x1 -rf-train /tmp/myforest.rf')
    c.pop()
    c.execute('-rf-apply /tmp/myforest.rf')
    rfmap = c.peek(-1)

    sitk.WriteImage(dsample_ras, os.path.join(workdir, "dsample_ras.nii.gz") )   # Save the image to disk
    sitk.WriteImage(samples, os.path.join(workdir, "samples.nii.gz") )   # Save the image to disk
    sitk.WriteImage(rfmap, os.path.join(workdir, "rfmap.nii.gz") )   # Save the image to disk

    c = Convert3D()

    c.push(rfmap)
    c.execute('-as R -smooth-fast 1vox -resample 50% -stretch 0 1 -1 1 -dup')
    c.push(samples)
    c.execute('-thresh 1 1 1 -1 -reslice-identity -levelset-curvature 5.0 -levelset 300')
    levelset = c.peek(-1)
    c.execute('-insert R 1 -reslice-identity -thresh 0 inf 1 0')
    mask = c.peek(-1)

    sitk.WriteImage(levelset, os.path.join(workdir, "levelset.nii.gz") )   # Save the image to disk
    sitk.WriteImage(mask, os.path.join(workdir, "mask.nii.gz") )   # Save the image to disk

    c = Convert3D()
    s = io.StringIO()  # Create an output stream

    c.execute(f'{os.path.join(workdir, "mask.nii.gz")} -info-full', out=s)     # Run command, capturing output to s
    # print(f'c3d output: {s.getvalue()}') 

    line = s.getvalue().split('\n')[1]        # Take first line from output
    dimensions = re.findall(r'\d+', line)
    dimensions = [int(num) for num in dimensions]
    dimx, dimy, dimz = dimensions

    # print(dimensions)

    # Amount to trim: 18cm = 90vox
    regz = head_length // 2 + clearance // 2
    trim = clearance // 2

    # Translate the RAI code into a region specification
    regcmd = f"0x0x0vox {dimx}x{dimy}x{regz}vox"

    c = Convert3D()

    c.push(mask)
    c.execute(f'-dilate 1 {dimx}x{dimy}x0vox -trim 0x0x{trim}vox -region {regcmd} -thresh -inf inf 1 0')
    slab = c.peek(-1)
    c.execute('-popas S')
    c.push(img)
    c.execute('-as I -int 0 -push S -reslice-identity -trim 0vox -as SS')
    slab_src = c.peek(-1)
    c.push(img)
    c.execute('-reslice-identity -o %s' % trimmed_image)
    target = c.peek(-1)

    sitk.WriteImage(slab_src, os.path.join(workdir, "slab_src.nii.gz") )   # Save the image to disk
    # sitk.WriteImage(target, trimmed_image )   # Save the image to disk


    # If mask is requested, reslice mask to target space
    if mask_file:
        subprocess.run(f"c3d {input_image} {levelset} -reslice-identity -thresh 0 inf 1 0 -o {mask_file}", shell=True)

    print(f"Trimmed image saved to {trimmed_image}")


# The main program launcher
class NeckTrimLauncher:

    def __init__(self, parse):

        # Add the arguments
        parse.add_argument('--input-image', metavar='input_image', type=pathlib.Path, 
                        help='filename of input image')
        parse.add_argument('--trimmed-image', metavar='trimmed_image', type=pathlib.Path, 
                        help='Specify the filename of the trimmed image')
        parse.add_argument('--head-length', metavar='head_length', type=int, default=180,
                        help='Length (sup-inf) of the head that should be captured. Default is 180 mm.')
        parse.add_argument('--clearance', metavar='clearance', type=int, default=10,
                           help='Clearance above the head that should be captured. Default is 10 mm.')
        parse.add_argument('--mask-file', metavar='mask_file', type=str, 
                        help='Location to save the mask used for the computation. Default is None.')
        parse.add_argument('--workdir', type=str, 
                        help='Location to store intermediate files. Default is system temp directory.')
        parse.add_argument('--debug', action='store_true', help='Enable verbose/debug mode. Default is False.')
        
        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        trim_neck_rf(args.input_image, args.trimmed_image, workdir=args.workdir, debug=args.debug)



