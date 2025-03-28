

**DeepAtrophy** is a general purpose library for learning-based tools for brain atrophy/progression estimation.


# Instructions

To use the DeepAtrophy library, either clone this repository and install the requirements listed in `setup.py` or install directly with pip.

```
pip install deepatrophy
```

## Pre-trained models

See list of pre-trained models available [here](data/readme.md#models).

## Training

If you would like to train your own model, you will likely need to customize some of the data loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you have a directory containing training data files in npz (numpy) format. It's assumed that each npz file in your data folder has a `vol` parameter, which points to the numpy image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation (for semi-supervised learning). It's also assumed that the shape of all image data in a directory is consistent.

For a given `/path/to/training/data`, the following script will train the dense network (described in MICCAI 2018 by default) using scan-to-scan registration. Model weights will be saved to a path specified by the `--model-dir` flag.

```
./scripts/tf/train.py /path/to/training/data --model-dir /path/to/models/output --gpu 0
```

Scan-to-atlas registration can be enabled by providing an atlas file with the `--atlas atlas.npz` command line flag. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.


## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `model.h5` trained to register a subject (moving) to an atlas (fixed), we could run:

```
./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
```

This will save the moved image to `warped.nii.gz`. To also save the predicted deformation field, use the `--save-warp` flag. Both npz or nifty files can be used as input/output in this script.


## Testing (measuring Dice scores)

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
./scripts/tf/test.py --model model.h5 --atlas atlas.npz --scans scan01.npz scan02.npz scan03.npz --labels labels.npz
```

Just like for the training data, the atlas and test npz files include `vol` and `seg` parameters and the `labels.npz` file contains a list of corresponding anatomical labels to include in the computed dice score.


# DeepAtrophy Papers

If you use deepatropohy package or some part of the code, please cite (see [bibtex](citations.bib)):

  * DeepAtrophy, provides a progression score in longitudinal images:

    **Regional deep atrophy: Using temporal information to automatically identify regions associated with Alzheimer’s disease progression from longitudinal MRI.**  
    Mengjin Dong, Long Xie, Sandhitsu R. Das, Jiancong Wang, Laura E.M. Wisse, Robin deFlores, David A. Wolk, Paul A. Yushkevich
    Imaging Neuroscience. 2024. [eprint arxiv:2304.04673](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00294/124226)


  * Regional Deep Atrophy, provides a binary attention map in addition to the progression score:  

    **DeepAtrophy: Teaching a neural network to detect progressive changes in longitudinal MRI of the hippocampal region in Alzheimer's disease**
    Mengjin Dong, Long Xie, Sandhitsu R. Das, Jiancong Wang, Laura E.M. Wisse, Robin deFlores, David A. Wolk, Paul A. Yushkevich, for the Alzheimer's Disease Neuroimaging Initiative 
    NeuroIamge, 243, 118514. 2021. (https://www.sciencedirect.com/science/article/pii/S1053811921007874)

# Contact:

For any problems or questions please [open an issue](https://github.com/MengjinDong/DeepAtrophyPackage/issues/new?template=Blank+issue) for code problems/questions or [start a discussion](https://github.com/MengjinDong/DeepAtrophyPackage/discussions) for general registration/voxelmorph question/discussion.
