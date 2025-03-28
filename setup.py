from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="deepatrophy",
    version="0.0.4",
    description="DeepAtrophy: a Longitudinal MRI Analysis Package for Brain Atrophy Estimation",
    url='https://github.com/MengjinDong/DeepAtrophyPackage', 
    keywords=['atrophy', 'longitudinal', 'deep learning', 'temporal interence', 'mri'],
    package_dir={"": "src"},
    packages=find_packages("src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mengjin Dong",
    author_email="dmengjin@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    install_requires=["bson >= 0.5.10",
                        "numpy",
                        "torch",
                        "torchvision",
                        "tensorboardX",
                        "pykeops",
                        "pymeshlab",
                        "vtk",
                        "SimpleITK",
                        "geomloss",
                        "statsmodels",
                        # "monai",
                        "picsl_c3d>=1.4.2.1",
                        "picsl_greedy>=0.0.5",
                        # "picsl_cmrep>=1.0.2.0",
                        'scikit-image',
                        'h5py',
                        'numpy',
                        'scipy',
                        'nibabel',
                        'neurite',
                        "nnunetv2"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.6",
)
