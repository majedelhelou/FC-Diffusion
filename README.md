# Fuzzy-Conditioned Diffusion

### Fuzzy-conditioned diffusion and diffusion projection attention applied to facial image correction

**Author**: [Majed El Helou](https://majedelhelou.github.io/) (Media Technology Center ETH Zurich - [MTC](https://mtc.ethz.ch/))

**Paper**: [arxiv preprint](https://arxiv.org/abs/2011.01406)

**Publisher**: IEEE International Conference on Image Processing

![Python 3.9.0](https://img.shields.io/badge/python-3.9.0-green.svg?style=plastic)
![pytorch 1.13.1](https://img.shields.io/badge/pytorch-1.13.1-green.svg?style=plastic)
![CUDA 11.7](https://img.shields.io/badge/cuda-11.7-green.svg?style=plastic)

**Key take-aways**: condition image diffusion on an input image, to varying degrees and per spatial location. We use this, in conjunction with diffusion-projection-based anomaly detection, for facial image correction. Generally, this fuzzy conditioning also enables control over the degree of prior-based hallucination (c.f. [BIGPrior TIP'22](https://github.com/majedelhelou/BIGPrior)). 


> **Abstract:**
Image diffusion has recently shown remarkable performance in image synthesis and implicitly as an image prior. Such a prior has been used with conditioning to solve the inpainting problem, but only supporting binary user-based conditioning. 
>
>We derive a fuzzy-conditioned diffusion, where implicit diffusion priors can be exploited with controllable strength. Our fuzzy conditioning can be applied pixel-wise, enabling the modification of different image components to varying degrees. Additionally, we propose an application to facial image correction, where we combine our fuzzy-conditioned diffusion with diffusion-derived attention maps. Our map estimates the degree of anomaly, and we obtain it by projecting on the diffusion space. We show how our approach also leads to interpretable and autonomous facial image correction.

## Repo setup
### Code
```bash
git clone https://github.com/majedelhelou/FC-Diffusion.git
```

### Environment
```bash
# Create a conda env with python 3.9 and activate it
# The code was tested wit PyTorch 1.13.1 and cuda 11.7
# mpi4py installation causes errors, it is excluded from reqs; install manually through conda
conda create -n FCD python=3.9 anaconda
conda activate FCD
pip install requirements.txt
conda install -c conda-forge mpi4py
```

### Data download
Download the CelebA pretrained model checkpoint, pre-computed zscores, and some data to experiment with:
```bash
pip install --upgrade gdown && bash ./download.sh
```

### Structure:
```bash
.  
├── data
│   ├── pretrained          # Pretrained checkpoints (to be created and downloaded)
│   ├── samples             # Face example (in "gts") with a user-defined mask (in "gt_keep_masks")
│   ├── test_set            # A test set that is a small subset of CelebA for illustration
│   └── zscores             # Pre-computed deviation-map z-scores, for diffusion depths (300,400,500,600)
├── FuzzyInpaint            # Code for fuzzy-guided diffusion, builds on guided-diffusion and repaint (see LICENSES)
├── LICENSES                # Our license and the licenses of underlying code
├── log                     # Stores all intermediate results and final outputs and parameters of any experiment
├── create_zscore           # Can be ignored as the zscores are already provided pre-computed for CelebA
├── dataset_multi_inversion # Can also be ignored, used for zscore map intermediate computations
├── download                # Run to get checkpoints and data
├── main                    # All experiments can be controlled through main
└── requirements            # All dependencies (except mpi4py that has issues, refer to Environment section)
```


## Experiments
You can test 3 features with this code:
1. synthesize a new face with fuzzy conditioning on an input image
2. synthesize a new face with fuzzy conditioning on an input + inpaint a mask area
3. use fuzzy conditioning with diffusion-projection prediction as the guidance scale

For the third feature, you can also enable self degradation to self-damage the image.

### 1. Fuzzy conditioning
To run image synthesis with fuzzy conditioning (weight 0.07) on the images from `--gt_path` and save in `log/replace/`:
```bash
python main.py --gt_path ../data/test_set --out_path ../log/replace/ --mask_override replace --mask_s 0.07 
```

### 2. Fuzzy conditioning + inpainting mask
To run image synthesis with inpainting inside the mask in `--mask_path` and fuzzy conditioning outside the mask (weight 0.07) on the images from `--gt_path` and save in `log/modulate/`:
```bash
python main.py --gt_path ../data/samples/gt --mask_path ../data/samples/mask --out_path ../log/modulate/ --mask_override modulate --mask_s 0.07
```
>The mask and ground-truth paths should contain images with matching names.

### 3. Fuzzy conditioning applied to correction
To apply fuzzy conditioning to image correction, you can run the following command on `test_set`, with self_degrade set to True. This will:
1. randomly degrade the image (random areas are randomly degraded in RGB)
2. re-project the image onto four different depths of a diffusion process
3. compute statistics deviations at each depth, and fuse them into one map
4. use this map as fuzzy guidance for conditioning the diffusion synthesis
```bash
python main.py --gt_path ../data/test_set --mask_override get_zscore --out_path ../log/self --self_degrade True
```
The results are stochastic, and different images inherently deviate from the CelebA prior to different extents (CelebA also contains unclean images). To improve results, you can manually adjust the mapping from *projection error statistics* to the diffusion *fuzzy-conditioning weight* with the following parameters; the defaults that are used for all results in the paper are:
```bash
{as above ^} --ood_lambda 0.1 --ood_expon 2 --lower_bound 1
```

This will generate the following:
- `err` :            contains the diffusion-reprojection deviations
- `errzscore` :      contains the stat normalized deviations
- `gt_image` :       contains the self-degraded images (or inputs if *self-degrade=False*)
- `gt_masked` :      contains the images with the applied (re-scaled) conditioning maps
- `inpainted` :      contains the final output images
- `mask_nolambda` :  contains the original diffusion maps that are not re-scaled
- `mask_used` :      contains the re-scaled masks that are applied to get `gt_masked`
- `projected` :      contains each of the diffusion re-projections of the inputs
- `args.json` :      holds all the parameters used in the experiment


### Data domain
Note that the provided checkpoint model is trained on facial data. For data in other domains, a different diffusion prior should be learned. Please refer to [guided-diffusion](https://github.com/openai/guided-diffusion) for training from scratch. 


## Citation
```bibtex
@inproceedings{el2023fuzzy,
    title     = {Fuzzy-conditioned diffusion and diffusion projection attention applied to facial image correction},
    author    = {El Helou, Majed},
    booktitle = {IEEE International Conference on Image Processing (ICIP)},
    year      = {2023}
}
```

## Acknowledgement

This work was supported by Align Technology, Ringier, TX Group, NZZ, SRG, VSM, Viscom, and the ETH Zurich Foundation.

The repository builds on:

https://github.com/openai/guided-diffusion

https://github.com/hojonathanho/diffusion

https://github.com/andreas128/RePaint

