# One-Step Optimal Transport Flow for Anisotropic MRI Super-Resolution

Official PyTorch implementation of the IEEE TMI paper "One-Step Optimal Transport Flow for Anisotropic MRI Super-Resolution". The complete code will be released upon publication of the paper.

We provide .gif and .mp4 files of super-resolution results ($R$=7) in the "figs" folder. We recommend downloading the .mp4 file for easier frame-by-frame viewing. Examples of the generated MR volumes via one-step generation:

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/HCP_7mm.gif) 

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/OASIS_7mm.gif)

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/SKM-TEA_7mm.gif)

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/PRLHR-CT_7mm.gif)



## The proposed method

The overview of OS-DDPM are illustrated in the following figure:

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/Framework.png)

## Datasets

Three public datasets are used to verify OS-DDPM, including the [HCP dataset](https://balsa.wustl.edu/), the [OASIS dataset](https://sites.wustl.edu/oasisbrains/home/oasis-3/), the [SKM-TEA dataset](https://stanfordaimi.azurewebsites.net/datasets/), and the  [PRLHR-CT dataset](https://github.com/smilenaxx/RPLHR-CT)

## Results
Qualitative results on the HCP dataset ($R$=7): 

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/HCP.png)

Qualitative results on the OASIS dataset ($R$=7): 

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/OASIS.png)

Qualitative results on the SKM-TEA dataset ($R$=7): 

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/SKM-TEA.png)

Qualitative results on the PRLHR-CT dataset ($R$=7): 

![](https://github.com/yanaynghui1998/OSOT-Flow/blob/main/figs/PRLHR-CT.png)

## Dependencies

We recommend using a [conda](https://github.com/conda-forge/miniforge#mambaforge) environment to install the required dependencies. You can create and activate such an environment called `OSOT-Flow` by running the following commands:

```python
mamba env create -f environment.yml
mamba activate OSOT-Flow
```

## Pre-training OT-ODE 

Run the following command. Change the parameters directly in the `train.py` as needed.

```
python train.py
```

## One-step training to get OSOT-Flow

Run the following command. Change the parameters directly in the `train_one-step.py` as needed.

```
python train_onestep.py
```

## One-step generation

Run the following command. Change the parameters directly in the `sample_onestep.py` as needed.

```
python sample_onestep.py
```

