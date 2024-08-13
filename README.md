# The Demo of DAS-N2N Denoising Model #

The model is finetuned from [https://github.com/sachalapins/DAS-N2N](https://github.com/sachalapins/DAS-N2N) which is originally trained with **Rutford Ice Stream** in **Antarctica**. [paper link](https://arxiv.org/pdf/2304.08120)

Our data is recorded in Taichung Powe Plant and is used to finetune the DAS-N2N model for denoising.

<<<<<<< HEAD:README.md
## Environment
- platform: Win11 x64
- python: 3.9.19
- tensorflow: 2.15.0

To create environment using Anaconda: $ conda env create -f <environment.yml>

## Data
=======
## Data ##
>>>>>>> f41e178683d29af995d8cb46178fd820e3be8a23:README
- date: 2023.11.21
- shape: (30000,2132) along temporal and spatial axis. While DAS-N2N model accepts the input shape of (128,96), therefore, we slice ours into a block of the shape (2585, 128, 96).

## Finetune ##
Following the training process of the original paper, the only pre-processing step applied is the z-score normalization. 
We finetuned the whole DAS-N2N model with 31 records of our DAS data, iterating for 1 epoch. 

## Demo
- jupyter notebook: select the environment that just import and run the cells in `demo.ipynb`
- command line: simply run the `demo.py`: $ python demo.py and it should do the work. (Figures are saved in `./fig/`)