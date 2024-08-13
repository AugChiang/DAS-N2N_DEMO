# The Demo of DAS-N2N Denoising Model #

The model is finetuned from [https://github.com/sachalapins/DAS-N2N](https://github.com/sachalapins/DAS-N2N) which is originally trained with **Rutford Ice Stream** in **Antarctica**. [paper link](https://arxiv.org/pdf/2304.08120)

Our data is recorded in Taichung Powe Plant and is used to finetune the DAS-N2N model for denoising.

## Environment
- platform: Win11 x64
- python: 3.9.19
- tensorflow: 2.15.0

To create environment using Anaconda, type `$ conda env create -f environment.yml`. (default env name is "demo")

## Data
- date: 2023.11.21
- shape: (30000,2132) along temporal and spatial axis. While DAS-N2N model accepts the input shape of (128,96), therefore, we slice ours into a block of the shape (2585, 128, 96).

## Finetune
Following the training process of the original paper, the only pre-processing step applied is the z-score normalization. 
We finetuned the whole DAS-N2N model with 31 records of our DAS data, iterating for 1 epoch. 

## Content
Description of files or folders in this repo.

### fig
The folder that stores the prediction figures.

### test_data
Test data contains a piece of DAS data (`.npy` file) for demo.

### weights
The original DAS-N2N model (weights) is stored in `./weights/dasn2n_model` and is loaded for prediction.
While the `TunedModel.h5` in `./weights/` is the model (weights) after finetuning.

### demo.ipynb and demo.py
- jupyter notebook: select the environment that just import and run the cells in `demo.ipynb`
- command line: simply run the `demo.py` typing `$ python demo.py` and it should do the work. (Figures are saved in `./fig/`)