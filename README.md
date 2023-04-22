>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

[//]: # (# Medical-Record-Linkage-Ensemble)

[//]: # ()
[//]: # (This repository contains the full code of the paper "Statistical supervised meta-ensemble algorithm for data linkage", published in Journal of Biomedical Informatics. )

[//]: # ()
[//]: # (Authors: )

[//]: # (Kha Vo <kha.vo@unsw.edu.au>,)

[//]: # (Jitendra Jonnagaddala <jitendra.jonnagaddala@unsw.edu.au>,)

[//]: # (Siaw-Teng Liaw <siaw@unsw.edu.au>.)

[//]: # ()
[//]: # (+ All of the code in this repository used Python 3.6 or higher with these prerequisite packages: `numpy`, `pandas`, `sklearn`, and `recordlinkage`. To install a missing package, use command `pip install package-name` in a terminal &#40;i.e., cmd in Windows, or Terminal in MacOS&#41;.)

[//]: # ()
[//]: # (1. Prepare the cleaned datasets for Scheme A, which are stored in two files `febrl3_UNSW.csv` and `febrl3_UNSW.csv`. To reproduce those two files, use Python Notebook &#40;i.e., Jupyter Notebook of Anaconda3 distribution&#41; to run `Rectify_Febrl_Datasets.ipynb`.)

[//]: # ()
[//]: # (2. Prepare the synthetic ePBRN-error-simulated datasets for Scheme B, which are stored in two files `ePBRN_D_dup.csv` and `ePBRN_F_dup.csv`. The original FEBRL dataset &#40;all original, no duplicate&#41; is contained in 2 files: `/ePBRN_Datasets/ePBRN_D_original.csv` and `/ePBRN_Datasets/ePBRN_F_original.csv`. To reproduce `ePBRN_D_dup.csv` and `ePBRN_F_dup.csv`, run `Error_Generator.ipynb`. In the first cell of the notebook, change variable `inputfile` to either `ePBRN_D_original` or `ePBRN_F_original`, which is respectively corresponding to variable `outputfile` of `ePBRN_D_dup` or `ePBRN_F_dup`. )

[//]: # ()
[//]: # (3. Reproduce results for Scheme A in the paper by running `FEBRL_UNSW_Linkage.ipynb`.)

[//]: # ()
[//]: # (4. Reproduce results for Scheme B in the paper by running `ePBRN_UNSW_Linkage.ipynb`.)

[//]: # ()
[//]: # (5. The plots in the paper can be reproduced by running `Plots.ipynb`.)
