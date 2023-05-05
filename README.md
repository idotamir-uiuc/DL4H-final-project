# Reproducibility Project Of Ensemble Paper

This repository is an attempt to reproduce the results of the paper Statistical supervised meta-ensemble algorithm for medical record linkage ([Vo et al., 2019](https://www.sciencedirect.com/science/article/pii/S1532046419301388)). The original paper has a [publicly available repository on github](https://github.com/ePBRN/Medical-Record-Linkage-Ensemble).  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Preprocess Data
To preprocess the data (this step is necessary before any other step) run:

```setup
python FEBRL_process_data.py
python ePBRN_process_data.py
```

## Hyperparameter Selection

To reproduce how hyperparameters were chosen for each data set and reproduce Scheme_A.png and Scheme_B.png, run:

```train
python FEBRL_hyperparameters.py
python ePBRN_hyperparameters.py
```

## Train And Evaluate Models

To evaluate train and evaluate the models, run:

```eval
python FEBRL.py
python ePBRN.py
```

## Results

The following are the results the algorithms achieve on FEBRL and ePBRN in our attempted reproduction of the paper:

| Dataset | Algorithm   | Precision | Recall | F-1 score |
|---------|-------------|-----------|--------|-----------|
| FEBRL   | SVM         | 0.961     | 0.997  | 0.978     |
| FEBRL   | SVM-bag     | 0.952     | 0.997  | 0.974     |
| FEBRL   | NN          | 0.932     | 0.996  | 0.963     | 
| FEBRL   | NN-bag      | 0.929     | 0.996  | 0.961     |
| FEBRL   | LR          | 0.922     | 0.998  | 0.958     |
| FEBRL   | LR-bag      | 0.929     | 0.996  | 0.961     |
| ePBRN   | SVM         | 0.322     | 0.986  | 0.485     |
| ePBRN   | SVM-bag     | 0.734     | 0.964  | 0.834     |
| ePBRN   | NN          | 0.693     | 0.965  | 0.807     |
| ePBRN   | NN-bag      | 0.713     | 0.964  | 0.820     |
| ePBRN   | LR          | 0.593     | 0.968  | 0.736     |
| ePBRN   | LR-bag      | 0.911     | 0.998  | 0.952     |
| FEBRL   | Stack & Bag | 0.975     | 0.995  | 0.985     |
| ePBRN   | Stack & Bag | 0.730     | 0.964  | 0.831     |

The results from the original paper are in table 6 where scheme A corresponds to FEBRL and scheme B corresponds to ePBRN. Comparing the two it is evident that the performance on ePBRN in our reproduction study was significantly worse although the performance on FEBRL was very close.



## Citations
1. Kha Vo, Jitendra Jonnagaddala, and Siaw-Teng Liaw. 2019. Statistical supervised meta-ensemble algorithm for medical record linkage. Journal of Biomedical Informatics, 95:103220.
