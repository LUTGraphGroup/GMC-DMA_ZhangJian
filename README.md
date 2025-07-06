# GNMBC
GNMBC: GNN-Mamba Co-contrastive Optimization for metabolite-disease association prediction

## 🏠 Overview
![image](flow_chart.jpg)


## 🛠️ Dependecies
- Python == 3.9
- pytorch == 1.12.1
- numpy == 1.22.4+mkl
- pandas == 1.4.4


## 🗓️ Dataset
```
disease-metabolite associations: association_matrix.csv and disease-metabolite.xlsx
Disease similarity network: diease_network_simi.csv
Metabolite similarity network: metabolite_ntework_simi.csv
```

## 🛠️ Model options
```
--epochs           int     Number of training epochs.                   Default is 200.
--GCNlayers        int     the number of layers in GCN.                 Default is 2.
--Mambalayers      int     the number of heads in multi-head attention. Default is 1.
--dropout          float   Dropout rate                                 Default is 0.
--lr               float   Learning rate                                Default is 0.005.
--wd               float   weight decay                                 Default is 5e-4.

```

## 🎯 How to run?
```
1. Run `similarity_compute.py` to obtain the similarity matrix.
2. Run `main.py` to obtain the experimental results.
```
