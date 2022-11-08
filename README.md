# COMP-4222-Project


## Data:
* https://www.kaggle.com/datasets/justinas/startup-investments
* **Raw data CSVs**
    * [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cmslau_connect_ust_hk/EliRyyIOmHJPjvnQSazTQuYBftxnTFjv0UmRHQSNlLdxqw?e=pGdIgk)

    * Download all csv and put them under the folder *raw_data*

* Data then is formatted under the folder *data*

## Result
| Model   |      Train AUC |  Val AUC | Test AUC | Epochs |
|:----------|:-------------:|:------:|:------:|:------------------:|
| GraphSAGE |   0.94       |  0.84 | 0.85  |1500               |
| GCN |              0.80  |   0.76|   0.74| 42           | 
| GAT |   0.88    |             0.80 |   0.80  |      300        |
  


## Further Direction
1. Overfitting Problem
    * Overfitting occurs even a high dropout rate (0.8) is added in several models. In the future, more normalization techniques could be added.