# COMP-4222-Project

## rmb to put data under data folder before submission

## Topic:
**Graph Neural Network in recommending venture capital investments for startups and investors**
### Background:
Investors and venture capitalists want to invest in startups through venture capital investments. However, due to limited resources and information, startups may not be able to find interested investors; while investors might not be aware that some startups are raising funds or selling their shares. Therefore, there is a need to match the sellers with possible buyers by recommending different startups for investors, and vice versa.

### Goal:
Build a recommendation system to match the startup companies and investors

### Methodology:
We structure this question into a bipartite graph 𝐺 = (𝑈, 𝑉, 𝐸). The set U refers to the investors / funds, while V refers to startups. Our goal is to predict whether deals (links) would occur between investors and startups, formulating a link prediction problem.
We would also include some features which affect the decision of investors or startups. For example, investors' background, as well as their previous invested industries would be treated as the node properties of U. The startup industries, deal size, deal round, etc. would be considered as the node properties of V. We would also
consider previous deal transactions, which act as the links for training data.

---

## Data:
* https://www.kaggle.com/datasets/justinas/startup-investments
* Raw data CSVs
    * [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cmslau_connect_ust_hk/EliRyyIOmHJPjvnQSazTQuYBftxnTFjv0UmRHQSNlLdxqw?e=pGdIgk)

    * Download all csv and put them under the folder *raw_data*

* Data then is formatted under the folder *data*

#### Stats
* Number of Nodes (25446)
    * Investors: xxxxx
    * Startups: xxxxx
* Number of Edges (45621)

---
## Models
We tested serveral models, such as
* Matrix Factorization
* GraphSAGE
* Graph Convolutional Network
* Graph Attention Network
* ...


---

## Result
| Model   |      Train AUC |  Val AUC | Test AUC | Epochs |
|:----------|:-------------:|:------:|:------:|:------------------:|
| Matrix Factorization (Baseline) |   0.91    |             / |   /  |      3000        |
| GraphSAGE |   0.94       |  0.84 | 0.85  |1500               |
| GCN |              0.80  |   0.76|   0.74| 42           | 
| GAT |   0.88    |             0.80 |   0.80  |      300        |
| VGAE |   0.72    |             0.75 |   0.74  |      511        |

---
## Further Direction
1. Overfitting Problem
    * Overfitting occurs even a high dropout rate (0.8) is added in several models. In the future, more normalization techniques could be added.
2. VGAE result? 