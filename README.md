# IT4868E-Movies-Recommendation-System
Capstone project

Apply LightGCN model and Heterogeneous model to experiment performance on Movie Lens dataset

## Usage
### Datasets
Download the [ml-latest-small](https://husteduvn-my.sharepoint.com/:u:/g/personal/minh_nqn242051m_sis_hust_edu_vn/ESe6FkhXx_FKjPr31zqpLQABc-BZwlRZIPZv8_6xqh-Abg?e=oIvQIi) or [ml-32m](https://husteduvn-my.sharepoint.com/:u:/g/personal/minh_nqn242051m_sis_hust_edu_vn/EfgKP3GmTJdDgXKUs5JgOf4B1kkDg94I8vaKdubilYtyqA?e=WcTLes), which include user, movie, genre, director, etc., into the `dataset/<name_data>`

### Config
Open `config.yaml`, set desired data paths to your csv files, sure it contains 4 data paths as it is in `config.yaml`.

If you want to use other information than `user` and `movie`, remove some properties in `.model.exclude_node`.

### Train
```
train.py [-h] [--config CONFIG] [--checkpoint CHECKPOINT] [--resume RESUME]
```
See in `train.py` arguments for more details.

### Evaluation
```
eval.py [-h] --checkpoint CHECKPOINT [--split {val,test}]
```
See in `eval.py` arguments for more details.

> **⚠️Note:** The dataset default is splited by 7:2:1 respecting to train:val:test when running `train.py` or `val.py`. Random seeds are set to 0 to guarantee the deterministic dataset between trainning and evaluation. 