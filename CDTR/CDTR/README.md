# Causally Debiased Time-aware Recommendation
A PyTorch implementation of Causally Debiased Time-aware Recommendation

[project page](https://ifqre.github.io/IFQRE/)



# Illustration

An illustration of our recommendation paradigm CDTR, where the users can explicitly indicate their disclosing willingness, and the model needs to trade-off the recommendation quality and user willingness. CDTR first finds the best strategy that balances the recommendation quality and user willingness. Then the user interactions are disclosed according to the optimal strategy. At last, the recommender model is trained based on the disclosed interactions.



# Usage

1.Install the required packages according to requirements.txt.

```bash
pip install -r requirements.txt
```


2.Prepare the datasets.

(1) Directly download the processed datasets used in this paper:

[ML-1M](https://drive.google.com/drive/folders/1arpebtUF5sg5T5sJY3S-XIjF_AdOW5u8?usp=share_link)
[Amazon](https://drive.google.com/drive/folders/1_jVor_2i32by-VSTtw49OFvw6T-n33e1?usp=share_link)
[Food](https://drive.google.com/drive/folders/1LFCMEb88E7wH6A_-ksygwyEqG_H4qX4q?usp=share_link)

(2) Use your own datasets:

Ensure that your data is organized according to the format: `user_id:token, item_id:token, timestamp:float,bin_id:float`.


3.Rename the dataset by `dataset-name.train/valid/test.inter`, and put it into the folder `./dataset/dataset-name/` (note: replace "dataset-name" with your own dataset name).


4.Run `main.py` to train and save model for propensity score, where the training parameters can be indicated through the config file.

For example:

```bash
python main.py --model=TimeProp --dataset=ML_1M_S1 --config_files=time_getp_ML_1M_S1.yaml 
python main.py --model=ItemProp --dataset=ML_1M_S1 --config_files=item_getp_ML_1M_S1.yaml 
```
5.Run `main.py` to train the debiased recommender model, where the training parameters can be indicated through the config file.

For example:

```bash
python main.py --model=TimeSVD --dataset=ML_1M_S1 --config_files=timesvd_ips_ML_1M_S1.yaml 
```

The parameter tuning ranges in our paper are as follows:

| Parameter                | Range                   |
| ------------------------ | ----------------------- |
| learning rate            | [0.0001,0.001,0.01,0.1] |
| imputation learning rate | [0.0001,0.001,0.01,0.1] |
| UM learning rate | [0.0001,0.001,0.01,0.1] |
| embedding size           | [32,64,128]             |
| $\Gamma_v$               | [1.1,1.3,1.5,1.8,2]             |
| $\Gamma_t$               | [1.1,1.3,1.5,1.8,2]             |
| batch size               | [1024]                  |
| training epochs          | [300]                   |

where `learning_rate` corresponds to the recommendation model, `imputation learning rate` represents the learning rate of the imputation model and `UM learning rate` is the learning rate of the model for updating the uncertanty sets.


# Detailed file structure

`CDTR`:  the main program that contains our model.

| catalog             | Description                                   |
| ------------------- | --------------------------------------------- |
| `CDTR.config`      | configed program                              |
| `CDTR.data`        | dataloading and handling program              |
| `CDTR.evaluator`   | evaluation program                            |
| `CDTR.model`       | our methods,sampler model and anchor model    |
| `CDTR.properties`  | predefined configuration of model and dataset |
| `CDTR.quick_start` | quick start program                           |
| `CDTR.sampler`     | data sampler                                  |
| `CDTR.trainer`     | training,validating and testing program       |
| `CDTR.utils`       | utils used to construct model                 |

| catalog   | Description         |
| --------- | ------------------- |
| `dataset` | predefined dataset  |
| `log`     | running information |
| `saved`   | saved model pth     |
| `asset`   | model asset         |
