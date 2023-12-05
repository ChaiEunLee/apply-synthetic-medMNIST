# Applicability of synthetic data in medical field
>2023-2 데이터사이언스 특강 (전이학습 기반의 딥러닝)   
>Sehyun Park, [Chaieun Lee](https://github.com/ChaiEunLee), Jungguk Kim, Hyunbin Jin  (각자 깃허브 링크 달기!)

## Description
In medical field, lack of data is always a challenge. So, these days, there are many cases of creating synthetic data to increase training data.
This project is focused on **'whether using synthetic data actually improves the model's performance'**.    

> Models : 1) Resnet50 , 2) [Deit,tiny](https://huggingface.co/facebook/deit-tiny-patch16-224)   
Fine tuning : 1) Change head, 2) Complex layers, 3) Domain Adaptation    
Pre-trained : 1) ImageNet, 2) Backbone, 3) chest MNIST

### Dataset
* Train : [Synthetic Dataset](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1)
* Test : PneumoniaMNIST from [MedMNIST Official](https://github.com/MedMNIST/MedMNIST)

### Task
Image Classification of pneumonia.   
Train synthetic data and test in real world data (pneumoniaMNIST)

### Process and Stacks
* GPU :
* cpus-per-task=4


## Result
### ResNet50  

> * Fine tuning
>   * ```Simple``` : Change head to class numbers   
>   * ```Complex``` : Stack more layers   
>   * ```Domain Adaptation``` : Fixibi
> * Pre-trained : 1) [ImageNet](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 2) [Backbone](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 3) [chest MNSIT](https://zenodo.org/records/7782114) from [official medMNIST](https://github.com/MedMNIST/experiments)
> * [Hyperparameters]()   
> ```BATCH_SIZE = 128```
> ```NUM_EPOCHS = 50```
> ```lr = 0.0001```

Fine tuning | Pre-trained | Test Accuracy 
---- | ---- | ----
Simple | ImageNet | 0.1
Simple | Backbone | 0.1
Simple | chest MNIST | 0.1
Complex | ImageNet | 0.1
Complex | Backbone | 0.1
Complex | chest MNIST | 0.1
DA | ImageNet | 0.1
DA | Backbone | 0.1
DA | chest MNIST | 0.1

### Deit 
> * Fine tuning
>   * ```Simple``` : Change head to class numbers   
>   * ```Complex``` : Stack more layers   
>   * ```Domain Adaptation``` : CDTrans
> * Pre-trained : 1) [ImageNet](https://github.com/facebookresearch/deit/blob/main/models.py), 2) [Backbone](https://github.com/facebookresearch/deit/blob/main/models.py), 3) [chest MNIST]()
> * [Hyperparameters](https://www.nature.com/articles/s41597-022-01721-8)    
> ```BATCH_SIZE = 128```
> ```NUM_EPOCHS = 50```
> ```lr = 0.0005 * BATCH_SIZE/512```
> ```weight_decay = 0.05```
> ```warmup_steps = 5 ```
> * Time : 3:00:00 


Fine tuning | Pre-trained | Test Accuracy 
---- | ---- | ----
Simple | ImageNet | 0.1
Simple | Backbone | 0.1
Simple | chest MNIST | 0.1
Complex | ImageNet | 0.1
Complex | Backbone | 0.1
Complex | chest MNIST | 0.1
DA | ImageNet | 0.1
DA | Backbone | 0.1
DA | chest MNIST | 0.1


## Code Structure   
* ```requirement.yaml``` : To install dependencies
* /Deit
  * /synthetic : train data 
  * /synthetic_shapred.zip : train data.zip 
  * ```Deit.py``` : train and evaluation
  * ```slurm-244193.out```, ```slurm-244248.out``` : output of the training
  * ```DeiT_chest_pretrained.pth``` : : Pre-trained model of chestMNIST
  * ```DeiT_imagenet.pth``` : After training imagenet pre-trained model with synthetic dataset.
  * ```DeiT_naive.pth``` : After training backbone model with synthetic dataset.
  * ```DeiT_chest.pth``` : After training chestMNIST pre-trained model with synthetic dataset.
* /pretrain_chest
  * /ckpt : checkpoints for time limit
  * ```Deit_chest_pretrain.py``` : training chest MNIST to DeiT
  * ```Deit_chest_pretrained.pth``` : Result of pre-trained model of chestMNIST
  * ```slurm-234211.out```, ```slurm-234354.out``` : output of the training

## Installation and Requirements
* clone the repository: 
```
git clone https://github.com/ChaiEunLee/apply-synthetic-medMNIST.git
cd apply-synthetic-medMNIST
```
* open yaml and modify the name and path     
-> change the 'name' and 'prefix' in yaml file.

* Install dependencies:
```
conda env create -f environment.yaml
conda activate {YOUR_ENV_NAME}
```
## Run File
* Resnet
  - Simple Fine-tuning
  ```
  cd resnet
  python ${HOME}/Resnet.py
  ```
  - Complex Fine-tuning
  ```
  cd resnet
  python ${HOME}/Resnet.py
  ```
  - Domain Adaptation
  ```
  cd resnet
  python ${HOME}/Resnet.py
  ```
* Deit
  - Simple Fine-tuning
  ```
  cd Deit
  python ${HOME}/Deit.py
  ```
  - Complex Fine-tuning
  ```
  cd Deit
  python ${HOME}/Deit_finetune.py
  ```
  - Domain Adaptation
  ```
  python ${HOME}/Deit_da.py
  ```
## Make chest MNIST pre-trained model
  ```
  python ${HOME}/pretrain/Deit_chest_pretrain.py
  ```
# Reference
[MedMNIST Experiment Official](https://github.com/MedMNIST/experiments), [deit-tiny-pathch16-224 hugging face official](https://huggingface.co/facebook/deit-tiny-patch16-224)    
[Synthetic Dataset](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1), [MedMNIST Official](https://github.com/MedMNIST/MedMNIST)


## Language
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
</p>

