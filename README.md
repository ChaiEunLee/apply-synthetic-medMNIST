# Applicability of synthetic data in medical field
>2023-2 데이터사이언스 특강 (전이학습 기반의 딥러닝)   
>[Sehyun Park](https://github.com/sehyunpark99), [Chaieun Lee](https://github.com/ChaiEunLee), Jungguk Kim, [Hyunbin Jin](https://github.com/hyunbinui)

## Description   
This project is focused on **'Would using a synthetic data feasible to be used in the medical field?'**.    

> AI model of medical field usually require a substantial amount of training data, which is always a challenge. Therefore, recently, there are many cases of creating synthetic data to increase training data. However, the effectiveness of this method is indeed questionable. This project checks its effectiveness of using the synthetic data by results of synthetic image quality and synthetic augmented classification. 

### - Task
Image Classification of Pneumonia.   
> Models : 1) Resnet50 , 2) [DeiT-tiny](https://huggingface.co/facebook/deit-tiny-patch16-224)   
Framework : 1) Base 2) Synthetic base 3) Synthetic augmentation 4) Synthetic fine-tuning    
Pre-trained by: 1) Backbone, 2) ImageNet, 3) chestMNIST

### - Dataset
* Real World Dataset : PneumoniaMNIST([MedMNIST Official](https://github.com/MedMNIST/MedMNIST))     
* Synthetic Dataset : [synthetic-covid-cxr-dataset](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1)
* Pre-trained Dataset : ImageNet / ChestMNIST([MedMNIST Official](https://github.com/MedMNIST/MedMNIST))     

### - Process and Stacks
* GPU: 1 A6000
* cpus-per-task=4

## Result
### 1. ResNet50  
> * Pre-trained : 1) [Backbone](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 2)[ImageNet](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 3) [ChestMNIST](https://zenodo.org/records/7782114) 
> * Framework
>   * ```Base``` : Train and test with real data.   
>   * ```Synthetic base``` : Train in synthetic data and test with real data.   
>   * ```Synthetic augmentation``` :  Trained from scratch on varying splits of real and synthetic training data.
>   * ```Synthetic fine-tuning``` : Trained of real data and fine-tuning with synthetic data. 
> * [Hyperparameters](https://arxiv.org/abs/2012.12877)    
> ```BATCH_SIZE = 128```
> ```NUM_EPOCHS = 50```
> ```lr = 0.0005 * BATCH_SIZE/512```
> ```weight_decay = 0.05```
> ```warmup_steps = 5 ```

Pre-trained | Frame work | Ratio | Test Accuracy 
---- | ---- | ---- | ---
Backbone | Base | &nbsp; |  0.83654
&nbsp; | Synthetic base | &nbsp; |  0.66186
&nbsp; | Synthetic augmentation | 100% | 0.83013
&nbsp; | &nbsp; | 200%  | 0.84455  
&nbsp; | &nbsp; | 300%  | 0.85256  
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.66506  
ImageNet | Base | &nbsp; | 0.87500
&nbsp; | Synthetic base | &nbsp; | 0.63942 
&nbsp; | Synthetic augmentation | 100% | 0.87821
&nbsp; | &nbsp; | 200%  | 0.85577
&nbsp; | &nbsp; | 300%  | 0.84615
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.72115
ChestMNIST | Base | &nbsp; |  0.88462
&nbsp; | Synthetic base | &nbsp; |  0.71795
&nbsp; | Synthetic augmentation | 100% | 0.88462
&nbsp; | &nbsp; | 200%  | 0.88462
&nbsp; | &nbsp; | 300%  | 0.88462
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.65064

### 2. DeiT
> * Pre-trained : 1) [Backbone](https://github.com/facebookresearch/deit/blob/main/models.py), 2) [ImageNet](https://github.com/facebookresearch/deit/blob/main/models.py), 3) ChestMNIST
> * Framework
>   * ```Base``` : Train and test with real data.   
>   * ```Synthetic base``` : Train in synthetic data and test with real data.   
>   * ```Synthetic augmentation``` :  Trained from scratch on varying splits of real and synthetic training data.
>   * ```Synthetic fine-tuning``` : Trained of real data and fine-tuning with synthetic data. 
> * [Hyperparameters](https://arxiv.org/abs/2012.12877)    
> ```BATCH_SIZE = 128```
> ```NUM_EPOCHS = 50```
> ```lr = 0.0005 * BATCH_SIZE/512```
> ```weight_decay = 0.05```
> ```warmup_steps = 5 ```


Pre-trained | Frame work | Ratio | Test Accuracy 
---- | ---- | ---- | ---
Backbone | Base | &nbsp; |  0.83173
&nbsp; | Synthetic base | &nbsp; |  0.36699
&nbsp; | Synthetic augmentation | 100% | 0.80609
&nbsp; | &nbsp; | 200%  | 0.83494
&nbsp; | &nbsp; | 300%  | 0.83173
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.55929
ImageNet | Base | &nbsp; |  0.86859
&nbsp; | Synthetic base | &nbsp; | 0.41186  
&nbsp; | Synthetic augmentation | 100% | 0.86218
&nbsp; | &nbsp; | 200%  | 0.85256
&nbsp; | &nbsp; | 300%  | 0.84615
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.40705
ChestMNIST | Base | &nbsp; |  0.80929
&nbsp; | Synthetic base | &nbsp; | 0.39423   
&nbsp; | Synthetic augmentation | 100% | 0.84455
&nbsp; | &nbsp; | 200%  | 0.76923  
&nbsp; | &nbsp; | 300%  | 0.77564
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.41506

## Code Structure   
* ```requirement.yaml``` : To install dependencies
* **/deit**
  * **/experiment_backbone** : Backbone
    * ```Deit_base.py``` : Train and test with real data.   
    * ```Deit_pnue_synthetic_ratio.py``` : Trained from scratch on varying splits of real and synthetic training data.    
    * ```Deit_synthetic_ft.py``` : Trained of real data and fine-tuning with synthetic data.    
    * ```Deit_synthetic.py``` :  Train in synthetic data and test with real data.   
  * **/expeirment_imagenet** : Imagenet pre-trained
    * *Same with **./experiment_backbone***
  * **/experiment_chest** : ChestMNIST pre-trained
    * *Same with **./experiment_backbone***

  
* **/resnet**
```
FILL HERE
```
  
* **/pretrain_chest**
  * ```Deit_chest_pretrain.py``` : Make pre-train model with ChestMNIST to DeiT
  * ```Deit_chest_pretrained.pth``` : Result of pre-trained model of ChestMNIST

## Installation and Requirements
* Clone the repository: 
```
git clone https://github.com/ChaiEunLee/apply-synthetic-medMNIST.git
cd apply-synthetic-medMNIST
```
* Open yaml and modify the name and path   
  : Change the ```{YOUR_ENV_NAME}``` and ```{PREFIX_OF_YOUR_DIRECTORY}``` in yaml file.

* Install dependencies:
```
conda env create -f environment.yaml
conda activate {YOUR_ENV_NAME}
```

* Prepare datasets
   : Install [Synthetic dataset](https://github.com/hasibzunair/synthetic-covid-cxr-dataset) and save as **/synthetic_shapred.zip**
```
import zipfile
with zipfile.ZipFile(f"{PATH}/synthetic_shared.zip", 'r') as zip_ref:
    zip_ref.extractall(f"{PATH}")
```

## Run File
**1) Resnet50**     
```
FILL HERE
```

**2) DeiT**
  - Change current directory before running Deit.
  ```
  cd deit
  ```
  - Select ```{file_name}``` written above and run.
    - Backbone
    ```
    python ${PATH}/deit/experiment_backbone/{file_name}.py
    ```
    - ImageNet pre-trained
    ```
    python ${PATH}/deit/experiment_imagenet/{file_name}.py
    ```
    - chestMNIST pre-trained
    ```
    python ${PATH}/deit/experiment_chest/{file_name}.py
    ```
**3) Make chest MNIST pre-trained model**
  ```
  python ${PATH}/pretrain_chest/Deit_chest_pretrain.py
  ```

# Reference
* Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou. (2020). [Training data-efficient image transformers & distillation through attention. arXiv:2012.12877.](https://arxiv.org/abs/2012.12877)   
* MedMNIST, “GitHub - MedMNIST/experiments: Codebase for reproducible benchmarking experiments in MedMNIST v2,” GitHub. [https://github.com/MedMNIST/experiments](https://github.com/MedMNIST/experiments)   
* MedMNIST, “GitHub - MedMNIST/MedMNIST: MedMNIST Introduction in MedMNIST v2,” GitHub. [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST)    
* synthetic-covid-cxr-dataset, GitHub - hasibzunair/synthetic-covid-cxr-dataset : Synthetic COVID-19 Chest X-ray Dataset for Computer-Aided Diagnosis, GitHub. [https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1)   
* [deit-tiny-pathch16-224 hugging face official](https://huggingface.co/facebook/deit-tiny-patch16-224)
* ResNet - 1) [Backbone](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 2)[ImageNet](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 3) [ChestMNIST](https://zenodo.org/records/7782114) 
* DeiT -  1) [Backbone](https://github.com/facebookresearch/deit/blob/main/models.py), 2) [ImageNet](https://github.com/facebookresearch/deit/blob/main/models.py)   

## Language
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
</p>

