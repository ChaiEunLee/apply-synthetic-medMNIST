# Applicability of synthetic data in medical field
>2023-2 데이터사이언스 특강 (전이학습 기반의 딥러닝)   
>[Sehyun Park](https://github.com/sehyunpark99), [Chaieun Lee](https://github.com/ChaiEunLee), Jungguk Kim, [Hyunbin Jin](https://github.com/hyunbinui)  (각자 깃허브 링크 달기!)

## Description   
This project is focused on **'Would synthetic data indeed beneficial in training medical AI model and improve the model's performance'**.    

AI model of medical field usually require a substantial amount of training data, which is always a challenge. Therefore, recently, there are many cases of creating synthetic data to increase training data. However, the effectiveness of this method is indeed questionable. This project checks its effectiveness of using the synthetic data by results of synthetic image quality and augmented classification. 

### Task
Image Classification of Pneumonia.   
> Models : 1) Resnet50 , 2) [DeiT-tiny](https://huggingface.co/facebook/deit-tiny-patch16-224)   
Framework : 1) Base 2) Synthetic base 3) Synthetic augmentation 4) Fine tuning    
Pre-trained by: 1) Backbone, 2) ImageNet, 3) chestMNIST

### Dataset
* Train : [Synthetic Dataset](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1)
* Test : PneumoniaMNIST from [MedMNIST Official](https://github.com/MedMNIST/MedMNIST)   
* Pre-trained : ImageNet, chestMNIST from [MedMNIST Official](https://github.com/MedMNIST/MedMNIST)   

### Process and Stacks
* GPU: 1 A6000
* cpus-per-task=4

## Result
### 1. ResNet50  
> * Pre-trained : 1) [ImageNet](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 2) [Backbone](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py), 3) [chest MNIST](https://zenodo.org/records/7782114) from [official medMNIST](https://github.com/MedMNIST/experiments)
> * Framework
>   * ```Base``` : Train and test with real data.   
>   * ```Synthetic base``` : Train in synthetic data and test with real data.   
>   * ```Synthetic augmentation``` :  Trained from scratch on varying splits of real and synthetic training data.
>   * ```Synthetic fine-tuning``` : Trained of real data and fine-tuning with synthetic data. 
> * [Hyperparameters]()   
> ```BATCH_SIZE = 128```
> ```NUM_EPOCHS = 50```
> ```lr = 0.0001```

Pre-trained | Frame work | Ratio | Test Accuracy 
---- | ---- | ---- | ---
Backbone | Base | &nbsp; |  0.1
&nbsp; | Synthetic base | &nbsp; |  0.1
&nbsp; | Synthetic augmentation | 100% | 0.1
&nbsp; | &nbsp; | 200%  | 0.1
&nbsp; | &nbsp; | 300%  | 0.1
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.1
ImageNet | Base | &nbsp; |  0.1
&nbsp; | Synthetic base | &nbsp; |  0.1
&nbsp; | Synthetic augmentation | 100% | 0.1
&nbsp; | &nbsp; | 200%  | 0.1
&nbsp; | &nbsp; | 300%  | 0.1
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.1
ChestMNIST | Base | &nbsp; |  0.1
&nbsp; | Synthetic base | &nbsp; |  0.1
&nbsp; | Synthetic augmentation | 100% | 0.1
&nbsp; | &nbsp; | 200%  | 0.1
&nbsp; | &nbsp; | 300%  | 0.1
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.1

>   * ```Domain Adaptation``` : [Fixbi](https://github.com/NaJaeMin92/FixBi)

Pre-trained | Frame work | Test Accuracy 
---- | ---- | ---
Backbone | DA |  0.1
ImageNet | DA |  0.1
ChestMNIST | DA |  0.1

### 2. DeiT
> * Pre-trained : 1) [ImageNet](https://github.com/facebookresearch/deit/blob/main/models.py), 2) [Backbone](https://github.com/facebookresearch/deit/blob/main/models.py), 3) [chest MNIST]()
> * Framework
>   * ```Base``` : Train and test with real data.   
>   * ```Synthetic base``` : Train in synthetic data and test with real data.   
>   * ```Synthetic augmentation``` :  Trained from scratch on varying splits of real and synthetic training data.
>   * ```Synthetic fine-tuning``` : Trained of real data and fine-tuning with synthetic data. 
> * [Hyperparameters](https://www.nature.com/articles/s41597-022-01721-8)    
> ```BATCH_SIZE = 128```
> ```NUM_EPOCHS = 50```
> ```lr = 0.0005 * BATCH_SIZE/512```
> ```weight_decay = 0.05```
> ```warmup_steps = 5 ```
> * Time : 3:00:00 

Pre-trained | Frame work | Ratio | Test Accuracy 
---- | ---- | ---- | ---
Backbone | Base | &nbsp; |  0.83173
&nbsp; | Synthetic base | &nbsp; |  0.1
&nbsp; | Synthetic augmentation | 100% | 0.80609
&nbsp; | &nbsp; | 200%  | 0.83494
&nbsp; | &nbsp; | 300%  | 0.83173
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.55929
ImageNet | Base | &nbsp; |  0.86859
&nbsp; | Synthetic base | &nbsp; |  
&nbsp; | Synthetic augmentation | 100% | 0.86218
&nbsp; | &nbsp; | 200%  | 0.85256
&nbsp; | &nbsp; | 300%  | 0.84615
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.40705
ChestMNIST | Base | &nbsp; |  0.80929
&nbsp; | Synthetic base | &nbsp; | 
&nbsp; | Synthetic augmentation | 100% | 0.84455
&nbsp; | &nbsp; | 200%  | 0.1
&nbsp; | &nbsp; | 300%  | 0.77564
&nbsp; | Synthetic Fine-tuning | &nbsp; | 0.41506

>   * ```Domain Adaptation``` : [CDTrans](https://github.com/CDTrans/CDTrans)

Pre-trained | Frame work | Test Accuracy   
---- | ---- | ---
Backbone | DA |  0.1
ImageNet | DA |  0.1
ChestMNIST | DA |  0.1


## Code Structure   
* ```requirement.yaml``` : To install dependencies
* **/synthetic** : train data 
* **/synthetic_shapred.zip** : train data.zip 
* **/Deit**
  * **/experiment** : Backbone
    * ```Deit_base.py``` : Base model with
    * ```Deit_pneu_synthetic.py``` :
    * ```Deit_pnue_synthetic_ratio.py``` :
    * ```Deit_synthetic_ft.py``` " 
  * **/expeirment_imagenet** : Imagenet pre-trained
    * ```Deit_base.py``` : Base model with
    * ```Deit_pneu_synthetic.py``` :
    * ```Deit_pnue_synthetic_ratio.py``` :
    * ```Deit_synthetic_ft.py``` " 
  * **/experiment_chest** : chest MNIST pre-trained
    * ```Deit_base.py``` : Base model with
    * ```Deit_pneu_synthetic.py``` :
    * ```Deit_pnue_synthetic_ratio.py``` :
    * ```Deit_synthetic_ft.py``` : 
* **/resnet**
```
FILL HERE
```
  
* **/pretrain_chest**
  * **/ckpt** : checkpoints for time limit
  * ```Deit_chest_pretrain.py``` : training chest MNIST to DeiT
  * ```Deit_chest_pretrained.pth``` : Result of pre-trained model of chestMNIST

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
**1) Resnet50**
**2) DeiT**
  - Change current directory before running Deit.
  ```
  cd Deit
  ```
  - Select {file_name} written above and run.
    - Backbone
    ```
    python ${HOME}/Deit/experiment/{file_name}.py
    ```
    - ImageNet pre-trained
    ```
    python ${HOME}/Deit/experiment_imagenet/{file_name}.py
    ```
    - chestMNIST pre-trained
    ```
    python ${HOME}/Deit/experiment_chest/{file_name}.py
    ```
**3) Make chest MNIST pre-trained model**
  ```
  python ${HOME}/pretrain_chest/Deit_chest_pretrain.py
  ```

# Reference
[MedMNIST Experiment Official](https://github.com/MedMNIST/experiments), [deit-tiny-pathch16-224 hugging face official](https://huggingface.co/facebook/deit-tiny-patch16-224)    
[Synthetic Dataset](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1), [MedMNIST Official](https://github.com/MedMNIST/MedMNIST)

## Language
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
</p>

