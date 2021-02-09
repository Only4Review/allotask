# How to distribute data across tasks for meta-learning?
This anonymous repository contains the code for reproducing the experimental results in the ICML 2021 submission 4168 "How to distribute data across tasks for meta-learning?".
## Environment
* We provide the *requirements.txt* for users to set up the working environment.
* We also provide a **DOCKERFILE** for users to create a docker environment.
## Prepare datasets for few-shot image classification
We do few-shot image classification experiments on the commonly used datasets **Cifar-FS** and **mini-ImageNet** [Download](https://github.com/bertinetto/r2d2).
## How to run the experiments
* To run Sinusoid regression experiments:
`python ./experiment/SinusoidExperiment.py`
* To run Cifar experiments: 
`python ./experiment/CifarExperiment.py`
* To run mini-ImageNet experiments:
`python ./experiment/ImagenetExperiment.py`
* For *infinite* budget, one needs to specify `no_tasks=-1` in the script.
## How to visualize the results
* We provide visualization scripts used to generate figures in the submission:
  * `python ./utils/visual_inspection.py`
  * `python ./utils/CifarVisualisation.py`
  
