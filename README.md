# Knolling
## CV version

### Setup

```bash
gh repo clone H-Y-H-Y-H/Knolling
cd Knolling
conda create -n knolling python=3.9
conda activate knolling
pip install -r requirements.txt
```
### Intro

Here are the introduction for folders included in this project.

**Diffusion-Models-pytorch**

This is an easy-to-understand implementation of diffusion models from an external source.

**Diffusion_knolling_model**

My implementation of the diffusion model. Not finished yet.

**dataset**

Store the dataset

**ASSET**

Store 
1. models used in our previous work, such as Graspability Estimation Model, Transformer-based Knolling Model and Visual Perception Model.
2. Urdf related files for the simulation
3. URDF models of sundry goods for the data collection

**CV_knolling_model**

Main Part of the project, details seen in scripts of the folder.

**Running Sequence for frequently used processes**

1. Collecting Data for VAE model:

- run ```CV_knolling_model/collection.py``` with ```command``` equal to ```collection```, to generate all txt data of objects in the neat/random arrangement. 
you can run multiple instances of the program to speed up the data collection
- run ```merge_txt``` function in ```CV_knolling_model/preprocess.py``` to merge txt data of different instances into one
- run ```CV_knolling_model/collection.py``` with ```command``` equal to ```recover```, to convert txt data of the neat/random arrangement to image data of it.
you can run multiple instances of the program to speed up the data collection

Train the VAE model:

- run ```preprocess_img``` function in ```CV_knolling_model/preprocess.py``` to resize images to (128, 128), which is used in the VAE model
- run ```CV_knolling_model/Train&test/VAE_train.py```