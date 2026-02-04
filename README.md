# AXIS
Automated crystal identification system developed at EMBL Grenoble Marquez group by Aurélien Personnaz.

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.11.03.685844-b31b1b.svg)](https://doi.org/10.1101/2025.11.03.685844)

![AXIS Overview](images/overview.png)

## Presentation
AXIS proposes a simple procedure to train a crystal identification system on any Crystallography infrastructure images. The system provided will return the presence probability of any crystalline material (large crystals, needles, micro-crystals, etc.) in an given drop micrograph. It was fully integrated into [EMBL Grenoble Crystallographic Image Management System (CRIMS)](https://www.embl.org/services-facilities/grenoble/high-throughput-crystallisation/) (for an academic license please contact [HTX@embl.fr](mailto:HTX@embl.fr)), but can be easily replicated in another infrastructure with the following steps.

You can find more details about the system and the use of UV light images in the [paper](https://doi.org/10.1101/2025.11.03.685844).

## Installation

Clone the repository, then create a python virtual enviroment and install the dependencies using the command below

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sign up to the HuggingFace repository and obtain a *write* token.
The models are pulled and stored in the [HuggingFace repository](https://huggingface.co/) automatically.

## Inference
Use the python script below to use the inference models described here. Provide a visible or UV light image or both from the same experiment. You will need to pass as parameters the visible and UV light models you want to use. The script will output the probability of the presence of crystals.

The HuggingFace token must be passed with the environment variable HUGGINGFACE_TOKEN
```
export HUGGINGFACE_TOKEN=*your token*
```
Here is an example with a pair of HTX Grenoble images and the AXIS-v3 models for both types of images:
```
python AXIS_inference.py \
--vis_model_checkpoint Marquez-Group-EMBL/AXIS-CRIMS_v3-vis --vis_image_path  ./images/Crystals_Vis.jpg \
--uv_model_checkpoint Marquez-Group-EMBL/AXIS-CRIMS_v3-uv --uv_image_path ./images/Crystals_UV.jpg
```
Outputs:
```
Visible light crystal probability: 100.00%
UV light crystal probability: 99.98%
Argmax crystal probability: 100.00%
```

## Training
To improve perfomance you can fine-tune the models described here by indlcuing data from your facility. This process can be carried out iteratively. For each iteration you will need: 
- A training dataset  labelled as “crystal” and “other”. As an example, EMBL grenoble HTX team initial dataset can be downloaded [there](https://doi.org/10.5281/zenodo.17279591).
- A test dataset with the same labels to evaluate performance As an example, EMBL grenoble HTX team test dataset can be downloaded [there](https://doi.org/10.5281/zenodo.17279081)


We higly recomend logging your training project with wandb (See wandb [user manual](https://docs.wandb.ai/get-started)). You will need to login first:

`wandb login`

Then the wandb entity and project name can be passed by setting the environment variables WANDB_ENTITY and WANDB_PROJECT
```
export WANDB_ENTITY=*your entity*
export WANDB_PROJECT=*your project*
```

In order to use the models described here and to store the new models trained, the HuggingFace token must be passed with the environment variable HUGGINGFACE_TOKEN
```
export HUGGINGFACE_TOKEN=*your token*
```

Then run a training with the training.py script. This scripts takes as parameters: 
- the image type (*vis* / *uv*) with  `--image_type`
- the number of epochs with `--n_epoch`
- the training and test datasets with  `--train_dataset` and `--test_dataset`
- the model checkpoint to train from with `--model_checkpoint`(defaults to *Marquez-Group-EMBL/AXIS-foundation*) 
- the batch size with `--batch_size` (defaults to 10)

Example:
```
python AXIS_training.py --image_type vis --n_epoch 2 --train_dataset ~/data/CRIMS-v1 --test_dataset ~/data/CRIMS-test
```

The new model will be stored in your huggingface repository, with a name starting by **AXIS-** with the training parameter values. To use the new model use instructions. Multiple training iterations can be done to improve the system efficiency.
