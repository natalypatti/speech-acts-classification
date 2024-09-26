# Speech Acts Classification


This repository provides an implementation of BERTimbau adapted with transductive learning and addition of morphosyntactic labels and context window. This adapted model was used for speech act classification in version 2 of the porttinari-base corpus annotated with speech acts. More details about the used dataset can be found in our paper: https://aclanthology.org/2024.propor-1.14/

## Code structure:

#### Data
- datasets used: src/data/data/
- main dataset (v2 of the published dataset): src/data/data/porttinari_base_labeled_consensus.csv

#### Results
- analysis of results: src/notebooks/
- model output: src/data/outputs/results/

### Scripts

- src:
  - main.py: main file used to run a single training of the proposed model
  - main_cross.py: used to run a single training of the proposed model with cross-validation
  - main_inference.py: used to generate predictions with the trained model
  - call_main.py: allows running the main.py file several times with different parameters
  - call_main_cross.py: allows running the main_cross.py file several times with different parameters
  - call_main_best.py: allows running the best trained models several times
  - speech-acts-classification/src/scripts/main_data.py: used to prepare the data for training
  - src/scripts: model implementation
 
### Setup:

- Creates a python virtual venv
  - conda create -n my-venv
- Activate the virtual env
  - conda activate my-venv
- install requirements
  - pip install -r requirements.txt

 
### How to run:

- first prepare the data (if you want to use your own database, change the path): python .\main_data.py

- training (you can change the training parameters directly in the script or during the call in the file): python src/main.py

- training more than one model: python src/call_main.py
