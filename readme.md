# Computer Vision - Group 18

## Folder Structure
- data: contain the code for data processing
- model: contain the code for models 
- utils: contain the code for utilities
- base_model.py, kg_assisted_model.py: the wrapper classes for the baseline model and proposed model
- main.py: the code for main flow to boost up training, evaluation

## Code Execution

Conda environment:
```
conda create -n <name> python=3.8
conda activate <name>
conda install --file requirements.txt
```

Ececute the code (with optional flags):
```
python main.py
```