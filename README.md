ML_Graphs_Project
==============================

Environment Setup
------------
First create a virtual environment by
```
conda create -n <env_name> python=3.10
conda activate <env_name>
```
Then install the requirements and the repository setup by.
```
pip install -r requirements.txt
```

Additionally, the torch related packages can be installed using
```
pip install --no-cache-dir torch
pip install git+https://github.com/rusty1s/pytorch_sparse.git
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install git+https://github.com/rusty1s/pytorch_cluster.git
pip --no-cache-dir install torch-geometric
```

pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html

pip install --no-cache-dir install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip --no-cache-dir install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
#pip --no-cache-dir install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117

Running Experiments
------------
Experiments are executed using the `src/experiments/run_experiments.py` file.
Example 1:
```
python3 src/experiments/run_experiments.py --config-name='base.yaml' model=DownStream
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
