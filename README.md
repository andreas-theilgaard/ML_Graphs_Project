ML Graphs Project
==============================

About
------------
...

Environment Setup
------------
First create a virtual environment by
```
conda create -n <env_name> python=3.10
conda activate <env_name>
```
Then install the torch related packages using the following
```
pip3 install --no-cache-dir torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip3 install --no-cache-dir torch-geometric
pip3 install -—no-cache-dir torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip3 install -—no-cache-dir torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip3 install -—no-cache-dir torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```
Finally, setup the environment and install any auxiliary dependencies using
```pip3 install -r requirements.txt
```

Running Experiments
------------
Experiments are executed using the `src/experiments/run_exps.py` file.
### Example 1:
This example shows how a simple MLP baseline can be trained for node classification on the ogbn-arxiv dataset.
```
python3 src/experiments/run_experiments.py --config-name='base.yaml' dataset='ogbn-arxiv' model_type=DownStream task='NodeClassification'
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── data
    │   ├── ogbl-collab
    │   ├── ogbn-arxiv
    │   ├── ....
    │   └── ....
    │
    ├── playground
    ├── requirements.txt
    ├── results
    │   └── get_results.py
    ├── setup.py
    ├── src
    │   ├──init__.py
    │   ├── config
    │   │   ├── base.yaml
    │   │   └── dataset
    │   │       ├── ogbl-collab.yaml
    │   │       └── ogbn-arxiv.yaml
    |   |       ....
    |   |       ....
    │   ├── data
    │   │   ├── __init__.py
    │   │   └── get_data.py
    │   ├── experiments
    │   │   ├── __init__.py
    │   │   └── run_exps.py
    │   ├── models
    │   │   ├── LinkPrediction
    │   │   │   ├── GNN_link.py
    │   │   │   └── mlp_linkpredict.py
    │   │   ├── Node2Vec.py
    │   │   ├── NodeClassification
    │   │   │   ├── GCN.py
    │   │   │   ├── GNN.py
    │   │   │   ├── GraphSage.py
    │   │   │   ├── mlp.py
    │   │   │   └── mlp_nodeclass.py
    │   │   ├── __init__.py
    │   │   ├── logger.py
    │   │   ├── shallow.py
    │   │   └── utils.py
    │   └── visualizations
    │       ├── __init__.py
    │       ├── embed_viz.py
    │       └── graph_viz.py
    ├── tests
    │   ├── __init__.py
    │   └── test_endpoints.py
    └── tox.ini
--------
