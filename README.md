# Le pourquoi
Il s'agit de(s) notebook(s) pour illustrer le cours de Stéphane Mallat du Collège de France

# Le comment
Typiquement, il vous faut un environement `anaconda` avec `Python 3.8` 
- installer anaconda
- ensuite mieux vaut se faire un environement
```python
conda create --name myenv python=3.8
conda activate myenv
```

Ensuite, pour une installation pour tourner sur un CPU
```python
pip install -U pip
pip install -U "jax[cpu]"
pip install -U scikit-learn
pip install -U matplotlib
pip install -U numpy
pip install -U torch torchvision
```
Pour tourner sur GPU cela dépend de la version de CUDA, voir https://github.com/google/jax#installation

# Docs des packages
- JAX: https://jax.readthedocs.io
- PyTorch: https://pytorch.org/docs/stable/index.html
- scikit-learn = https://scikit-learn.org/stable/index.html
- matplotlib : https://matplotlib.org/stable/index.html
- numpy : https://numpy.org/doc/stable/reference/index.html

# Autres
- Anaconda : https://docs.anaconda.com/anaconda/install/index.html
- environement anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
- PIP : https://docs.python.org/fr/3.8/installing/index.html#basic-usage
