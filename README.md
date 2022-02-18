En 2022, afin d'illustrer le cours de [Stéphane Mallat du Collège de France](https://www.di.ens.fr/~mallat/CoursCollege.html) j'ai mis sur pieds ce petit repository que vous pouvez cloner/forker et me faire des retours.

# Voici le contenu en 2022:
- `classif_simple_JAX.ipynb`: usage de classifiateurs simples pour classifier des objets fictifs avec 2 features. 
On optimise le classifiocateur via une descente de gradient stochastique (ici Adam) et l'on reprensente les separations entre classes.
- `MLE_Fisher_Info_1D.ipynb`: montre sur un exemple simple le fait que le MLE a bien une distribution normale dont la varaince est donnée par l'Information de Fisher.
- `fisher-mtx.ipynb`: calcul dans un cas simple multi-dimensionel de la matrice de Fisher. Puis en utilsant une librairie de génération de chaine de Markov, on compare les contours à n-sigmas des proba a posteriori jointes entre 2 parametres avec ceux obtenus en utilisant la matrice de Fisher.

# Le comment
Typiquement, il vous faut un environement `anaconda` avec `Python 3.8` 
- installer anaconda
- ensuite mieux vaut se faire un environement
```python
conda create --name myenv python=3.8
conda activate myenv
```

Ensuite, pour une installation en vu d'utiliser le nb `classif_simple_JAX.ipynb` (tourner sur un CPU) 
```python
pip install -U "jax[cpu]"
pip install -U scikit-learn
pip install -U matplotlib
pip install -U torch torchvision
```

- `Jax` est une library d'auto-differentation et acceleration de code tres "nympy-like"
- `scikit-learn` est une library generaliste d'outils ML
- `torch`/`Pytorch`  est une library dediee aux reseaux de neurones mais ici on utilise uniquement quelques outils. 

Pour le nb `MLE_Fisher_Info_1D.ipynb` il faudra installer
```python
pip insatll -U jaxopt
```
qui est une librairie d'optimisation sans ou avec contrainte. 

Pour le nb `fisher-mtx.ipynb` il faudra en plus installer
```python
pip install -U numpyro
pip install -U arviz
pip install -U corner
```
- `numpyro`est une librairy "Probabilistic programming with NumPy" via Jax.
C'est la nouvelle version de `Pyro` en Jax. 
- `arviz` et `corner` sont des librairies de presentation de resultats (ex. contour plots) de génération de chaine de Markov.



Pour tourner sur GPU cela dépend de la version de CUDA, voir https://github.com/google/jax#installation

# Docs des packages
- JAX: https://jax.readthedocs.io
- PyTorch: https://pytorch.org/docs/stable/index.html
- scikit-learn = https://scikit-learn.org/stable/index.html
- matplotlib : https://matplotlib.org/stable/index.html
- numpy : https://numpy.org/doc/stable/reference/index.html
- Numpyro : https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro
- arviz : https://arviz-devs.github.io/arviz/index.html

# Autres
- Anaconda : https://docs.anaconda.com/anaconda/install/index.html
- environement anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
- PIP : https://docs.python.org/fr/3.8/installing/index.html#basic-usage
