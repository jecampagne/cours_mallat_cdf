En 2022, afin d'illustrer le cours de [Stéphane Mallat du Collège de France](https://www.di.ens.fr/~mallat/CoursCollege.html) j'ai mis sur pieds ce petit repository que vous pouvez cloner/forker et me faire des retours.

La plus part des nbs sont jouables sur Google Colab.

# Année 2022:
- `classif_simple_JAX.ipynb`: usage de classifiateurs simples pour classifier des objets fictifs avec 2 features. 
On optimise le classifiocateur via une descente de gradient stochastique (ici Adam) et l'on reprensente les separations entre classes.
- `MLE_Fisher_Info_1D.ipynb`: montre sur un exemple simple le fait que le MLE a bien une distribution normale dont la varaince est donnée par l'Information de Fisher.
- `fisher-mtx.ipynb`: calcul dans un cas simple multi-dimensionel de la matrice de Fisher. Puis en utilsant une librairie de génération de chaine de Markov, on compare les contours à n-sigmas des proba a posteriori jointes entre 2 parametres avec ceux obtenus en utilisant la matrice de Fisher.
- `Simple_huffman_code.ipynb`: une implemtation tres simple d'un code de Huffman quasi-optimal du point de vue de la borne de Shannon.
- `Allocation_de_bits.ipynb`: un exemple d'allocation de bits avec un algorithme glouton
- `image_compression.ipynb`: proposition de deux images classiques 512x512 de "lena" et "boat" pour effectuer des compressions de type JPEG et JPEG2000 à divers degré de compression (bit-per-pixel) et l'on mesure la qualité de restitution via le PSNR. 
```diff
- Pour le moment (fin mars 2022) le scaling PSNR(R) n'est pas celui attendu par la theorie pour R>1, ni en JPEG ni en JPEG2000. Nous n'avons pour le moment pas trouver la raison.
```
# Année 2023:
- `randomwalk.ipynb` : processus $X_{n+1} = \rho X_n + Z_{n+1}$ avec $Z_{n+1}$ une v.a $\{-1,+1\}$ (prob. 1/2) en 1D. Avec $\rho=1$ on obtient une marche aléatoire 
- `urne_Ehrenfest.ipynb` : illustration du moèle de gaz parfait contunu dans 2 boites séparées par une parois porreuse 

- `Jax` est une library d'auto-differentation et acceleration de code tres "nympy-like"
- `scikit-learn` est une library generaliste d'outils ML
- `torch`/`Pytorch`  est une library dediee aux reseaux de neurones mais ici on utilise uniquement quelques outils. 

Pour le nb `MLE_Fisher_Info_1D.ipynb` on utilise `jaxopt`  une librairie d'optimisation écrite en JAX. 

- `numpyro`est une librairy "Probabilistic programming with NumPy" via Jax.
C'est la nouvelle version de `Pyro` en Jax. 
- `arviz` et `corner` sont des librairies de presentation de resultats (ex. contour plots) de génération de chaine de Markov.

# Docs des packages
- JAX: https://jax.readthedocs.io
- PyTorch: https://pytorch.org/docs/stable/index.html
- scikit-learn = https://scikit-learn.org/stable/index.html
- matplotlib : https://matplotlib.org/stable/index.html
- numpy : https://numpy.org/doc/stable/reference/index.html
- Numpyro : https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro
- arviz : https://arviz-devs.github.io/arviz/index.html
