En 2022, afin d'illustrer le cours de [Stéphane Mallat du Collège de France](https://www.di.ens.fr/~mallat/CoursCollege.html) j'ai mis sur pieds ce petit repository que vous pouvez cloner/forker et me faire des retours.

Le directory "**Notes**" contient les PDFs des notes de cours depuis l'année 2018 incluse.
**Since 2023, this directory includes an english version of all the lectures since 2018.**

La plus part des nbs sont jouables sur Google Colab.

# Année 2025:
Un certain nombre de nb de 2024 sont dans le thème de cette année. Voici les nouveaux:

- 1 note sur les réseaux de J HopField et les Boltzmann machines de G. Hinton (Nobel de Physique 2024)
- 1 note sur les Normalizing Flows écrite (français) que j'ai écrit en 2022.

- `TensorFlow_bijector_1D_simple.ipynb`: illustration de Normalizing Flow (bijector) pour générer une distribution 1D (TensorFlow)
- `JAX_FLows_MAF_NVP_simple.ipynb`: (JAX) Normalizing Flow génération of 2D distributions. 
- `JAX_blob_GAN_vanilla.ipynb`: Mise en oeuvre d'un GAN simple et loss min-max (vanilla) pour générer une muli-gaussian 2D distribution. Selon le nombre de "blobs" le modele peut avoir des problèmes (mode collapses).
- `JAX_blob_GAN_Wasserstein_regul.ipynb`: (JAX) une implementation d'un GAN avec la fonction de coût de Wasserstein et une façon d'imposer le caractère Lipschitz du "critic" via une contrainte sur les gradients. 

# Année 2024:
Un certain nombre de nb de 2023 sont dans le thème de cette année comme `Monte_Carlo_Sampling.ipynb` et `Monte_Carlo_Sampling_2.ipynb` traitent
de l'échantillonnage, 
`gaussian_vs_turbulent_fow.ipynb` montre les carences d'un modèle gaussien tandis que `TextureSynthesis.ipynb` montre des qualités de générations
de champs non gaussiens par une modélisation micro-canonique.

- `ScoreDiffusionGene.ipynb`: illustration **Score Based Diffusion model**. Usage d'une Stochastic Differential Equation (inversible) (**Ornstein-Uhlenbeck proces**s)  en mode Forward & Backward pour générer de nouveaux échantillons d'une pdf 1D cible en partant d'une distribution Normale.
- `Ising2D_Metropolis.ipynb`: une petite implémentation de l'algorithme de Metropolis pour la génération de champ d'Ising 2D sur une grille $NxN$. On utilise des spins (0,1) codés sur des entiers non-signés de 32 bits pour générer 32 chaines de Markov en paralèlle. On obtient une courbe de la magnétisatiion moyenne en fonction de la température qui est assez proches de la théorie. On ne peut aller plus loin avec ce type d'algorithme.
- `Ising2D-Checkerboard-Flax.ipynb`: c'est une variation sur le thème de la génération de réseau d'Ising 2D classique, en utilisation l'algorithme de Checkerboard en association avec celui de **Metropolis**. On utilise alors la convolution par un kernel qui reflète les interactions entre plus proches voisins du réseau. Tourner sur GPU est à envisager.
- `jax_phi4_langevin.ipynb` : Génération de cartes 2D de champ pour un théorie  $\lambda \varphi^4$ en utilisant la résolution de l'**équation de Langevin** par une méthode itérative simple. Tourner sur CPU reste limité, GPU c'est mieux.
- `jax_phi4_HMCsimple.ipynb` : Génération de cartes 2D de champ pour un théorie  $\lambda \varphi^4$  en utilisant une implémentation simple d'un Hamiltonian Monte Carlo.
- `Ornstein_Uhlenbeck.ipynb`: exemple de processus 1D de Ornstein_Uhlenbeck $dX = \alpha\ (\mu - X)\ dt + \sigma\ dW$
- `Wavelet1DDenoising.ipynb`: exemple de débruitage d'un sigma 1D via "hard thresholding" sur les wavelet coefficients. Comparaison avec un débruitage par filtrage de Fourier 


# Année 2023:
Le nb de 2022 `fisher-mtx.ipynb` peut servir : il compare les contours de la posterior des paramatres issus de le matrice de  Fisher avec ceux obtenus en analysant la chaine de Markov d'un Monte Carlo Hamiltonien (cf. HMC voire NUTS aka No-U-Turns) 

- `randomwalk.ipynb` : processus $X_{n+1} = \rho X_n + Z_{n+1}$ avec $Z_{n+1}$ une v.a $\{-1,+1\}$ (prob. 1/2) en 1D. Avec $\rho=1$ on obtient une marche aléatoire 
- `urne_Ehrenfest.ipynb` : illustration du moèle de gaz parfait contunu dans  boite à 2 compartiments séparées par une paroi porreuse. Si $X_n$ représente le nombre de boules dans un compartiment, alors $X_{n+1} = X_n + Z_{n+1}$ avec $Z_{n+1}$ une v.a $\{-1,+1\}$ mais cette fois $P(Z_{n+1} = −1|X_n = x) = x/N$, donc ce n'est pas une marche aléatoire.
- `Monte_Carlo_Sampling.ipynb` : nb pedagogique sur la procédure d'échantillonnage (moyenne, Metropolis-Hasting MCMC, Hamiltionian MCMC) pour le calcul d'intégrale et l'inférence de paramètres.
- `Monte_Carlo_Sampling_2.ipynb` : dans la continuité du nb précédent, on utilise une librairie (Numpyro) pour effectuer de la production de MCMC efficace.
- `gibbs_FFT.ipynb` : Phénomène de Gibbs par seuillage dans l'espace de Fourier
- `morlet_wave_1D_2D.ipynb`: Représentation de l'ondelette de Morlet en 1D et 2D
- `wavelet1D.ipynb` : Décomposition en ondelettes de signaux 1D et représentation scalogramme (ie. temps-échelle).
- `wavelet2D_sparsity.ipynb` : Décomposition en ondelettes d'une image et calcul de la sparsité des coefficients d'ondelettes.
- `scattering1D.ipynb` : Example de Scattering transform sur un signal synthétisé 1D
- `scattering2D.ipynb` : Example de Scattering transform 2D 
- `gaussian_vs_turbulent_fow.ipynb` : A partir d'une image d'un flux turbulent, on construit un champ gaussien ayant le même spectre de puissance, et l'on compare visuellement les deux images: la géométrie des structures n'est pas préservée.
-  `TextureSynthesis.ipynb` : Example de synthèses de champs et textures à parir d'une modélisation micro-canonique utilisant à base de Transformation de Scattering et les correlation entre échelles.
-  `StatComponentSeparation.ipynb`: Example simple de séparation de composante (débruitage) utilsant une modélisation micro-canonique à base de Transformation de Scattering et les cross/auto-correlation entre échelles.


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


# Librairies
- `Jax` est une library d'auto-differentation et acceleration de code tres "nympy-like"
- `scikit-learn` est une library generaliste d'outils ML
- `torch`/`Pytorch`  est une library dediee aux reseaux de neurones mais ici on utilise uniquement quelques outils. 
- `numpyro`est une librairy "Probabilistic programming with NumPy" via Jax.
C'est la nouvelle version de `Pyro` en Jax. 
- `Kyamato` et `PyWavelets`, `ssqueezepy`: librairies pour traiter la transformée en Ondelettes
- `arviz` et `corner` sont des librairies de presentation de resultats (ex. contour plots) de génération de chaine de Markov.

# Docs des packages
- JAX: https://jax.readthedocs.io
- PyTorch: https://pytorch.org/docs/stable/index.html
- scikit-learn = https://scikit-learn.org/stable/index.html
- matplotlib : https://matplotlib.org/stable/index.html
- numpy : https://numpy.org/doc/stable/reference/index.html
- Numpyro : https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro
- arviz : https://arviz-devs.github.io/arviz/index.html
- Kyamato : https://www.kymat.io/
- PyWavelets : https://pywavelets.readthedocs.io
