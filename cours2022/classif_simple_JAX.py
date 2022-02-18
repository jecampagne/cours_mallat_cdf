# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: mallat
#     language: python
#     name: mallat
# ---

# +
import time
import functools
# JAX library : une nouvelle venue dans le domaine 
#       l'auto-differentiation
#       la compilation Just-In-Time 
#       l'optilisation XLA du code compile pour le device CPU/GPU/TPU

import jax
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap, value_and_grad
from jax.example_libraries import optimizers

from jax.config import config
config.update("jax_enable_x64", True)


from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax)
print("DEBUG Jax is running on device:",jax.devices())

# Numpy 
import numpy as np

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.style.use('seaborn-white')
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 18

# scikit-learn : ici on utilise qq utilitaires pour générer des datas
#     mais on pourrait faire la classification
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Pytorch : ici on utilise un utilitaire pour faire des lots de donnees: training/test + batch
#   mais on pourrait aussi faire de la classification 
from torch.utils.data import Dataset,  DataLoader
# -

# # Simple classification en 2D

num_features = 2
num_classes  = 5
batch_size   = 5
# Generate key which is used to generate random numbers
key = random.PRNGKey(42)


# # Generations de données

def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


n_samples = 1500
np.random.seed(0)
X, y= datasets.make_blobs(n_samples=2*n_samples, 
                          centers=num_classes,
                         random_state=70)
# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.coolwarm)
plt.gca().set_aspect('equal')
plt.xlabel("x1")
plt.ylabel("x2");

# Obtenir des lots de training et de test
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)


class MyDataset(Dataset):
    """ Class to use the torch Dataloader mechanism """
    def __init__(self, X,y):
        self.x_data, self.y_data = X,y
    
    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]

    def __len__(self):
        return len(self.x_data)



dataset_train = MyDataset(X_train,y_train)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = MyDataset(X_test,y_test)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


def plot_performance(train_loss, train_acc, test_acc,
                           sup_title="Loss Curve"):
    """ Visualize the learning performance of a classifier """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(train_loss)
    axs[0].set_xlabel("# Batch Updates")
    axs[0].set_ylabel("Batch Loss")
    axs[0].set_title("Training Loss")

    axs[1].plot(train_acc, lw=3, label="Training")
    axs[1].plot(test_acc, lw=3, ls="--", label="Test")
    axs[1].set_xlabel("# Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Prediction Accuracy")
    axs[1].legend()

    # Give data more room to bloom!
    for i in range(2):
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    fig.suptitle(sup_title, fontsize=25)
    fig.tight_layout(rect=[0, 0.03, 1, 0.925])


# +
class Classif():
    """ Class to optimize a classifier """
    def __init__(self, clf, num_features, num_classes, batch_size):
        
        self.num_features = num_features
        self.num_classes  = num_classes
        self.batch_size   = batch_size

        #initialisation & feed-forward pass    
        self.init_nn, self.apply_nn = clf

        # Optimizer Adam for Gradient Descent
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)

    
    def loss(self, params, data, targets):
        """ loss - Sum_i y_i log(p_\theta(x_i)) """
        preds = self.apply_nn(params, data)
        return -jnp.sum(preds * targets)

    @functools.partial(jit, static_argnums=0)
    def update(self, params, x, y, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(self.loss)(params, x, y)
        opt_state = self.opt_update(0, grads, opt_state)
        return self.get_params(opt_state), opt_state, value
    
    def accuracy(self, params, data_loader):
        """ Compute the mean accuracy for a batch """
        acc_total = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = jnp.array(data)
            targets = one_hot(np.array(target), self.num_classes)

            target_class = jnp.argmax(targets, axis=1)
            predicted_class = jnp.argmax(self.apply_nn(params, data), axis=1)
            acc_total += jnp.sum(predicted_class == target_class)
        return acc_total/len(data_loader.dataset)


    def run_training_loop(self, num_epochs, train_loader, test_loader, seed=42):
        """ Implements a learning loop over epochs. """
        # Initialize placeholder for loggin
        log_acc_train, log_acc_test, train_loss = [], [], []
        
        key = jax.random.PRNGKey(seed)
        _,params_init = self.init_nn(key, (self.batch_size, self.num_features))
        opt_state = self.opt_init(params_init)

        # Get the initial set of parameters
        params = self.get_params(opt_state)
    
        # Get initial accuracy after random init
        train_acc = self.accuracy(params, train_loader)
        test_acc = self.accuracy(params, test_loader)
        log_acc_train.append(train_acc)
        log_acc_test.append(test_acc)
    
        # Loop over the training epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                x = jnp.array(data)
                y = one_hot(np.array(target), self.num_classes)
                params, opt_state, loss = self.update(params, x, y, opt_state)
                train_loss.append(loss)

            epoch_time = time.time() - start_time
            train_acc = self.accuracy(params, train_loader)
            test_acc = self.accuracy(params, test_loader)
            log_acc_train.append(train_acc)
            log_acc_test.append(test_acc)
            print(f"Epoch {epoch+1} | T: {epoch_time:0.2f} | Train Acc: {train_acc:0.4f} | Test Acc: {test_acc:0.4f}")

        return train_loss, log_acc_train, log_acc_test, params, params_init


# -

# ## Models:
# - Simple model linéaire 
# - Un Neural net à 1 couche cachée et un rectificateur comme non-linéairité

# +
model = stax.serial(Dense(num_classes),
                        LogSoftmax        # compute the log of the softmax
                      )
clf = Classif(model, num_features=num_features, num_classes=num_classes, batch_size=batch_size)

num_epochs=40
train_loss, log_acc_train, log_acc_test, params, params_init = \
        clf.run_training_loop(num_epochs, train_loader, test_loader)

init_nn, apply_nn = model
# -

plot_performance(train_loss, log_acc_train, log_acc_test)



# +
xx, yy = np.meshgrid(np.linspace(np.min(X[:,0])*1.1,np.max(X[:,0])*1.1,100), 
                     np.linspace(np.min(X[:,1])*1.1,np.max(X[:,1])*1.1,100))
positions = jnp.vstack([xx.ravel(), yy.ravel()]).T
predict_init = jnp.argmax(jnp.exp(apply_nn(params_init,positions)),axis=1)
f_init = np.reshape(predict_init,xx.shape)

predict = jnp.argmax(jnp.exp(apply_nn(params,positions)),axis=1)
f = np.reshape(predict,xx.shape)

fig,axs = plt.subplots(1,2,figsize=(16,8))
axs[0].contourf(xx,yy,f_init,cmap=plt.cm.coolwarm, alpha=0.3)
axs[0].scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.coolwarm)
axs[0].set_aspect('equal')
axs[0].set_title(f"Classif. of {num_classes} classes (random param)")

axs[1].contourf(xx,yy,f,cmap=plt.cm.coolwarm, alpha=0.3)
axs[1].scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.coolwarm)
axs[1].set_aspect('equal')
axs[1].set_title(f"Classif. of {num_classes} classes (optim. param)");
# -

n_hidden = 100
model= stax.serial(
        Dense(n_hidden),
        Relu,
        Dense(num_classes),
        LogSoftmax
)

# +
clf1 = Classif(model, num_features=num_features, num_classes=num_classes, batch_size=batch_size)

num_epochs=20
train_loss, log_acc_train, log_acc_test, params, params_init = \
        clf1.run_training_loop(num_epochs, train_loader, test_loader)

init_nn, apply_nn = model
# -

plot_performance(train_loss, log_acc_train, log_acc_test)

# +
xx, yy = np.meshgrid(np.linspace(np.min(X[:,0])*1.1,np.max(X[:,0])*1.1,100), 
                     np.linspace(np.min(X[:,1])*1.1,np.max(X[:,1])*1.1,100))
positions = jnp.vstack([xx.ravel(), yy.ravel()]).T
predict_init = jnp.argmax(jnp.exp(apply_nn(params_init,positions)),axis=1)
f_init = np.reshape(predict_init,xx.shape)

predict = jnp.argmax(jnp.exp(apply_nn(params,positions)),axis=1)
f = np.reshape(predict,xx.shape)

fig,axs = plt.subplots(1,2,figsize=(16,8))
axs[0].contourf(xx,yy,f_init,cmap=plt.cm.coolwarm, alpha=0.3)
axs[0].scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.coolwarm)
axs[0].set_aspect('equal')
axs[0].set_title(f"Classif. of {num_classes} classes (random param)")

axs[1].contourf(xx,yy,f,cmap=plt.cm.coolwarm, alpha=0.3)
axs[1].scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.coolwarm)
axs[1].set_aspect('equal')
axs[1].set_title(f"Classif. of {num_classes} classes (optim. param)");
