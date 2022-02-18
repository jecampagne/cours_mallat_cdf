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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
mpl.rcParams['font.size'] = 18
import numpy as np


def f(x,b):
    return x**4 + (1 + 2*b)*x**2


xvals = np.arange(-2,2,0.01)

fig = plt.figure(figsize=(8,8))
for b in [-1.,-1/3,-1/2,1.]:
    plt.plot(xvals,f(xvals,b),label=f"b={b:.2}")
plt.ylim([-0.5,0.8])
plt.xlim([-1.5,1.5])
plt.legend()
plt.grid()


