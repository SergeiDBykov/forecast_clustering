# Cosmological forecast for  angular clustering of AGN and Clusters in eROSITA All-Sky X-ray survey


This are  scripts and notebooks which are used to calculate the  results for the cosmological forecast of **eROSITA All-sky survey** in  X-ray. 

The  analysis results I used in my [paper]().

------

## The structure of the code is as follows:

- `./scripts`  
  the main scripts for the analysis are placed here. It containt a few files to manage the forecast.
    1. `./utils.py`  is used to set up pathes and contains some utility/plotting functions
    2. `./luminosity_funtions.py` contains functions and classes for X-ray  luminosity function (XLF) calulations. It includes XLF for AGN and clusters (the halo model for the latter), calculators of redshift/luminosity  distributions, linear bias factors, and other useful functions. 
    3. `./forecast.py` contains classes and functions for the cosmologial calculations, including angular power spectrum and Fisher Matrices.
    4. `./tests/` if a folder in which some tests for the code are stored, mainly for comparison with cosmological packages like CAMB, CCL, jax_cosmo.
    5. `./k_correction/`  contains a notebook for calulatin K-correction for the cluster spectra and tabulating it in a table `k_correction.npz`.
    


- `./results` is a folder in which the results are stored, in a form of text/`.npz` files and plots.  



- `./notebooks`  
  the main notebooks for the analysis are stored here. It containt a few folder and separate notebooke to make the forecast. The content of the folder is as follows (roughly in the order of  the paper sections):
    1. `./luminosity_functions/`  contains three notebooks for the luminosity functions, one for AGN, one for clusters, and one for the plotting both distributions in one figure (in section 3 of the paper). Those notebooks demonstrate the use of the classes and functions in `./luminosity_functions.py`, including `dn/dz`, `logN-logS` calculation for AGN and clusters. The instructions are found in the notebooks.
    2. `effective_volume.ipynb` calculates the effective volume of the survey for AGN and Cluster tracers (setion 2 of the paper). It uses  results from luminosity function calculators. 
    3. `./Cell_plot.ipynb` plots the angular power spectrum for AGN and clusters and its derivative for a representative redshift bins (section 4 of the paper. 
