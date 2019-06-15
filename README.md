# the_cocktail_party_nightmare
Code associated with the manuscript 'Quantifying the nightmare in the sonar cocktail party' - Thejasvi Beleyur, Holger R Goerlitz

[![DOI](https://zenodo.org/badge/114679151.svg)](https://zenodo.org/badge/latestdoi/114679151)

### What is the repository about?
Echolocating bats fly under acoustically challenging conditions when they fly in groups - in what has been called the ['cocktail party nightmare'](https://www.pnas.org/content/105/25/8491), as opposed to the [cocktail party problem/effect](https://en.wikipedia.org/wiki/Cocktail_party_effect). E

### Requirements:
The code runs on a range of Python 2.7 versions, and with the Numpy, Scipy, Pandas libraries. It also requires the [bridson](https://pypi.org/project/bridson/) package to simulate the poisson disc arrangement of bats in the cocktail party nightmare. The code runs on Windows and Linux. 
To ensure out-of-the-box running of the code - install the following package versions in your conda/virtual environment:
Numpy 1.16.3
Scipy 1.2.1
Pandas 0.24.2

### Running simulations described in the paper: the simulations folder
All simulation code is in the 'simulations' folder within. Each of the variable manipulations are under separate folders. The easiest way to run any one of the simulations is to move to the simulation directory of interest :

```
cd effect_of_group_size
python group_size_simulations
```

### Simulation output:
By default a batch of 100 simulations are output as .pkl files. Each .pkl file has the following pattern:
```simulationtype_CPN_<uuidcode>_numpyseed_<seednumber>.pkl . ```
 Each pickle file has 100 simulations by default. The seed number in each pickle 

A common set of simulation parameters is used for most simulations - and only the variable of interest is altered in some cases. The common simulation parameters is a Pickle file called 'commonsim_params.pkl' - and this can be changed using the 'create_and_svae_common_simparameters.py'. The wrapper code to change a single variable, run and save multiple simulations is called 'simulate_effect_of.py'. Within each simulation scenario this wrapper is called to alter only the variable of interest. 

**
