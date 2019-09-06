# The Cocktail Party Nightmare
Code associated with the manuscript 'Quantifying the nightmare in the sonar cocktail party' - Thejasvi Beleyur, Holger R Goerlitz

[![DOI](https://zenodo.org/badge/114679151.svg)](https://zenodo.org/badge/latestdoi/114679151)

### What is the repository about?
Echolocating bats fly under acoustically challenging conditions when they fly in groups - in what has been called the ['cocktail party nightmare'](https://www.pnas.org/content/105/25/8491), as opposed to the [cocktail party problem/effect](https://en.wikipedia.org/wiki/Cocktail_party_effect). This repository contains the code which can be used to replicate the results from the published paper above on it. 

### Requirements:
The code runs on a range of Python 2.7 versions, and with the Numpy, Scipy, Pandas libraries. The code runs on Windows and Linux. 
To replicate the development environment exactly - install the following versions in your conda/virtual environment:
Python 2.7.12

Numpy 1.16.3

Scipy 1.2.1

Pandas 0.24.2

### Running simulations described in the paper: the simulations folder
All simulation code is in the 'simulations' folder within. Each of the variable manipulations are under separate folders. The easiest way to run any one of the simulations is to move to the simulation directory of interest :

```
cd effect_of_group_size
python group_size_simulations
```
By default all the simulations are set to run on all the available cores of the computer. If you'd like to change this to run serially on one core only please change the current ```Pool.map``` to the more standard ```map``` in the modules. If you'd like to change the number of cores that the module uses to run, please change the multiprocessing pool initiation lines from  ```Pool(multiprocessing.cpu_count())``` to ```Pool(<insert_number_of_cores_here>)```. 

*Note* : I've not seen the parallel running simulation outputs while using Spyder on Windows - this is [normal](https://stackoverflow.com/a/48099756/4955732). It's best to run the modules from terminal on Linux + Windows either way. 

### Simulation output:
By default a batch of 100 simulations are output as .pkl files. Each .pkl file has the following pattern:
```simulationtype_CPN_<uuidcode>_numpyseed_<seednumber>.pkl ```.
 Each pickle file has 100 simulations by default. The seed number in each pickle file refers to the numpy random seed used to generate the 
 positions, echo and call arrival times. In case a batch needs to re-run this numpy seed can be used to replicate the results. 

TODO : *describe the contents of the .pkl file* Each .pkl file is stored as a dictionary with the following 
 
### Simulation parameters
A common set of simulation parameters is used for most simulations - and only the variable of interest is altered in the 'effect_of' simulations. The common simulation parameters is stored in a Pickle file called 'commonsim_params.pkl' - and this can be changed using the 'create_and_svae_common_simparameters.py'. The wrapper code to change a single variable, run and save multiple simulations is called 'simulate_effect_of.py'. Within each simulation scenario this wrapper is called to alter only the variable of interest. 

**
