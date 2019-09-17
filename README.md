# The Cocktail Party Nightmare
Code associated with the manuscript 'Quantifying the nightmare in the sonar cocktail party' - Thejasvi Beleyur, Holger R Goerlitz

[![DOI](https://zenodo.org/badge/114679151.svg)](https://zenodo.org/badge/latestdoi/114679151)

### What is the repository about?
The code in this repository simulate what a bat flying in a group of other bats may be experiencing. Echolocating bats fly in the night and live in caves where they cannot 'see' objects with their eyes like we do. They emit loud calls and listen to the returning echoes to detect their surroundings. See [here](https://www.cell.com/current-biology/fulltext/S0960-9822(05)00686-X) for a quick review. 

Echolocation works very well when individual bats are flying around alone while hunting or commuting. However, when there are  loud sounds, such as those from other bats - the faint returning echoes may not be heard. This means that when a bat flies close to many other bats - it may be flying metaphorically 'blind'. However, many bat species are very social, and live in large groups, fly around together in caves and even emerge in the millions - the Mexican Free-Tailed bat is a classic example. 

The problem of listening to a target signal in the presence of louder sounds that affect signal detection has been called the 'cocktail party problem' by [Cherry 1953](https://asa.scitation.org/doi/10.1121/1.1907229), (also the [wikipedia article](cocktail party problem/effect](https://en.wikipedia.org/wiki/Cocktail_party_effect))). In the context of bat echolocation, given the orders of magnitude louder bat calls and the large number of calls and echoes that need to be dealt with when flying in a group - [Ulanovsky & Moss 2008](https://www.pnas.org/content/105/25/8491.short) termed group echolocation a 'cocktail party nightmare'. 

This repository simulates the auditory detection of echoes, sound propagation and other relevant phenomena that occur in the cocktail party nightmare and quantifies how many neighbours a bat may be detecting as it flies in a group. 


### Requirements:
The code runs on a range of Python 2.7 versions. The code runs on Windows and Linux. 
To replicate the development environment exactly - install the following versions in your conda/virtual environment:
Python 2.7.12

Numpy 1.16.3

Scipy 1.2.1

Pandas 0.24.2


### Running simulations described in the paper: the simulations folder
All simulation code is in the 'simulations' folder. Every set of simulations is defined by a parameter file which sets the number of simulation runs, call properties, group size and other relevant parameters. The simulations themselves are run by calling the ```run_simulations``` module through a command line call. Let's go through it step by step.


#### Step 1 : Setting up a simulations 
The simulations parameter files define what kind of simulations are run and how many simulation runs will occur. 
Within each simulation folder  - the 'make_<simulation_type>.py' modules create the parameter files for differenct simulation scenarios. 

##### What is a parameter file?
A parameter file consists of a dictionary with simulation parameters. 
The file is created by saving the dictionary through the [dill](https://pypi.org/project/dill/) package. Having setup the parameters you can simply run the 'make.. .py' file and you should get a bunch of ```.paramset``` files in your folder. 
The ```make_<simulation_type>.py``` modules in each simulations scenario ('effect_of_group_size, effect_of_position, multivariable_simualtions') are the reference examples. 

#### Creating parameter files
To start with if you want to recreate the simulations that have been run in the paper, you can directly run the ```make_<simulation_type>.py``` module directly through an IDE like Spyder or run it from the command line (Ctrl+Alt+T in Ubuntu or Command Prompt in Windows) with the following commands. 
```
# move into the directory where the make .py module is. 
cd simulations/effect_of_group_size
python make_params_groupsize.py
```
Note : The #'s are comments - please do not copy-paste them into your comm

Here you should see multiple ```.paramset``` files, each of which has the parameters required to initiate simulations for the different group sizes. 

#### Step 2 : Starting the simulations
The simulations are initiated through the ``` run_simulations.py``` module in the simulations folder. The ```run_simulations``` module initiates the appropriate number of simulations runs as described by each parameter file in a folder, and saves the simulation outputs into the given destination folder. 

#### Starting simulations runs 
```
#Change directories to the 'simulations' folder. 
cd the_cocktail_party_nightmare/simulations/
# run effect of group size simulations 
python run_simulations -name "groupsize_effect" -info "a trial run to see if things work okay" -param_file "effect_of_group_size/*.paramset" -dest_folder "effect_of_group_size/" -numCPUS 2
```

The ```-name``` argument will set the prefix for all simulation output files. Here all outputs will start with 'groupsize_effect'.

The ```-info``` argument will be saved into the simulation output so you can quickly refer to what it is that was being run without actually loading the information. 

The ```-param_file``` argument refers to the folder and file format of the parameter files to be used to initiate the simulations. 

The ```-dest_folder``` argument refers to the destination folder for all the simulation outputs. 

The ```-numCPUS``` refers to the number of CPUs that are to be used to run the simulations in parallel. Simulations with larger group sizes (>50 bats) can be intensive and make an impact on your user-experience if you plan to do other things. It may then be wise to limit the number of CPUs that are being used for the simulations. The number of CPUs defaults to the number available on the device if not specified. 

#### What a succesful simulation initiation looks like : 
[](docs_imgs/Screenshot from 2019-09-17 15-44-27.png)


#### Step 3 : 
