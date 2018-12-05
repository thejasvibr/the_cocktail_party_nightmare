# the_cocktail_party_nightmare
Code associated with the manuscript 'Quantifying the nightmare in the sonar cocktail party' - Thejasvi Beleyur, Holger R Goerlitz



[![DOI](https://zenodo.org/badge/114679151.svg)](https://zenodo.org/badge/latestdoi/114679151)

### A brief description of the modules in this repository:


*parallelising the CPN* : Simulates the cocktail party nightmare across a wide range of echo numbers and conspecific call numbers. 
The output is a pickle file with multiple arrays. The simulation runs 10000 times for each echo number across a range of conspecific call numbers. Each echo number has a set Nechoes x 10000 numpy array associated with it. Each row is a one-hot coded representation of which echoes are heard in a given simulation run. This is the raw data behind the Nconspecific calls vs Nechoes heatmaps in figure 1

**
