# qkrig
Understanding the spatial autocorrelation of streamflow across the U.S.


## Structure

This repository should have code that follows, roughly, the following structure.

### src
This is where the bulk of the code lives. This is the code that does all the real work. There is currently a file called `camelskrig.py` that has code to load in discharge data for the CAMELS catchments and do a kriging analysis, inclding a variogram, interpolation and error variance plot.

### Notebooks
This is how to run and visualize the code for basic development and prototyping. There is currently an example called `camelskrig.ipynb` which runs the code for the CAMELS basins.

### Scripts
This is any code that does a quick and dirty task, such as generating configuration files, launching jobs on a high performance computer, or replacing some values in a bunch of files, etc. There is currently an example configuration `camelskrig.yaml` that has some options to do a kriging analysis on runoff for the CAMELS dataset.

### configs
This is all the basic information needed to run code for any specific purpose. This would be like options, directories, hyperparameters, etc.

### environment
This directory has information on what needes to be installed before the code in this repository can be run.