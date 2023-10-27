[![Documentation Status](https://readthedocs.org/projects/photonlib/badge/?version=latest)](https://photonlib.readthedocs.io/en/latest/?badge=latest)

# Photon Library
This is a python API to use Photon Library. This README describes how to install and where to find tutorials. For users, you might find a complimentary documentation at the [ReadTheDocs](https://photonlib.readthedocs.io/en/latest/). For developers, make sure you read the [Contribution Guide](/contributing.md).

## Installation
Once `git clone` this repository, go inside and:
```
pip install .
```
After installation, you may need to download a data file and a tutorial notebook.
These supportive materials are gathered in a publicly accessible folder in [this google drive link](https://drive.google.com/drive/folders/1IjRUMMVW7aiGWGcZFGRb9nT8dCRVYolE?usp=share_link).

### Downloading a data file
As explained below, Photon Library is a look-up table. To use, you have to download the table content data file.
You can download and use the ICARUS data file as an example.
After installation, executing the command below will download this datafile `plib.h5` in your current path:
```
download_icarus_plib.sh
```

### Simple tutorial

Notebook is coming soon...

## What is Photon Library?

Photon Library refers to the technique used by neutrino experiments with Liquid Argon Time Projection Chambers (LArTPCs).
Physics events (signal) in a LArTPC produce lots of photons (~20k/MeV) isotropically, and some of them are observed by optical detectors.

A typical physics event produces 100 million or sometimes more than billions of photons.
Modeling the transportation of every single photon from the production point to individual optical detector with a monte-carlo simulation 
(i.e. calculating every possible physics processes explicitly) take prohibitive amount of time.

Instead, experiments pre-calculate the visibility, namely the probability for a photon produced at a position R to be observed by an optical detector D.
Photon Library is a look-up table that stores the visibility values for the detector volume and all optical detectors in the detector.
In this scheme, when simulating physics events, we can immediately estimate how many photons are detected by skipping calculating explicit physics processes.

As it is a table, Photon Library discretizes positions in the detector and this means it loses some spatial resolution.
We cal each entity a "voxel" (volume-pixel).
The sides of each voxel is uniform along each axis, and typically the same across axis (i.e. a voxel is typically a cube).
For example, in the ICARUS experiment, a voxel is a cube of 5 cm.


## How is Photon Library generated?
coming soon
