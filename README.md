# Electron density learn
These scripts are used to train a convolutional neural network (CNN) that predicts certain properties of a protein residue based on the electron density data around that residue.
Currently, the CNN estimates how far the $C_\alpha$ atom of the residue is from its correct position.

Default values of the arguments discussed below can be found in `src/config.py`.

## `generate_models.py`
The `generate_models.py` script is provided a txt file which is a comma-delimited list of PDB IDs.
These PDB IDs should correspond to models that are considered highly-refined (for example, all proteins with a refinement resolution better than some given value).

The script will then download the mtz file for each model and dispose of all reflections data _better_ than some given value (the default is 3.5 angstroms).

It will then use the modelcraft software to generate a model based on this truncated data.

## `extract_training_data.py`
This script will iterate through the models generated by `generate_models.py` and extract the training data that will be fed to the CNN.

1 element of training data is extracted from each residue within the generated models. 

The input (features) of the training data is electron density data centred on the carbon alpha atom of the residue. This electron density is sampled at an $N\times N\times N$ grid centred on a residue, where $N$ can be set as an argument to this script. 

Furthermore, the orientation of this grid is standardised. Denote the positions of the carbon alpha atom, nitrogen and carbonyl carbon as $\mathbf{p}_C_\alpha$, $\mathbf{p}_N$ and $\mathbf{p}_C$ respectively, and then we define the following basis vectors:
$$
\mathbf{x} = \mathbf{p}_C - \mathbf{p}_{C_\alpha} \\
\mathbf{y} = \mathbf{x} \times (\mathbf{p}_N - \mathbf{p}_{C_\alpha}) \\
\mathbf{z} = \hat{\mathbf{x}} \times \hat{\mathbf{y}}
$$
The grid is defined by these basis vectors. The electron density maps used are an argument to this script.

The output (target) of the training data is the vector from the carbon alpha atom of the residue in the generated model, to the carbon alpha atom of the corresponding residue of the _reference_ model.

The data is written to a hdf5 file.

## `train_cnn.py`
This script trains a convolutional neural network based on the training data generated by `extract_training_data.py`.

