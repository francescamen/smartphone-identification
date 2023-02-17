# Smartphone identification via passive traffic fingerprinting
This repository contains the reference code for the paper [''Smartphone Identification via Passive Traffic Fingerprinting: a Sequence-to-Sequence Learning Approach'' DOI: 10.1109/MNET.001.1900101](https://ieeexplore.ieee.org/document/9003304).

If you find the project useful and you use this code, please cite our paper:
```
@article{Meneghello2020Network,
	author={Francesca Meneghello and Michele Rossi and Nicola Bui},
	title={Smartphone Identification via Passive Traffic Fingerprinting: a Sequence-to-Sequence Learning Approach},
	journal={IEEE Network},
	volume={34},
	number={2},
	pages={112--120},
	year={2020}
}
```

# How to use
Clone the repository and enter the folder with the python code:
```bash
cd <your_path>
git clone https://github.com/francescamen/smartphone_identification
cd code
```

Download the input data from http://researchdata.cab.unipd.it/id/eprint/292, unzip and put them into the ```input_files``` folder.

## Train and test the framework
To create the smartphone fingerprints and uses them to correctly associate unknown traffic traces to the user labels execute the following commands: 
```bash
python data_loading_preprocessing.py 
python train_denoising_autoencoder.py <hidden_neurons> <layers> <epochs_RNN>
python run_encoder.py <hidden_neurons> <layers>
python words_clustering.py <num_clusters>
python users_identification.py <num_clusters> <epochs_CNN>
```
## Visualize the results
In the ```code``` folder you can find other utilities functions.
To visualize the performance of the autoencoder run the following command:
```bash
python plot_autoencoder.py <hidden_neurons> <layers>
```
The confusion matrix can be computed and plotted throught the command:
```bash
python confusion_matrix_analysis.py 
```
The users similarity assessment is performed by executing the following commands:
```bash
python users_disambiguation.py <num_clusters> <epochs_CNN>
python Bhattacharyya_distance.py <num_clusters> 
```

## Parameters
The results on the article are obtained with the following parameters: ```<hidden_neurons>=32``` ```<layers>=2``` ```<epochs_RNN>=100``` ```<num_clusters>=50``` ```<epochs_CNN>=100```.

# Authors
Francesca Meneghello, Michele Rossi, Nicola Bui

# Contact
meneghello@dei.unipd.it
github.com/francescamen
