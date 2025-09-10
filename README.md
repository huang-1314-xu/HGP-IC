#HGP-IC
HGP-IC: Graph Neural Networks via Hotness-Based Partitioning and Information Compensation

##HGP-IC Model

<img width="1584" height="960" alt="image" src="https://github.com/user-attachments/assets/1ee92a42-6a3c-4bff-8b1a-7c465da6e989" />

## Requirement

- Python3.8
- Numpy
- Pandas
- PyTorch (>= 1.6.0)
- dgl


## Description of data and files

- **models (directory)**: The folder for storing the model.
- **config.py**: The configuration file for setting the input data location, model parameters and model storage path.
- **data_process**: For extracting the data of the selected center station and high correlated other stations, and transform the original data into the high dimensional matrix for matching the input structure of the model.
- **train.py**: It implements the reading parameters, data preparation and training procedure.
- **util**: This folder contains the core functions for generating the prediction task model, the loss function, and the partition loading function.

## Usage instructions

#### Configuration

All model parameters can be set in `config.py`, such as the learning rate, batch size, number of layers, kernel size, etc.

#### Training the model

```python
python trainL1.py
```

The program can automatically save the most accurate (with the lowest RMSE on validation set) model in the `models` directory.

#### Evaluation

```python
python evalL1.py
```

The saved model can be loaded and evaluating on the test set.
