from preprocessor import Preprocessor
from autoencoder import run_and_evaluate

config = {
    # Data parameters
    'days': 14, 

    # Model architecture
    'units_first_layer': 1000,  # Number of units in the first layer of the encoder
    'number_of_layers': 2,           # Number of layers in the encoder and decoder
    'unit_decline': 0.99,      # Share per layer to reduce the number of units by

    # Training parameters
    'num_epochs': 50,         # Number of epochs for training
    'batch_size': 512,        # Batch size for training
    'learning_rate': 0.0001,   # Learning rate for the optimizer
    'optimizer': 'Adam',      # Type of optimizer (e.g., 'adam', 'adadelta', 'sgd')
    'optimizer_params': {'beta_1':0.5, 'beta_2':0.99999, 'epsilon':1e-07, 'decay':0.1}    # Additional parameters for the optimizer
}

def run():
    preprocessor = Preprocessor(config)
    X_train, X_test = preprocessor.preprocess(config.get('days'))
    run_and_evaluate(X_train, X_test, config)

if __name__ == "__main__":
    run()
