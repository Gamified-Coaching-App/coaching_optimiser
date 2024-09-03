"""
model configs for testing and evaluation
"""
model_configs = [
    {
        'name': 'LSTM+Dense',
        'layers': [
            {'type': 'lstm', 'units': 64, 'return_sequences': True},
            {'type': 'lstm', 'units': 64, 'return_sequences': True},
            {'type': 'lstm', 'units': 64, 'return_sequences': True},
            {'type': 'lstm', 'units': 64, 'return_sequences': False},
            {'type': 'dense', 'units': 28, 'bias_initializer':'ones'},
            {'type': 'reshape', 'target_shape': (7, 4)},
            {'type': 'output'}  
        ],
        'learning_rate': 1.0,
        'weight_decay': 0.0,
        'optimiser': 'adadelta',
        'batch_size': 64
    }
]