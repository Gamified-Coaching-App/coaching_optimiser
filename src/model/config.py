model_configs = [
    {
        'layers': [
                    #{'type': 'lstm', 'units': 128, 'return_sequences': True},
                    # {'type': 'lstm', 'units': 64, 'return_sequences': True},
                    # {'type': 'lstm', 'units': 64, 'return_sequences': True},
                    # {'type': 'flatten'},
                    # {'type': 'dense', 'units': 256, 'activation': 'selu'},
                    # {'type': 'dense', 'units': 128, 'activation': 'selu'},
                    # {'type': 'dense', 'units': 64, 'activation': 'selu'},
                    # {'type': 'lstm', 'units': 512, 'return_sequences': True},
                    # {'type': 'lstm', 'units': 256, 'return_sequences': True},
                    # {'type': 'lstm', 'units': 128, 'return_sequences': True},
                    {'type': 'lstm', 'units': 64, 'return_sequences': True},
                    {'type' : 'batch_normalization'},
                    {'type': 'lstm', 'units': 64, 'return_sequences': True},
                    {'type' : 'batch_normalization'},
                    {'type': 'lstm', 'units': 64, 'return_sequences': False},
                    {'type' : 'batch_normalization'},
                    # {'type': 'lstm', 'units': 64, 'return_sequences': True},
                    #{'type': 'lstm', 'units': 64, 'return_sequences': True},
                    # {'type': 'lstm', 'units': 10, 'return_sequences': False},
                    {'type': 'dense', 'units': 100, 'activation': 'relu', 'kernel_initializer': 'he_normal', 'bias_initializer':'ones'},
                    {'type' : 'batch_normalization'},
                    {'type': 'dense', 'units': 100, 'activation': 'relu', 'kernel_initializer': 'he_normal', 'bias_initializer':'ones'},
                    {'type' : 'batch_normalization'},
                    {'type': 'dense', 'units': 100, 'activation': 'relu', 'kernel_initializer': 'he_normal', 'bias_initializer':'ones'},
                    {'type' : 'batch_normalization'},
                    {'type': 'dense', 'units': 49, 'activation': 'relu', 'kernel_initializer': 'he_normal', 'bias_initializer':'ones'},
                    {'type': 'reshape', 'target_shape': (7, 7)},
                    {'type': 'output'}
                ], 
                'learning_rate': 0.01,
                'optimiser': 'adadelta'
    }
]
    # {
    #     'layers': [
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 49, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
    #                 {'type': 'activation', 'activation': 'relu'},
    #                 {'type': 'reshape', 'target_shape': (7, 7)}
    #             ], 
    #             'learning_rate': 1.0,
    #             'optimiser': 'adadelta'
    # },
    # {
    #     'layers': [
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 128, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 256, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 49, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
    #                 {'type': 'activation', 'activation': 'relu'},
    #                 {'type': 'reshape', 'target_shape': (7, 7)}
    #             ], 
    #             'learning_rate': 1.0,
    #             'optimiser': 'adadelta'
    # },
    # {
    #     'layers': [
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 32, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 49, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
    #                 {'type': 'activation', 'activation': 'relu'},
    #                 {'type': 'reshape', 'target_shape': (7, 7)}
    #             ], 
    #             'learning_rate': 1.0,
    #             'optimiser': 'adadelta'
    # },
    # {
    #     'layers': [
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 49, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
    #                 {'type': 'activation', 'activation': 'relu'},
    #                 {'type': 'reshape', 'target_shape': (7, 7)}
    #             ], 
    #             'learning_rate': 1.0,
    #             'optimiser': 'adadelta'
    # },
    # {
    #     'layers': [
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': True},
    #                 {'type': 'lstm', 'units': 49, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
    #                 {'type': 'activation', 'activation': 'relu'},
    #                 {'type': 'reshape', 'target_shape': (7, 7)}
    #             ], 
    #             'learning_rate': 1.0,
    #             'optimiser': 'adadelta'
    # }
    # ]

# 'layers': [
#             {'type': 'reshape', 'target_shape': (56, 10)},  # Reshape to (56, 10)
#             {'type': 'conv1d', 'filters': 10, 'kernel_size': 3, 'activation': 'selu', 'padding': 'same'},  # 1D conv on (10,)
#             {'type': 'batch_normalization'},
#             {'type': 'conv1d', 'filters': 10, 'kernel_size': 3, 'activation': 'selu', 'padding': 'same'},  # 1D conv on (10,)
#             #{'type': 'flatten'},  # Flatten the output
#             {'type': "lstm", 'units': 49, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},  # LSTM layer
#             {'type': 'reshape', 'target_shape': (7, 7)}  # Reshape to (56, 4)
#         ],
# 'layers': [
#             {'type': 'conv2d', 'filters': 16, 'kernel_size': (3, 3), 'activation': 'selu', 'input_shape': (56, 10, 1), 'padding': 'same'},
#             {'type': 'batch_normalization'},
#             {'type': 'max_pooling2d', 'pool_size': (2, 2)},  # Output shape (28, 5, 32)
#             {'type': 'conv2d', 'filters': 16, 'kernel_size': (3, 3), 'activation': 'selu', 'padding': 'same'},  # Output shape (28, 5, 32)
#             {'type': 'batch_normalization'},
#             {'type': 'max_pooling2d', 'pool_size': (2, 2)},  # Output shape (14, 2, 32)
#             {'type': 'conv2d', 'filters': 8, 'kernel_size': (3, 3), 'activation': 'selu', 'padding': 'same'},  # Output shape (14, 2, 8)
#             {'type': 'batch_normalization'},
#             {'type': 'flatten'},  # Output shape (14 * 2 * 8 = 224)
#             {'type': 'dense', 'units': 49, 'activation': 'relu'},  # Ensure this produces (batch_size, 49)
#             {'type': 'reshape', 'target_shape': (7, 7)}  # 49 elements to be reshaped to (7, 7)
#         ],
# model_configs = [
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 16, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 5.0
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 16, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 0.1
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 16, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 0.01
#             }, 
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 0.01
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 0.1
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 64, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 1.0
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 128, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 0.01
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 128, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ], 
#                 'learning_rate': 0.1
#             },
#             {
#                 'layers': [
#                     {'type': 'lstm', 'units': 128, 'dropout': 0.0, 'recurrent_dropout': 0.0, 'return_sequences': False},
#                     {'type': 'dense', 'units': 49, 'activation': 'relu'},
#                     {'type': 'reshape', 'target_shape': (7, 7)}
#                 ],
#                 'learning_rate': 1.0
#             },
# ]