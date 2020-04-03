MODEL_PARAMS = {
    'femnist.cnn': (248, 62), # input_dim, num_hiddens, output_dim, layer_index
    'femnist.cnn_prox': (248, 62), # input_dim, num_hiddens, output_dim, layer_index
    'shakespeare.stacked_lstm': (2048, 10, 5000, -2), # input_dim, num_hiddens, output_dim, layer_index
    'celeba.cnn':(1152, 2),
    'celeba.cnn_prox':(1152, 2)
}

VARIABLE_PARAMS = {
	'femnist.cnn': "dense_1/kernel",
    'femnist.cnn_prox': "dense_1/kernel",
	'shakespeare.stacked_lstm': "dense/kernel",
    'celeba.cnn':"dense/kernel",
    'celeba.cnn_prox':"dense/kernel"
}

