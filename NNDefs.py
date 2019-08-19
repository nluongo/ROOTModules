import os
import numpy as np
import pandas as pd
from tensorflow import keras, set_random_seed


def build_and_train_network(train_ets, train_sig_back, test_ets, test_sig_back, is_class_nn=True, lr=0.1, epochs=10, use_bias=True,
                             hidden_layers=0, hidden_nodes=8, class_weight=None):
    if isinstance(train_ets, pd.DataFrame):
        input_nodes = len(train_ets.columns)
    else:
        input_nodes = len(train_ets[0])

    model = build_network(is_class_nn, input_nodes, hidden_layers, hidden_nodes, use_bias)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ets, train_sig_back, epochs=epochs, class_weight=class_weight)

    test_loss, test_acc = model.evaluate(test_ets, test_sig_back)

    print('Test accuracy: ', test_acc)
    
    return model

def build_network(is_class_nn, input_nodes, hidden_layers, hidden_nodes, use_bias):
    if is_class_nn == True:
        end_activation = 'sigmoid'
    else:
        end_activation = None

    # Create dictionary of different keras models parametrized by the number of hidden layers
    models = {}

    # Populate dictionary with models with up to two hidden layers, with the activation function of the last layer being
    #   sigmoid if a classification network and None if a regression network
    models[0] = keras.Sequential([
       keras.layers.Dense(1, input_shape=(input_nodes,), activation=end_activation, use_bias=use_bias),
    ])

    models[1] = keras.Sequential([
       keras.layers.Dense(hidden_nodes, input_shape=(input_nodes,), activation='relu'),
       keras.layers.Dense(1, activation=end_activation)
    ])

    models[2] = keras.Sequential([
       keras.layers.Dense(hidden_nodes, input_shape=(input_nodes,), activation='relu'),
       keras.layers.Dense(hidden_nodes, activation='relu'),
       keras.layers.Dense(1, activation=end_activation)
    ])

    return models[hidden_layers]

def get_layer_weights_from_txt(config_num):
    '''
    Return the neural network trained weights stored in the LayerWeights.txt file based on the number corresponding to
        the desired network configuration. If not working correctly, check that the information in LayerWeights.txt is
        formatted correctly

    :param config_num: Integer correesponding to the desired network configuration in the LayerWeights.txt file
    :return: A list of float layer weights and a float bias value

    '''

    layer_weights_file_path = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'NeuralNets', 'LayerWeights.txt')
    layer_weights_file = open(layer_weights_file_path, 'r')

    all_lines = layer_weights_file.readlines()

    config_string = str(config_num)
    for line_num, line in enumerate(all_lines):
        if line[0:3] == config_string + ' -' or line[0:4] == config_string + ' -':
            start_pos = line_num
            break

    weight_line_num = start_pos + 2
    bias_line_num = start_pos + 4

    weight_line = all_lines[weight_line_num]
    layer_weights = [float(i) for i in weight_line.split(',')]

    bias_line = all_lines[bias_line_num]
    bias = float(bias_line)

    return layer_weights, bias

def apply_layer_weights(tree1, tree2, config_num):

    layer_weights, bias = get_layer_weights_from_txt(config_num)
    print(layer_weights)
    print(bias)

    tree1.set_reco_et_layer_weights(layer_weights)
    tree1.set_reco_et_shift(bias)

    tree2.set_reco_et_layer_weights(layer_weights)
    tree2.set_reco_et_shift(bias)

    return None

def train_test_split(to_split, split_fraction=0.8):
    '''
    Split to_split list into a train list and test list and return both as numpy arrays. Default train/test split is 80/20.
        This assumes that to_split has already been sufficiently randomized.

    :param to_split: List of events to split into train and test arrays
    :param split_fraction: Fraction in decimal form of to_split to be placed in train array
    :return: Numpy arrays of train and test arrays
    '''
    split_fraction = 0.8
    event_num = len(to_split)

    test_train_cut = int(split_fraction * event_num)

    train_split = to_split[0:test_train_cut]
    test_split = to_split[test_train_cut:]

    train_split = np.array(train_split)
    test_split = np.array(test_split)

    return train_split, test_split

def flat_start_line(flat_file_path):
    '''
    Get the line number where actual data begins, skipping over file summary info. Assumes that all file summary info
        lines begin with **

    :param flat_file_path: Path to flat file
    :return: Integer, line number where data begins
    '''
    with open(flat_file_path, 'r') as f:
        line_num = 0
        for line in f:
            if line[0:2] == '**':
                line_num += 1
            else:
                break
    return line_num

def prepared_flat_file_lines(flat_file_path):
    '''
    Return lines from flat file at given path. Also remove initial file summary lines

    :param flat_file_path: Path where flat file resides
    :return: List of lines from flat file
    '''
    start_line = flat_start_line(flat_file_path)

    with open(flat_file_path, 'r') as f:
        all_lines = f.readlines()

    all_lines = all_lines[start_line:]

    return all_lines