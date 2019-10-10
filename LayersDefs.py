import ROOT
from ROOT import TGraph
from ROOTDefs import tau_data_directory, tau_signal_layer_file, tau_background_layer_file
from NNDefs import build_and_train_network
import os
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import uproot

def get_signal_and_background_frames(sig_file_path=None, back_file_path=None):
    '''
    Get standard tau analysis signal and background files from predetermined directories and load them into Pandas dataframes. Also populate new IsSignal column for both. If no paths provided for the files then system environment variables are checked.
    :param sig_file_path: Path to the signal layer file
    :param back_file_path: Path to the background layer file
    :return signal_frame: Pandas dataframe holding layer Ets for signal events plus new IsSignal column set to 1
    :return background_frame: Pandas dataframe holding layer Ets for background events plus new IsSignal column set to 0
    '''
    if sig_file_path is None:
        sig_file_path = os.path.join(tau_data_directory(), tau_signal_layer_file())
    if back_file_path is None:
        back_file_path = os.path.join(tau_data_directory(), tau_background_layer_file())
    
    fsig = uproot.open(sig_file_path)
    tsig = fsig['mytree']
    signal_frame = tsig.arrays(['L0Et', 'L1Et', 'L2Et', 'L3Et', 'HadEt'], outputtype=pd.DataFrame)
    signal_frame['IsSignal'] = 1

    fback = uproot.open(back_file_path)
    tback = fback['mytree']
    background_frame = tback.arrays(['L0Et', 'L1Et', 'L2Et', 'L3Et', 'HadEt'], outputtype=pd.DataFrame)
    background_frame['IsSignal'] = 0

    return signal_frame, background_frame

def shuffle_frames(input_frame, true_output_frame, test_size=0.2):
    '''
    Split input and output dataframes into training and test samples. Also reset their dataframe indices

    :param input_frame: Pandas dataframe holding the input events
    :param true_output_frame: Pandas dataframe holding the true output events
    :param test_size: Percentage of events to be put into the test set, remainder into the training set
    :return input_train: Training sample of input events
    :return input_test: Test sample of input events
    :return true_output_train: Training sample of true output values
    :return true_output_test: Test sample of true output values
    '''
    # Split layers and identifiers into training and testing sets
    input_train, input_test, true_output_train, true_output_test = train_test_split(input_frame, true_output_frame,
                                                                                    test_size=test_size)

    # Reset indices of new frames
    input_train = input_train.reset_index(drop=True)
    true_output_train = true_output_train.reset_index(drop=True)
    input_test = input_test.reset_index(drop=True)
    true_output_test = true_output_test.reset_index(drop=True)

    return input_train, input_test, true_output_train, true_output_test

def calculate_derived_et_column(frame, layer_weights=[1, 1, 1, 1, 1], column_names=['L0Et', 'L1Et', 'L2Et', 'L3Et', 'HadEt'],
                       output_column_name='TotalEt'):
    '''
    Create new derived Et column in frame which is the dot product of the given layer weights and the values of the given
        column names

    :param frame: Pandas dataframe holding event Ets
    :param layer_weights: List of weights to be applied to calculate derived Et
    :param column_names: List of names of columns to be weighted and summed to get derived Et
    :param output_column_name: String name of frame column where derived Et value is stored
    :return: None
    '''
    if len(layer_weights) != len(column_names):
        raise Exception('calculate_derived_et requires the layer_weights and column_names arguments to be of the same length')

    frame[output_column_name] = 0
    for i in range(len(layer_weights)):
        frame[output_column_name] += layer_weights[i] * frame[column_names[i]]

def calculate_derived_et_columns(signal_frame, background_frame, layer_weights=[1, 1, 1, 1, 1], column_names=['L0Et', 'L1Et', 'L2Et', 'L3Et', 'HadEt'],
                        output_column_name='TotalEt'):
    '''
    Calculate derived Et column for signal and background frame

    :param signal_frame: Pandas dataframe holding signal events
    :param background_frame: Pandas dataframe holding background events
    :param layer_weights: Layer weights to be applied when calculating derived Et
    :param column_names: List of names of columns to be weighted and summed to get derived Et
    :param output_column_name: String name of frame column where derived Et value is stored
    :return: None
    '''
    if len(layer_weights) != len(column_names):
        raise Exception('calculate_derived_et requires the layer_weights and column_names arguments to be of the same length')

    calculate_derived_et_column(signal_frame, layer_weights, column_names, output_column_name)
    calculate_derived_et_column(background_frame, layer_weights, column_names, output_column_name)

def build_roc_cuts_from_frames(signal_frame, background_frame, netcuts=100, target_90percent_signal=False):
    # If we are interested in a very precise 90% signal efficiency calculation, then set max at overall average to allow
    #   for finer cuts at smaller values where we're interested
    # Else set max at overall max
    if target_90percent_signal:
        max_value = np.mean([np.mean(signal_frame.values), np.mean(background_frame.values)])
    else:
        max_value = max([max(signal_frame.values), max(background_frame.values)])

    # Minimum should always be the overall minimum
    min_value = min([min(signal_frame.values), min(background_frame.values)])

    # Compute how much to increment our cut value each time
    scaler = float(max_value - min_value) / netcuts

    # Create list of cut values
    cuts = [min_value + i * scaler for i in range(netcuts)]

    return cuts

def roc_cuts(signal_frame, background_frame, cuts=None, netcuts=100, target_90percent_signal=False, return_efficiencies=True):
    '''
    Create arrays of efficiencies after Et cuts to be used in creating a ROC curve. Input frames must contain only one
    column in order to encourage best performance practices.

    :param signal_frame: Pandas dataframe holding signal events with column 'TotalEt'
    :param background_frame: Pandas dataframe holding background events with column 'TotalEt'
    :param cuts: List of floats containing the explicit values to cut on
    :param target_90percent_signal: Boolean denoting whether to set max value of cuts at the average, allowing finer
        cuts near the 90% signal efficiency
    ;param return_efficiencies: Boolean denoting whether to return efficiency results of cuts, if False return event cuts
    :return sig_out: Numpy array holding either percentage or number of signal events remaining after each cut
    :return back_out: Numpy array holding either percentage or number of background events remaining after each cut
    '''
    # Sanity check that input dataframes have only one column
    if len(signal_frame.columns) != 1 or len(background_frame.columns) != 1:
        raise Exception('Input frames to roc_efficiencies() must contain only one column')

    # If not given explicit list of cuts then construct from data
    if cuts == None:
        cuts = build_roc_cuts_from_frames(signal_frame, background_frame, netcuts, target_90percent_signal)

    # Create zeroed arrays to hold number of events left after each cut
    sig_cuts = np.zeros(netcuts)
    back_cuts = np.zeros(netcuts)

    # Make cuts on each value and compute how many events are left
    for i, cut in enumerate(cuts):
        # Create list holding 1 for each event with value greater than cut value and 0 for each event not
        sig_greater = [1 if et >= cut else 0 for et in signal_frame[signal_frame.columns[0]]]
        # Sum the above to get total number of events that passed the cut
        sig_cuts[i] = sum(sig_greater)

        # See above with signal events
        back_greater = [1 if et >= cut else 0 for et in background_frame[background_frame.columns[0]]]
        back_cuts[i] = sum(back_greater)

    # Returning either event efficiencies or event counts after each cut, depending on value of return_efficiencies
    if not return_efficiencies:
        sig_out = sig_cuts
        back_out = back_cuts
    else:
        # Divide by total number of events in the frames to convert event counts into efficiencies
        sig_eff = sig_cuts / len(signal_frame.values)
        back_eff = back_cuts / len(background_frame.values)

        if sig_eff[-1] > 0.9 and target_90percent_signal:
            print('Never achieved below 90% signal efficiency')

        sig_out = sig_eff
        back_out = back_eff

    return sig_out, back_out

def background_eff_at_target_signal_eff(signal_frame, background_frame, cut_column, target_signal_eff=0.9):
    '''
    Find the background efficiency at which the signal efficiency is equal to the given target efficiency

    :param signal_frame: Pandas dataframe of signal events
    :param background_frame: Pandas dataframe of background events
    :param cut_column: String name of the column to cut on to produce efficiencies
    :param target_signal_eff: Efficiency of the signal at which to take the background efficiency
    :return: Background efficiency when the signal achieves the target efficiency
    '''
    signal_events = len(signal_frame)

    # Find number of signal events that must be cut to fall below 90% efficiency
    sig_events_to_cut = math.ceil(signal_events * (1 - target_signal_eff))

    # Sort signal by the column to be cut on
    sorted_signal_et = signal_frame[[cut_column]].sort_values(by=[cut_column])

    # Find the cutoff Et that produces the 90% remaining signal efficiency
    cutoff = sorted_signal_et.iloc[sig_events_to_cut - 1][cut_column]

    # Find the number of background events that survive the previously calculated Et cut
    background_above_cutoff_each = [1 if value > cutoff else 0 for value in background_frame[cut_column]]
    background_above_cutoff = sum(background_above_cutoff_each)

    # Convert event number to efficiency
    end_background_efficiency = background_above_cutoff / len(background_frame)

    return end_background_efficiency

def roc_curve(signal_frame, background_frame, netcuts):
    '''
    Create a ROC curve from reconstructed Et cuts on signal and background eventsl. Frames must have only a single
        column to encourage best performance practices.

    :param signal_frame: Pandas dataframe holding signal events with TotalEt column
    :param background_frame: Pandas dataframe holding background events with TotalEt column
    :param netcuts: Number of cuts to make in creating the ROC curve

    :return: TGraph object holding the ROC curve
    '''
    signal_efficiencies, background_efficiencies = roc_cuts(signal_frame, background_frame, netcuts=netcuts)

    gr = TGraph(netcuts, signal_efficiencies, background_efficiencies)

    return gr

def train_and_predict_frame_nn(layer_frame, truth_frame, test_size=0.2, epochs=30, use_bias=True, class_nn=True, lr=0.1,
                               hidden_layers=0, hidden_nodes=8, class_weight=None):
    '''
    DEPRECATED: predict_nn_on_all now performs all functionality without a call to this function
    From separate Pandas dataframes holding the layer and truth information, build and train a neural network to distinguish
        signal from background.

    :param layer_frame: Pandas dataframe holding layer Ets
    :param identifier_frame: Pandas dataframe holding truth value e.g. true Et or signal/background identifier
    :param test_size: Percentage of events to use as a test dataset
    :param epochs: Number of epochs to train the network for
    :param use_bias: Whether to use a bias value when training the network or not
    :param class_nn: Whether to train a classification network or a regression network
    :param lr: Learning rate of network

    :return predicted_values_frame: Pandas dataframe holding the predicted values of the network being evalueate on all events
    :return model: Keras model
    '''
    # Split layers and identifiers into training and testing sets
    layers_train, layers_test, truth_train, truth_test = shuffle_frames(layer_frame, truth_frame,
                                                                                  test_size=0.2)

    model = build_and_train_network(layers_train, truth_train, layers_test, truth_test, is_class_nn=class_nn, lr=lr,
                                    epochs=epochs, use_bias=use_bias, hidden_layers=hidden_layers, hidden_nodes=hidden_nodes,
                                    class_weight=class_weight)

    # Apply trained model to all events
    predicted_values_frame = pd.DataFrame(model.predict(layer_frame), columns=['NNOutputValue'])

    return predicted_values_frame, model

def predict_nn_on_all_frame(all_frame, nn_input_columns, truth_column, epochs=30, use_bias=True, is_class_nn=True, lr=0.1,
                            hidden_layers=0, hidden_nodes=8, class_weight=None):
    '''
    Given a Pandas dataframe holding input and truth output of all signal and background events, train neural network then
        predict binary classification of all events in event frame.

    :param all_frame: Pandas dataframe holding input and truth output of all events
    :param nn_input_columns: List of dataframe column names to be fed as input into neural network
    :param truth_column: List of single dataframe column name holding truth output of event. Must be 1/0
    :param epochs: Number of epochs for the neural network while training
    :param use_bias: Boolean passed throught to keras fit(), whether to train a bias value
    :param class_nn: Boolean determining whether to train a classification or regression network
    :param lr: Float passed through to keras fit(), learning rate of network
    :param hidden layers: Integer number of hidden layers to include in the network
    :param hidden_nodes: Integer number of nodes in each hidden layer
    :param class_weight: Dictionary passed through to keras fit(), weighting for each classifier value

    :return predicted_signal_frame: Pandas dataframe with single column holding predicted classifier value for each signal event
    :return predicted_background_frame: Pandas dataframe with single column holding predicted classifier value for each background event
    '''
    # Split total frame into those with layer Ets and one with signal/background identifier
    layer_frame = all_frame[nn_input_columns]
    truth_frame = all_frame[truth_column]

    # Split layers and identifiers into training and testing sets
    layers_train, layers_test, truth_train, truth_test = shuffle_frames(layer_frame, truth_frame,
                                                                                  test_size=0.2)

    model = build_and_train_network(layers_train, truth_train, layers_test, truth_test, is_class_nn=is_class_nn, lr=lr,
                                    epochs=epochs, use_bias=use_bias, hidden_layers=hidden_layers, hidden_nodes=hidden_nodes,
                                    class_weight=class_weight)

    # Apply trained model to all events
    predicted_values_frame = pd.DataFrame(model.predict(layer_frame), columns=['NNOutputValue'])

    # # Run frames through neural network and return predicted values of entire dataset
    # predicted_values_frame, model = train_and_predict_frame_nn(layer_frame, truth_frame, epochs=epochs, use_bias=use_bias,
    #                                                            class_nn=class_nn, lr=lr, hidden_layers=hidden_layers,
    #                                                            hidden_nodes=hidden_nodes, class_weight=class_weight)

    # Break all predicted values back into signal and background frames to construct ROC curves
    predicted_signal_frame = predicted_values_frame.iloc[list(all_frame['IsSignal'] == 1)]
    predicted_background_frame = predicted_values_frame.iloc[list(all_frame['IsSignal'] == 0)]

    return predicted_signal_frame, predicted_background_frame, model

def manual_train_1d(signal_frame, background_frame, column_names, weights, target_eff=0.9):
    '''
    Manually search through weights for a single value, calculating the background efficiency at a given signal efficiency.
        Because only the ratio of weights is important, one weight can be fixed to 1. This therefore involves two columns
        being combined to create the total weighted Et.

    :param signal_frame: Pandas dataframe holding energies of signal events
    :param background_frame: Pandas dataframe holding energies of background events
    :param column_names: List of column names to be combined into total weighted Et, the first column weight is fixed to 1 and the second
        column weight varies
    :param weights: List of weights to use to calculate efficiencies
    :param target_eff: Float target signal efficiency, store background efficiency when achieved
    :return: List of float background efficiencies for each weight
    '''
    background_efficiencies = np.ones([len(weights)])

    for i, weight in enumerate(weights):
        calculate_derived_et_columns(signal_frame, background_frame, layer_weights=[1, weight],
                                     column_names=column_names, output_column_name='WeightedEt')

        end_background_efficiency = background_eff_at_target_signal_eff(signal_frame, background_frame, 'WeightedEt', target_eff)

        background_efficiencies[i] = end_background_efficiency

    return background_efficiencies

def min_manual_eff_1d(efficiencies, weights):
    '''
    Find the minimum efficiency given a list of efficiencies and a list of the weights that produced those efficiencies
        by the manual_train_1d function

    :param efficiencies: List of float efficiencies produces by manual_train_1d with the given weights
    :param weights: List of float weights that produced the given efficiencies when run through manual_train_1d
    :return min_eff: The minimum efficiency found in efficiencies
    :return min_weight: The weight that produced the minimum efficiency
    '''
    min_eff = float('inf')
    min_weight = 0

    for eff, weight in zip(efficiencies, weights):
        if eff == 0:
            continue
        if eff < min_eff:
            min_eff = eff
            min_weight = weight

    return min_eff, min_weight

def manual_train_2d(signal_frame, background_frame, column_names, weights, target_eff=0.9):
    '''
    Manually search through weights for two values, calculating the background efficiency at a given signal efficiency.
        Because only the ratio of weights is important, one weight can be fixed to 1. This therefore involves three columns
        being combined to create the total weighted Et.

    :param signal_frame: Pandas dataframe holding energies of signal events
    :param background_frame: Pandas dataframe holding energies of background events
    :param column_names: List of column names to be combined into total weighted Et, the first column weight is fixed to
        1 and the second and third column weights vary
    :param weights: List of weights to use to calculate efficiencies
    :param target_eff: Float target signal efficiency, store background efficiency when achieved
    :return: 2D ist of float background efficiencies for each weight combination
    '''
    steps = len(weights)
    signal_efficiencies = np.ones([steps, steps])

    for i, first_weight in enumerate(weights):
        for j, second_weight in enumerate(weights):
            calculate_derived_et_columns(signal_frame, background_frame, layer_weights=[1, first_weight, second_weight],
                                         column_names=column_names, output_column_name='WeightedEt')

            end_background_efficiency = background_eff_at_target_signal_eff(signal_frame, background_frame, 'WeightedEt',
                                                                            target_eff)

            signal_efficiencies[i][j] = end_background_efficiency

    return signal_efficiencies

def min_manual_eff_2d(efficiencies, weights):
    '''
    Find the minimum efficiency given a 2D list of efficiencies and a list of the weights that produced those efficiencies
        by the manual_train_2d function

    :param efficiencies: 2D List of float efficiencies produces by manual_train_1d with the given weights
    :param weights: List of float weights that produced the given efficiencies when run through manual_train_2d
    :return min_eff: The minimum efficiency found in efficiencies
    :return min_first_weight: The first weight that produced the minimum efficiency
    :return min_second_weight: The second weight that produced the minimum efficiency
    '''
    min_eff = float('inf')
    min_first_weight = 0
    min_second_weight = 0

    for eff_list, first_weight in zip(efficiencies, weights):
        for eff, second_weight in zip(eff_list, weights):
            if eff == 0:
                continue
            if eff < min_eff:
                min_eff = eff
                min_first_weight = first_weight
                min_second_weight = second_weight

    return min_eff, min_first_weight, min_second_weight
