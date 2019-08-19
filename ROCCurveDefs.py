import ROOT
from ROOT import TGraph
from ROOTDefs import prepare_event, prepared_flat_file_lines
import numpy as np
from math import exp

def et_identity(et):
    return et

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def increment_roc_counter(val, counter_array, scaler = 1, min_val = 0):
    '''
    Loop through counter_array and increment the ith element if val > i / scaler. Used for creating histograms that show events
        remaining after varying cuts.
    Examples: If len(counter_array) > 10, then increment_roc_counter(10, counter_array) would cause the first 10 elements
        of counter_array to be incremented by 1
    Scaler allows for non-integer cuts, so if you wanted to cut on .1 increments, then set scaler = .1

    :param et: Value for which ith elements of counter_array for which i / scaler < val will be incremented
    :param counter_array: Array whose ith elements such that i / scaler < val will be incremented
    :param scaler: Scale factor between index increment (1) and cut increments
    '''
    for i in range(len(counter_array)):
        if val > (float(min_val) + (float(i) * scaler)):
            counter_array[i] += 1

def roc_efficiencies_from_cuts(sig_cuts, back_cuts):
    '''
    Take arrays holding the number of events left in a sample after arbitrary cuts on e.g. reconstructed Et and convert
        to efficiencies i.e. percentage of events left after those same cuts
    NOTE: netcuts was removed as an input because it should be the same as len(sig_cuts)

    :param sig_cuts: Array holding signal events remaining after cuts
    :param back_cuts: Array holding background events remaining after cuts
    :param netcuts: Number of cuts
    :return sig_eff: Array holding signal efficiencies remaining after cuts
    :return back_eff: Array holding background efficiencies remaining after cuts
    '''
    netcuts = len(sig_cuts)

    sig_eff = np.zeros(netcuts)
    back_eff = np.zeros(netcuts)

    if sig_cuts[0] == 0:
        print('No signal events')
    if back_cuts[0] == 0:
        print('No background events')

    for i in range(netcuts):
        sig_eff[i] = sig_cuts[i] / sig_cuts[0]
        back_eff[i] = back_cuts[i] / back_cuts[0]

    return sig_eff, back_eff

def create_roc_counter(tree, netcuts, min_value, max_value, reco_true=0, et_function=et_identity):
    '''
    Create and fill ROC counter array holding number of events left after performing cuts on Et-derived value

    :param tree: Tree holding events to be loaded
    :param netcuts: Number of cuts to be performed on events
    :param min_value: Minimum of Et-derived values
    :param max_value: Maximum of Et-derived values
    :param reco_true: Bit value, 0 for using the reconstructed Et and 1 for using the true Pt
    :param et_function: Arbitrary to be applied to the Et of event
    :return: Numpy array holding number events left after cuts
    '''
    entries = tree.entries

    # Scale factor to be used in increment_roc_counter
    scaler = float(max_value - min_value) / float(netcuts)

    # This array will hold the number of signal events with Et-derived value greater than a given index
    #   (assuming scaler = 1)
    roc_counter = np.zeros(netcuts)

    for i in range(entries):
        event = prepare_event(tree, i, 1, 1, 0)

        # Use the reconstructed Et if reco_true set to 0, true Pt if set to 1
        if reco_true == 0:
            event_et = event.reco_et
        else:
            event_et = event.true_tau_pt

        event_et = et_function(event_et)

        increment_roc_counter(event_et, roc_counter, scaler, min_value)

    if roc_counter[0] < entries:
        print('Values exist below defined minimum')
    if roc_counter[-1] > 0:
        print('Values exist above defined maximum')

    return roc_counter


def et_roc_curve(tsig, tback, netcuts, min_value, max_value, reco_true=0, et_function=et_identity):
    '''
    Create a ROC based on Et cuts given a TTree signal file and a TTree background file.

    :param tsig: Signal TTree file
    :param tback: Background TTree file
    :param netcuts: Total number of Et cuts the ROC curve should contain
    :param max_value: Maximum value of the Et range
    :param min_value: Minimum value of the Et range
    :param reco_true: Bit denoting whether to use the true Pt or reconstructed Et for the signal sample. 1 for true Pt,
        0 for reco Et
    :param et_function: Function to apply to the Et value before using in the ROC curve e.g. sigmoid
    :return: TGraph object with the ROC curve plotted
    '''

    # Get lists of event numbers surviving after cuts on Et-derived value, return as numpy arrays
    sig_et_cut = create_roc_counter(tsig, netcuts, min_value, max_value, reco_true, et_function)
    back_et_cut = create_roc_counter(tback, netcuts, min_value, max_value, reco_true, et_function)

    # These will hold the efficiency for a cut at the index Et e.g. reco1_sig_eff[20] holds the percentage of total events
    #   left after a 20 GeV cut
    sig_eff, back_eff = roc_efficiencies_from_cuts(sig_et_cut, back_et_cut)

    gr = TGraph(netcuts, back_eff, sig_eff)

    return gr


def classification_roc_curve(class_file_path, netcuts, min_value, max_value):
    # The number of cuts that will be applied and whose efficiency will be calculated
    scaler = float(max_value - min_value) / float(netcuts)

    # This array will hold the number of signal events with classifier value greater than a given index
    # E.g. classifier_cuts[0.5] will hold the number of signal events with classifier value > 0.5
    class_sig_cuts = np.zeros(netcuts)
    class_back_cuts = np.zeros(netcuts)

    all_lines = prepared_flat_file_lines(class_file_path)

    for line in all_lines:
        pred, truth = line.split(',')
        pred, truth = float(pred), int(truth)

        if truth == 1:
            increment_roc_counter(pred, class_sig_cuts, scaler, min_value)
        elif truth == 0:
            increment_roc_counter(pred, class_back_cuts, scaler, min_value)

    class_sig_eff, class_back_eff = roc_efficiencies_from_cuts(class_sig_cuts, class_back_cuts)

    gr = TGraph(netcuts, class_back_eff, class_sig_eff)

    return gr