import numpy as np
import ROOT
from ROOT import TH1F, TGraph, TGraph2D, TFile
import ROOTClassDefs
import os
import statistics as stats
import math
from math import exp

def set_po_tree_parameters(tree):
    '''
    Sets standard parameters for custom Tree instance for PO files

    :param tree: Custom Tree instance
    :return: Custom Tree instance with parameters set
    '''

    # Set dimensions of layers
    tree.set_layer_dim(1, 12, 3)
    tree.set_layer_dim(2, 12, 3)
    # Set region from which to select the seed cell
    tree.set_seed_region(4, 7, 1, 1)
    # Set the method for choosing adjacent cells in certain layers
    new_adj_dict = {4: -1, 5: 0, 6: 0, 7: 1}
    tree.set_adjacent_eta_cells(new_adj_dict)

def resize_root_layer_to_array(layer_et, eta_dim, phi_dim):
    '''
    Takes in the one-dimensional Et layer object as returned by ROOT GetEntry() and reformats it into a two-dimensional
        numpy array with dimensions given by eta_range and phi_range

    :param layer_et: One dimensional Et layer object as returned by ROOT GetEntry()
    :param eta_dim: Size of desired output array in the eta dimension
    :param phi_dim: Size of desired output array in the phi dimension

    :return: Numpy array with the dimensions given by eta_dim and phi_dim
    '''

    #Convert to array, conversion to list first is necessary for some reason
    layer_oned_array = np.asarray(list(layer_et))
    #Resize and return
    return np.resize(layer_oned_array, (eta_dim, phi_dim))

def load_root_layer_to_class(tree, layer_et, key):
    '''
    Load a layer as returned by ROOT GetEntry() into the custom Layer class. Actions done by this function are not
        included in the Layer class because it may be necessary to load a custom layer directly from an array.

    :param layer_et: ROOT layer object as returned by GetEntry()
    :param key: Numeric key to assign to the given layer

    :return: A custom Layer class instance
    '''
    eta_dim = tree.layer_dim_keys[key][0]
    phi_dim = tree.layer_dim_keys[key][1]

    while eta_dim * phi_dim != len(layer_et):
        print('Invalid eta and phi dimensions for layer of key', key)
        print('Dimensions must produce', len(layer_et), 'total cells')
        eta_dim = int(input('Eta dimension: '))
        phi_dim = int(input('Phi dimension: '))

    # Once valid dimensions are given, write back to Tree instance for use in later events
    tree.set_layer_dim(key, eta_dim, phi_dim)

    #Call function to convert ROOT object to two-dimensional numpy array
    layer_array = resize_root_layer_to_array(layer_et, eta_dim, phi_dim)
    #Load array into Layer class instance
    layer_class_instance = ROOTClassDefs.Layer(layer_array, eta_dim, phi_dim, key)
    return layer_class_instance

def build_event_instance(tree, mctau=0, reco_et=1, fcore=0):
    '''
    Build custom Event class instance. Intended to be used after the ROOT GetEntry function is called. First creates
        custom Layer instances which are then loaded into the Event.
    Assumes the following names for layers: L0CellEt, L1CellEt, L2CellEt, L3CellEt, HadCellEt

    :param tree: ROOT TTree object with event loaded using the GetEntry() function
    :param mctau: Bit declaring whether to load the mctau attribute of the event
    :param reco_et: Bit declaring whether to calculate the reconstructed Et attributes of the event
    :param fcore: Bit declaring whether to calculate the FCore attributes of the event

    :return: A custom Event class instance
    '''

    l0_layer = load_root_layer_to_class(tree, tree.L0CellEt, 0)
    l1_layer = load_root_layer_to_class(tree, tree.L1CellEt, 1)
    l2_layer = load_root_layer_to_class(tree, tree.L2CellEt, 2)
    l3_layer = load_root_layer_to_class(tree, tree.L3CellEt, 3)
    had_layer = load_root_layer_to_class(tree, tree.HadCellEt, 4)
    event  = ROOTClassDefs.Event(tree, l0_layer, l1_layer, l2_layer, l3_layer, had_layer, mctau, reco_et, fcore)

    return event

def calc_reco_et(event):
    reco_et = event_reco_et(event, event.reco_et_l0_eta, event.reco_et_l0_phi,
                            event.reco_et_l1_eta, event.reco_et_l1_phi,
                            event.reco_et_l2_eta, event.reco_et_l2_phi,
                            event.reco_et_l3_eta, event.reco_et_l3_phi,
                            event.reco_et_had_eta, event.reco_et_had_phi,
                            event.reco_et_layer_weights, event.reco_et_shift)
    return reco_et

def prepare_event(tree, i, mctau=0, reco_et=1, fcore=0):
    '''
    Create and prepare custom Event class instance. Loads event information into custom Tree, builds Event, then phi
        orients

    :param tree: Custom Tree class instance
    :param i: Event number to load
    :param mctau: Bit declaring whether to load the mctau attribute of the event
    :param reco_et: Bit declaring whether to calculate the reconstructed Et attributes of the event
    :param fcore: Bit declaring whether to calculate the FCore attributes of the event
    :return: Custom Event class instance
    '''

    tree.get_entry(i)
    event = build_event_instance(tree, mctau, reco_et, fcore)
    event.phi_orient()

    return event

def event_reco_et(event, l0_reco_eta, l0_reco_phi, l1_reco_eta, l1_reco_phi, l2_reco_eta, l2_reco_phi, l3_reco_eta,
                  l3_reco_phi, had_reco_eta, had_reco_phi, layer_weights=[1,1,1,1,1], shift_et = 0):
    '''
    Calculate the Et of an event according to the reconstructed Et definition defined by the given eta and phi
        values for each layer.
    If layer_weights passed then the Et of each layer is multiplied by the corresponding weight before summing. If
        layer_weights not provided then all weights default to 1
    If shift_et passed then that value is added after layers are weighted and summed

    :param event: Custom Event class instance
    :param l0_reco_eta: Number of eta cells in the L0 layer to be included
    :param l0_reco_phi: Number of phi cells in the L0 layer to be included
    :param l1_reco_eta: Number of eta cells in the L1 layer to be included
    :param l1_reco_phi: Number of phi cells in the L1 layer to be included
    :param l2_reco_eta: Number of eta cells in the L2 layer to be included
    :param l2_reco_phi: Number of phi cells in the L2 layer to be included
    :param l3_reco_eta: Number of eta cells in the L3 layer to be included
    :param l3_reco_phi: Number of phi cells in the L3 layer to be included
    :param had_reco_eta: Number of eta cells in the Had layer to be included
    :param had_reco_phi: Number of phi cells in the Had layer to be included
    :param layer_weights: List of 5 integers holding the weights for each layer reco Et to be multiplied by
    :param shift_et: Extra amount to add to Et independent of any layer

    :return: The Et of the event according to the given reconstructed Et definition
    '''

    #Check here if reconstucted Et definition is valied
    if (is_layer_reco_def_valid(event.l0_layer.eta_dim, event.l0_layer.phi_dim, event.l0_layer.key, l0_reco_eta,
                                l0_reco_phi) == 0 or
        is_layer_reco_def_valid(event.l1_layer.eta_dim, event.l1_layer.phi_dim, event.l1_layer.key, l1_reco_eta,
                                l1_reco_phi) == 0 or
        is_layer_reco_def_valid(event.l2_layer.eta_dim, event.l1_layer.phi_dim, event.l1_layer.key, l2_reco_eta,
                                l2_reco_phi) == 0 or
        is_layer_reco_def_valid(event.l3_layer.eta_dim, event.l2_layer.phi_dim, event.l2_layer.key, l3_reco_eta,
                                l3_reco_phi) == 0 or
        is_layer_reco_def_valid(event.had_layer.eta_dim, event.had_layer.phi_dim, event.had_layer.key, had_reco_eta,
                                had_reco_phi) == 0):
        print("Invalid reconstructed Et definition")
        quit

    l0_reco_et = layer_reco_et(event.l0_layer, l0_reco_eta, l0_reco_phi, -1, -1, event.adjacent_eta_direction)
    event.l0_layer.reco_et = l0_reco_et
    event.l0_layer.reco_et_weighted = l0_reco_et * layer_weights[0]

    l1_reco_et = layer_reco_et(event.l1_layer, l1_reco_eta, l1_reco_phi, event.seed_eta, event.seed_phi)
    event.l1_layer.reco_et = l1_reco_et
    event.l1_layer.reco_et_weighted = l1_reco_et * layer_weights[1]

    l2_reco_et = layer_reco_et(event.l2_layer, l2_reco_eta, l2_reco_phi, event.seed_eta, event.seed_phi)
    event.l2_layer.reco_et = l2_reco_et
    event.l2_layer.reco_et_weighted = l2_reco_et * layer_weights[2]

    l3_reco_et = layer_reco_et(event.l3_layer, l3_reco_eta, l3_reco_phi, -1, -1, event.adjacent_eta_direction)
    event.l3_layer.reco_et = l3_reco_et
    event.l3_layer.reco_et_weighted = l3_reco_et * layer_weights[3]

    had_reco_et = layer_reco_et(event.had_layer, had_reco_eta, had_reco_phi, -1, -1, event.adjacent_eta_direction)
    event.had_layer.reco_et = had_reco_et
    event.had_layer.reco_et_weighted = had_reco_et * layer_weights[4]

    total_reco_et = event.l0_layer.reco_et_weighted + \
                    event.l1_layer.reco_et_weighted + \
                    event.l2_layer.reco_et_weighted + \
                    event.l3_layer.reco_et_weighted + \
                    event.had_layer.reco_et_weighted + \
                    shift_et

    return total_reco_et

def layer_reco_et(layer, eta_cells, phi_cells, seed_eta = -1, seed_phi = -1, adjacent_eta_def = 0):
    '''
    Calculate the Et of a layer according to the given reconstructed Et definition for the layer by summing the Et
        of all referenced cells

    :param layer: Custom Layer class instance whose Et is being calculated
    :param eta_cells: Number of cells in the eta direction in the reconstructed Et definition
    :param phi_cells: Number of cells in the phi direction in the reconstructed Et definition

    :return: The Et of the layer according to the given reconstructed Et definition
    '''
    # Invalid region definition
    if eta_cells <= 0 or phi_cells <= 0:
        return 0

    if eta_cells == layer.eta_dim and phi_cells == layer.phi_dim:
        return layer.total_et

    # Get min and max values for eta and phi, defining the region to sum over
    eta_min, eta_max = get_eta_range(layer.eta_dim, eta_cells, seed_eta)
    phi_min, phi_max = get_phi_range(layer.phi_dim, phi_cells)

    # If the seed is close to the edge of a coarse cell and we are only using a single eta cell in that layer,
    #   then include an adjacent cell in the eta direction for coarse layers
    if eta_cells == 1:
        if adjacent_eta_def == -1:
            eta_min = eta_min - 1
        elif adjacent_eta_def == 1:
            eta_max = eta_max + 1

    total_et = 0
    for i in range(eta_min, eta_max + 1):
        for j in range(phi_min, phi_max + 1):
            total_et += layer.cell_et[i][j]
    return total_et

def is_layer_reco_def_valid(layer_eta, layer_phi, layer_key, def_eta, def_phi):
    '''
    Determines if the given eta and phi dimensions of a reconstructed Et definition are valid for the given layer

    :param layer_eta: The number of cells in eta that the layer contains
    :param layer_phi: The number of cells in phi that the layer contains
    :param layer_key: The key value of the layer in the event
    :param def_eta: The number of cells in eta to be included in the reconstructed Et definition
    :param def_phi: The number of cells in phi to be included in the reconstructed Et definition

    :return: 1 if the definition is valid, 0 if not
    '''
    if def_eta > layer_eta:
        print("Reco eta definition larger than layer of key ", layer_key)
        return 0
    elif def_phi > layer_phi:
        print("Reco phi definition larger than layer of key", layer_key)
        return 0
    elif def_eta%2 != 1 and def_eta > 0:
        print("Reco eta definition must be an odd number of cells for layer", layer_key)
        return 0
    else:
        return 1

def get_eta_range(layer_eta, range_length, seed_eta = -1):
    '''
    Returns a range of eta whose central value is seed_eta if provided else the center of the layer and containing
    range_length total cells

    Example: get_eta_range(13, 5) = 4, 8

    :param layer_eta: The number of cells in the given dimension of the layer
    :param range_length: The number of cells to be included in the range
    :param seed_eta: The eta index of the seed cell if provided, else the center of the layer is used

    :return eta_min: The minimum eta value
    :return eta_max: The maximum eta value
    '''
    # 0 is an invalid range
    if range_length == 0:
        return 0, -1
    # Need an odd number of cells for center cell plus same number to either side
    elif range_length%2 == 0:
        print("Reco eta definition must be an odd number of cells")
        print("Given value: ",range_length)
        exit()
    # Range can't require more cells than are in the layer
    elif range_length > layer_eta:
        print("Reco eta definition larger than layer")
        print("Given value: ",range_length)
        exit()
    # Here if given definition is valid
    else:
        etaoffsetfromcenter = (range_length - 1) / 2
        # If no given seed eta, then use the center of the layer
        if seed_eta == -1:
            etacenter = (layer_eta - 1) / 2
        else:
            etacenter = seed_eta

    eta_min = etacenter - etaoffsetfromcenter
    eta_max = etacenter + etaoffsetfromcenter
    return int(eta_min), int(eta_max)

def get_phi_range(layer_phi, range_length):
    '''
    Returns a range of phi whose central value is the center value of layer_phi and containing range_length
        total cells, unless range_length = 2, in which case the range includes only the center and off-center to one
        side

    Examples:   get_cell_range(3, 3) = 0, 2
                get_cell_range(3, 2) = 0, 1
                get_cell_range(3, 1) = 1, 1

    :param layer_phi: The number of cells in the given dimension of the layer
    :param range_length: The number of cells to be included in the range

    :return eta_min: The minimum phi value
    :return eta_max: The maximum phi value
    '''
    if range_length == 0:
        return 0, -1
    elif range_length == 2:
        return 0, 1
    elif range_length > layer_phi:
        print("Reco phi definition larger than layer")
        print("Given value: ",range_length)
        exit()
    else:
        phioffsetfromcenter = (range_length - 1) / 2
        phicenter = (layer_phi - 1) / 2
    phi_min = phicenter - phioffsetfromcenter
    phi_max = phicenter + phioffsetfromcenter
    return int(phi_min), int(phi_max)

def phi_flip_layer(layer):
    '''
    Flip a layer's Et cells phi-wise

    :param layer: Custom Layer class instance to flip
    '''
    eta_len = layer.eta_dim
    phi_len = layer.phi_dim
    cell_et_holder = np.zeros((eta_len, phi_len))
    for i in range(eta_len):
        for j in range(phi_len):
            cell_et_holder[i][j] = layer.cell_et[i][phi_len - 1 - j]
    layer.cell_et = cell_et_holder

def calculate_fcore(layer, fcore_core_def, fcore_isolation_def):
    '''
    Calculate FCore value for a given event and FCore definition

    :param layer: The custom Layer class instance for which FCore is being calculated
    :param fcore_core_def: List with two elements, [0] = eta core definition, [1] = phi core definition
    :param fcore_isolation_def: List with two elements, [0] = eta isolation definition, [1] = phi isolation definition

    :return: The float value of FCore
    '''
    #Create single custom Layer class instance whose Et for each cell is the sum of the Et of corresponding cells in L1
    #   and L2 layers
    core_et = layer_reco_et(layer, fcore_core_def[0], fcore_core_def[1])
    isolation_et = layer_reco_et(layer, fcore_isolation_def[0], fcore_isolation_def[1])
    return (core_et / isolation_et)

def find_histo_percent_bin(histo, percent):
    '''
    Calculate the bin of a histogram above which the specified percentage of the histogram entries lie. Start from bin
        0 and add events in each subsequent bin until the running sum exceeds the percentage of the total events
    :param histo: Histogram object whose bin you want calculated
    :param percent: The percent of the histogram that should lie above the calculated bin, percent not decimal form

    :return: An integer bin number above which the given percentage of the histogram lies
    '''
    num_of_bins = histo.GetNbinsX()
    num_of_events = histo.GetEntries()
    if num_of_events == 0:
        num_of_events = 1
    cumulative_events = 0
    for i in range(num_of_bins):
        cumulative_events += histo.GetBinContent(i)
        if cumulative_events / num_of_events > (1 - percent/100):
            return i
    print("No 95% bin found")

def build_2d_index_arrays(total_values, eta_dim, phi_dim):
    '''
    Create and fill arrays that can be passed to TGraph2D for the given dimensions of eta and phi

    :param total_values: Total number of points to be included in the TGraph2D
    :param eta_dim: Size of the layer's eta dimension
    :param phi_dim: Size of the layer's phi dimension

    :return: Two 1D arrays that hold the coordinates of every point to be included in the TGraph2D object
    '''
    eta_values = np.zeros(total_values)
    phi_values = np.zeros(total_values)
    for i in range(total_values):
        eta_values[i] = i % eta_dim
        phi_values[i] = int(i / eta_dim)
    return eta_values, phi_values

def find_et_seed(layer, seed_region):
    '''
    Scan over the Et cells for a given layer and find the coordinates of the cell with highest Et i.e the "seed".
        The max values given should be included in the range, as 1 is added to the range argument

    :param layer: Custom Layer class instance
    :param seed_region: Array containing [0][0] = minimum eta value of seed range
                                         [0][1] = maximum eta value of seed range
                                         [1][0] = minimum phi value of seed range
                                         [1][1] = maximum phi value of seed range

    :return seed_eta: Eta index of the seed cell
    :return seed_phi: Phi index of the seed cell
    '''
    # Unpack seed_region array to individual variables
    min_eta = seed_region[0][0]
    max_eta = seed_region[0][1]
    min_phi = seed_region[1][0]
    max_phi = seed_region[1][1]

    # Initialize max_et to large negative value so any reasonable Et will be recognized as larger
    max_et = -100000
    seed_eta = -1
    seed_phi = -1
    for i in range(min_eta, max_eta + 1):
        for j in range(min_phi, max_phi + 1):
            if layer.cell_et[i][j] > max_et:
                max_et = layer.cell_et[i][j]
                seed_eta = i
                seed_phi = j
    return seed_eta, seed_phi


def get_et_seed_region(layer, eta_cells, phi_cells, seed_region):
    '''
    Return the portion of the layer's Et array with the dimensions of eta_cells by phi_cells centered on the cell with
        the highest Et

    :param layer: Custom Layer class instance
    :param eta_cells: Eta dimension of returned region
    :param phi_cells: Phi dimension of returned region

    :return: Array with given dimensions centered on the seed cell
    '''
    # Find seed coordinates for that layer inside given seed region
    seed_eta, seed_phi = find_et_seed(layer, seed_region)

    # Find min and max eta/phi values that give a region of the given dimensions around the seed
    min_eta, max_eta = get_eta_range(layer.eta_dim, eta_cells, seed_eta)
    min_phi, max_phi = get_phi_range(layer.phi_dim, phi_cells)

    # Check if the ranges are reasonable
    if min_eta < 0 or max_eta > layer.eta_dim - 1 or min_phi < 0 or max_phi > layer.eta_dim - 1 or seed_phi != 1:
        return np.zeros((9, 3))

    return layer.cell_et[min_eta:max_eta + 1, min_phi:max_phi + 1]

def apply_tree_cut(old_tree, cut_string, temp_file):
    '''
    Apply a generalized cut to a given custom Tree class instance, returning a clone of the Tree with the cut applied.
        Trees are required to be written file as they may grow too big to fit in memory. temp_file is often temp_file.root
        and must be opened with TFile before this function is called and should be closed and deleted afterward.
        Note that any configuration done to the original tree must be redone after using this function

    :param old_tree: Custom Tree class instance to be cut
    :param cut_string: String containing the logic for which events should be added to the new Tree
    :param temp_file: Temporary file path that the new Tree is written to. Must be

    :return: Custom Tree class instance that has the cut applied
    '''
    # Copy ROOT TTree object from the Tree being cut
    new_tree = old_tree.root_ttree.CloneTree(0)
    new_tree.SetDirectory(temp_file)

    # Build query around cut logic to fill desired events into new TTree
    exec_string = "if " + cut_string + ":\n new_tree.Fill()"

    tree_entries = old_tree.entries
    # Loop through events of old_tree and execute built query string to load desire events into new_tree
    for i in range(tree_entries):
        event = prepare_event(old_tree, i, 1, 1, 0)

        exec(exec_string)

    # Write new_tree to temp_file
    temp_file.Write()

    # Load new TTree into custom Tree and return
    return ROOTClassDefs.Tree(new_tree)

def tau_formatted_root_directory():
    '''
    :return: Predetermined directory holding formatted ROOT files for the TauTrigger project
    '''
    directory_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', 'NTuples')
    return directory_path

def open_formatted_root_file(file_name):
    '''
    Find and return ROOT TFile object with name file_name in predetermined folder

    :param file_name: Name of file to open and return in predetermined folder
    :return: ROOT TFile object of file file_name
    '''
    tau_directory_path = tau_formatted_root_directory()
    file_path = os.path.join(tau_directory_path, file_name)
    file = ROOT.TFile(file_path)
    return file

def recreate_formatted_root_file(file_name):
    '''
    Recreate ROOT TFile object with name file_name in predetermined folder

    :param file_name: Name of file to open and return in predetermined folder
    :return: ROOT TFile object of file file_name
    '''
    tau_directory_path = tau_formatted_root_directory()
    file_path = os.path.join(tau_directory_path, file_name)
    file = ROOT.TFile(file_path, 'recreate')
    return file

def get_formatted_root_tree(file_name, tree_name = 'mytree'):
    '''
    Return ROOT TTree object in file file_name with name tree_name. Calls get_root_file which looks in predetermined
        folder. This must return the TFile object as well or else the file is closed on function return and the tree
        becomes unusable.

    :param file_name: Name of file that tree is located
    :param tree_name: Name of tree to find and return, defaults to 'mytree'
    :return: ROOT TTree object and ROOT TFile object in given file with given name
    '''
    file = open_formatted_root_file(file_name)
    tree = ROOTClassDefs.Tree(file.Get(tree_name))
    return tree, file

def get_po_signal_et_background_files():
    '''
    Retrieve custom Tree instances and TFiles with the standard setup being used for pretty much all tau trigger research. The signal
        sample is the PO and the background is the ET one. The signal file is also set up the default way for PO files
        and a >20 GeV true Pt cut is applied.

    :return: Signal custom Tree instance, signal TFile, background customer Tree instance, and background TFile
    '''
    tsig, fsig = get_formatted_root_tree('ztt_Output_formatted.root')
    set_po_tree_parameters(tsig)

    tback, fback = get_formatted_root_tree('output_MB80_formatted.root')

    return tsig, fsig, tback, fback

def get_temp_root_file(file_name):
    '''
    Return a TFile with the given name in the predefined folder. This is intended to be used with apply_tree_cut as that
        requires a temporary file to perform TTree actions.

    :param file_name: The name of the temp ROOT file
    :return: The TFile object of the temp ROOT file
    '''
    temp_file_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', file_name)
    temp_file = TFile(temp_file_path, 'recreate')
    return temp_file

def get_reco_stats(tree1, tree2):
    '''
    Return maximum, minimum, mean, and standard deviation values of the reconstructed Et for two custom Tree instances.
        The values are calculated for the combined set of events for both Trees.

    :param tree1: First Tree to be added to combined list
    :param tree2: Second Tree to be added to combined list
    :return max_et: Maximum reconstructed Et across both Trees
    :return min_et: Minimum recontructed Et across both Trees
    :return avg_et: Average reconstructed Et of all events across both Trees
    :return stdev_et: Standard deviation of reconstructed Et of all events across both Trees
    '''
    et_list = []

    for i in range(tree1.entries):
        event = prepare_event(tree1, i, 0, 1, 0)

        et_list.append(event.reco_et)

    for i in range(tree2.entries):
        event = prepare_event(tree2, i, 0, 1, 0)

        et_list.append(event.reco_et)

    max_et = max(et_list)
    min_et = min(et_list)
    avg_et = stats.mean(et_list)
    stdev_et = stats.stdev(et_list)

    return max_et, min_et, avg_et, stdev_et