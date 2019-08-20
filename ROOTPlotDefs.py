import ROOT
from ROOT import TH1F

def reco_et_tree_histogram(tree, bins, min_value, max_value):
    '''
    Create a histogram of the reconstructed Et values of the events in tree

    :param tree: A tree containing events whose reconstructed Et will be loaded into the returned histogram
    :param bins: Number of bins in the histogram
    :param min_value: Minimum of values loaded into histogram
    :param max_value: Maximum of values loaded into histogram

    :return: A histogram populated with the reconstructed Et values of the events in tree
    '''
    tree_entries = tree.entries

    histo = TH1F("histo", "Reconstructed Et", bins, min_value, max_value)

    for i in range(tree_entries):
        event = prepare_event(tree, i, 0, 1, 0)

        histo.Fill(event.reco_et)

    histo.GetXaxis().SetTitle("Reconstructed Et")
    histo.GetYaxis().SetTitle("Entries")

    return histo


def sig_and_back_reco_et_histograms(sig_tree, back_tree, bins, min_value, max_value):
    '''
    Plot reconstructed Et of signal and background trees into separate histograms. The returned histograms are intended
        to be plotted on the same set of axis.

    :param sig_tree: Signal custom Tree class instance to be loaded into histogram
    :param back_tree: Background custom Tree class instance to be loaded into histogram
    :param bins: Number of bins in each histogram
    :param min_value: Minimum value for the two histograms
    :param max_value: Maximum value of the two histograms
    :return sig_histo: Histogram holding signal reconstructed Et values
    :return back_histo: Histogram holding background reconstructed Et values
    '''
    sig_histo = reco_et_tree_histogram(sig_tree, bins, min_value, max_value)
    back_histo = reco_et_tree_histogram(back_tree, bins, min_value, max_value)

    return sig_histo, back_histo


def true_et_tree_histogram(tree):
    '''
    Create a histogram of the true Et values of the events in tree. This requires that the true Et value for an event
        is stored in the mctau.Et() variable

    :param tree: A custom Tree class instance containing events with true Et information that will be loaded into histo

    :return: A histogram populated with the true Et values of the events in tree
    '''
    tree_entries = tree.entries
    histo = TH1F("histo", "True Et", 100, 0, 100)
    for i in range(tree_entries):
        tree.get_entry(i)
        event = build_event_instance(tree, 1)
        histo.Fill(event.mctau.Et() / 1000)
    histo.GetXaxis().SetTitle("True Tau Et")
    histo.GetYaxis().SetTitle("Entries")
    return histo


def true_pt_tree_histogram(tree):
    '''
    Create a histogram of the true tau Pt of the events in tree

    :param tree: Custom Tree class instance containing events with true tau Pt information, assumed to be in GeV

    :return: A histogram populated with true tau Pt values of the events in tree
    '''
    tree_entries = tree.entries
    histo = TH1F("histo", "True Tau Pt", 100, 0, 100)
    for i in range(tree_entries):
        tree.get_entry(i)
        event = build_event_instance(tree, 1, 0, 0)
        histo.Fill(event.true_tau_pt / 1000)
    histo.GetXaxis().SetTitle("True Tau Pt")
    histo.GetYaxis().SetTitle("Entries")
    return histo


def fcore_tree_histogram(tree):
    '''
    Create a histogram of the FCore values of the events in tree

    :param tree: Custom Tree class instance containing events with FCore information that will be loaded into histo

    :return: A histogram populated with the FCore values of the events in tree
    '''
    tree_entries = tree.entries
    histo = TH1F("histo", "FCore", 100, 0, 1)
    for i in range(tree_entries):
        tree.get_entry(i)
        event = build_event_instance(tree, 0, 0, 1)
        histo.Fill(event.fcore)
    histo.GetXaxis().SetTitle("FCore")
    histo.GetYaxis().SetTitle("Entries")
    return histo


def had_seed_tree_histogram(tree):
    '''
    Create a histogram of the center cell of the hadronic layer

    :param tree: Custom Tree class instance containing events with Et information that will be loaded into histo

    :return: A histogram populated with the center cell of the hadronic layer of the events in tree
    '''
    tree_entries = tree.entries
    histo = TH1F("histo", "Center Had Cell Et", 100, 0, 20)
    for i in range(tree_entries):
        tree.get_entry(i)
        event = build_event_instance(tree, 1, 0, 0)
        histo.Fill(event.had_layer.cell_et[1][1])
    histo.GetXaxis().SetTitle("Center Had Cell Et")
    histo.GetYaxis().SetTitle("Entries")
    return histo

def layer_et_map(layer):
    '''
    Create a 2D graph showing the Et of each cell in layer

    :param layer: Custom Layer class instance whose cell Ets will be plotted

    :return: A TGraph2D object holding the Et of each cell in layer
    '''
    total_values = layer.eta_dim * layer.phi_dim
    eta_values, phi_values = build_2d_index_arrays(total_values, layer.eta_dim, layer.phi_dim)
    flat_ets = layer.cell_et.flatten("F")
    graph = TGraph2D(total_values, eta_values, phi_values, flat_ets)
    return graph

def tree_average_layer_et_map(tree, layer_key):
    '''
    Create a 2D graph of the average Et for each cell in layer over an entire custom Tree class instance

    :param tree: Custom Tree class instance to average all cells for the given layer
    :param layer_key: Key indicating which layer should be averaged and graphed

    :return: A TGraph2D object holding the average Et of the given layer over all events in tree
    '''
    tree_entries = tree.entries
    for i in range(tree_entries):
        tree.get_entry(i)
        event = build_event_instance(tree, 0, 0, 0)
        layer = event.layer_keys[layer_key]
        # For the first event extract dimensional information of the layer and create arrays that will be passed to
        # TGraph2D later
        if i == 0:
            sum_cell_et = np.zeros((layer.eta_dim, layer.phi_dim))
            total_values = layer.eta_dim * layer.phi_dim
            eta_values, phi_values = build_2d_index_arrays(total_values, layer.eta_dim, layer.phi_dim)
        sum_cell_et += layer.cell_et
    flat_ets = sum_cell_et.flatten('F')
    flat_ets = flat_ets / tree_entries
    graph = TGraph2D(total_values, eta_values, phi_values, flat_ets)
    # Define dictionary of titles to use for each layer
    layer_graph_names = {0 : 'L0 Cell Ets', 1 : 'L1 Cell Ets', 2 : 'L2 Cell Ets', 3 : 'L3 Cell Ets', 4 : 'Had Cell Ets'}
    graph.SetTitle(layer_graph_names[layer_key])
    graph.GetXaxis().SetTitle('Eta')
    graph.GetYaxis().SetTitle('Phi')
    graph.GetZaxis().SetTitle('GeV')
    return graph

def tree_average_seed_region_et(tree, layer_key, eta_cells, phi_cells):
    '''
    Create a 2D graph of the average per cell Et for the layer of a tree in a region around the seed cell defined by
        eta_cells and phi_cells

    :param tree: Custom Tree class instance holding the events to be averaged over
    :param layer_key: Key value of the layer whose cells' Et will be average and plotted
    :param eta_cells: Eta dimension of the region around the seed
    :param phi_cells: Phi dimension of the region around the seed

    :return: ROOT TGraph2D object holding the average Et of each cell in the given layer of the tree
    '''

    # Initialize array of zeros
    sum_cell_et = np.zeros((eta_cells, phi_cells))

    # Loop over all events in tree and add cell ets for that event
    for i in range(tree.entries):
        tree.get_entry(i)
        event = build_event_instance(tree, 0, 0, 0)
        layer = event.layer_keys[layer_key]

        new_cell_et = get_et_seed_region(layer, eta_cells, phi_cells, event.seed_region)

        sum_cell_et += new_cell_et

    average_cell_et = sum_cell_et / tree.entries

    eta_values, phi_values = build_2d_index_arrays(average_cell_et.size, len(average_cell_et), len(average_cell_et[0]))
    flat_cell_et = average_cell_et.flatten('F')

    graph = TGraph2D(average_cell_et.size, eta_values, phi_values, flat_cell_et)
    graph.GetXaxis().SetTitle('Eta')
    graph.GetYaxis().SetTitle('Phi')
    graph.GetZaxis().SetTitle('GeV')

    return graph