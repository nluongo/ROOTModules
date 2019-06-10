import ROOT
import ROOTDefs
#from ROOTDefs import find_et_seed

class Layer:
    '''
    A layer of detector data for a single event
    This assumes that the cells arrray holds MeV values and converts them into GeV

    :param cell_et: Array of cell Ets
    :param eta_dim: Size of eta dimension
    :param phi_dim: Size of phi dimension

    :att cell_et: Array holding the Et of each cell
    :att eta_dim: Length of the layer's ea dimension
    :att phi_dim: Length of the layer's phi dimension
    :att total_et: Total Et of layer, sum of Et of all cells in layer
    '''
    def __init__(self, cells, eta_dim, phi_dim, key = -1):
        self.cell_et = cells / 1000
        self.eta_dim = eta_dim
        self.phi_dim = phi_dim
        self.key = key
        # Define total Et present in all cells
        self.total_et = 0
        for i in range(eta_dim):
            for j in range(phi_dim):
                self.total_et += self.cell_et[i][j]


class Event:
    '''
    One event of detector data. This does not create Layer structure as well automatically because then it would need
        to assume the names of layer object in the ntuple.
    Should rarely need to call this directly, instead use build_event_instance()

    :param tree: TTree object with the relevant event loaded through GetEntry()
    :param l0_layer: Custom Layer class instance for L0
    :param l1_layer: Custom Layer class instance for L1
    :param l2_layer: Custom Layer class instance for L2
    :param l3_layer: Custom Layer class instance for L3
    :param had_layer: Custom Layer class instance for Had
    :param mctau: Bit declaring whether to load the mctau attribute of the event
    :param reco+et: Bit declaring whether to calculate reconstructed Et attributes of the event

    :att total_et: Total Et of the event, sum of total_et of each layer
    '''
    def __init__(self, tree, l0_layer, l1_layer, l2_layer, l3_layer, had_layer, mctau_yn=0, reco_et_yn = 1, fcore_yn = 1):
        self.l0_layer = l0_layer
        self.l1_layer = l1_layer
        self.l2_layer = l2_layer
        self.l3_layer = l3_layer
        self.had_layer = had_layer
        self.mctau_yn = mctau_yn
        self.reco_et_yn = reco_et_yn
        self.fcore_yn = fcore_yn

        # Define a dictionary assigning a numerical key to each layer
        self.layer_keys = {0 : self.l0_layer, 1 : self.l1_layer, 2 : self.l2_layer, 3 : self.l3_layer, 4 : self.had_layer}

        # Define total Et in event by adding total Et of each layer
        self.total_et = self.l0_layer.total_et + self.l1_layer.total_et + self.l2_layer.total_et\
                            + self.l3_layer.total_et + self.had_layer.total_et

        # Determine if the event is oriented in such a way that more of the energy is concentrated in a particular
        #   off-phi direction i.e. towards phi=0, set phi_oriented flag accordingly. Commented out and moved under
        #   find_l2_seed because seed information is necessary for calculating\
        self.seed_region = tree.seed_region_def
        self.seed_eta, self.seed_phi = ROOTDefs.find_et_seed(l2_layer, self.seed_region)

        # Determine if the event needs to be phi-oriented, which is determined by the sum of adjacent cells for L1 + L2
        self.phi_zero_sum = l1_layer.cell_et[self.seed_eta][0] + l2_layer.cell_et[self.seed_eta][0]
        self.phi_two_sum = l1_layer.cell_et[self.seed_eta][2] + l2_layer.cell_et[self.seed_eta][2]
        if self.phi_zero_sum >= self.phi_two_sum:
            self.phi_oriented = 1
        else:
            self.phi_oriented = 0

        # Define definition for when to include an adjacent eta cell in reconstructed Et based on seed cell position
        #   passed down from Tree
        self.adjacent_eta_cells = tree.adjacent_eta_cells
        self.adjacent_eta_direction = self.adjacent_eta_cells[self.seed_eta]

        # Define definition for fcore core and isolation region passed down from tree
        self.fcore_def = tree.fcore_def

        # Define layer weights for reconstructed Et passed down from tree
        self.reco_et_layer_weights = tree.reco_et_layer_weights

        # Define shift value for reconstrcuted Et passed down from tree
        self.reco_et_shift = tree.reco_et_shift

        # Define default reconstructed Et in event based on definition passed from Tree
        if reco_et_yn == 1:
            self.reco_et_l0_eta = tree.reco_et_def[0][0]
            self.reco_et_l0_phi = tree.reco_et_def[0][1]
            self.reco_et_l1_eta = tree.reco_et_def[1][0]
            self.reco_et_l1_phi = tree.reco_et_def[1][1]
            self.reco_et_l2_eta = tree.reco_et_def[2][0]
            self.reco_et_l2_phi = tree.reco_et_def[2][1]
            self.reco_et_l3_eta = tree.reco_et_def[3][0]
            self.reco_et_l3_phi = tree.reco_et_def[3][1]
            self.reco_et_had_eta = tree.reco_et_def[4][0]
            self.reco_et_had_phi = tree.reco_et_def[4][1]
            self.reco_et_def = [self.reco_et_l0_eta, self.reco_et_l0_phi, self.reco_et_l1_eta, self.reco_et_l1_phi,
                                self.reco_et_l2_eta, self.reco_et_l2_phi, self.reco_et_l3_eta, self.reco_et_l3_phi,
                                self.reco_et_had_eta, self.reco_et_had_phi]
            self.reco_et = ROOTDefs.calc_reco_et(self)

        # If mctau flag was passed then load truth information
        if mctau_yn == 1:
            if hasattr(tree, 'mctau'):
                self.mctau = tree.mctau
            if hasattr(tree, 'true_tau_pt'):
                self.true_tau_pt = tree.true_tau_pt
            if hasattr(tree, 'true_tau_charged'):
                self.true_tau_charged = tree.true_tau_charged
            if hasattr(tree, 'true_tau_neutral'):
                self.true_tau_neutral = tree.true_tau_neutral

        # If fcore flag was passed then calculate fcore using default definition
        if fcore_yn == 1:
            # Define FCore value of the event based on definition: core = 5x2, isolation = 13x3
            #self.l1l2_combined_layer = Layer(self.l1_layer.cell_et + self.l2_layer.cell_et, self.l1_layer.eta_dim, self.l1_layer.phi_dim)
            self.fcore = calculate_fcore(self.l2_layer, self.fcore_def[0], self.fcore_def[1])


    # Load truth attributes from tree if they were not loaded when the event was created
    def load_truth(self):
        if hasattr(tree, 'mctau'):
            self.mctau = tree.mctau
        if hasattr(tree, 'true_tau_pt'):
            self.true_tau_pt = tree.true_tau_pt
        if hasattr(tree, 'true_tau_charged'):
            self.true_tau_charged = tree.true_tau_charged
        if hasattr(tree, 'true_tau_neutral'):
            self.true_tau_neutral = tree.true_tau_neutral

    # Set a new definition for the reconstructed Et of the event and calculate it
    def set_reco_def(self, new_reco_def):
        self.reco_et_l0_eta = new_reco_def[0][0]
        self.reco_et_l0_phi = new_reco_def[0][1]
        self.reco_et_l1_eta = new_reco_def[1][0]
        self.reco_et_l1_phi = new_reco_def[1][1]
        self.reco_et_l2_eta = new_reco_def[2][0]
        self.reco_et_l2_phi = new_reco_def[2][1]
        self.reco_et_l3_eta = new_reco_def[3][0]
        self.reco_et_l3_phi = new_reco_def[3][1]
        self.reco_et_had_eta = new_reco_def[4][0]
        self.reco_et_had_phi = new_reco_def[4][1]
        self.reco_et_def = [self.reco_et_l0_eta, self.reco_et_l0_phi, self.reco_et_l1_eta, self.reco_et_l1_phi,
                            self.reco_et_l2_eta, self.reco_et_l2_phi, self.reco_et_l3_eta, self.reco_et_l3_phi,
                            self.reco_et_had_eta, self.reco_et_had_phi]
        self.reco_et = calc_reco_et(self)

    def set_fcore_def(self, new_fcore_def):
        self.fcore_def = new_fcore_def
        self.fcore = calculate_fcore(self.l2_layer, self.fcore_def[0], self.fcore_def[1])

    # Modify the values of layer weights used to calculate reconstructed Et
    def set_reco_et_layer_weights(self, new_layer_weights):
        self.reco_et_layer_weights = new_layer_weights
        self.reco_et = calc_reco_et(self)

    # Modify the value of shift Et used to calculate reconstucted Et
    def set_reco_et_shift(self, new_shift_et):
        self.reco_et_shift = new_shift_et
        self.reco_et = calc_reco_et(self)

    # If the off-phi is not concentrated in the 0 phi direction, then flip all layer so that it is and then recalculate
    #   all values that are orientation sensitive
    def phi_orient(self):
        if self.phi_oriented == 1:
            pass
        else:
            # Flip each layer
            ROOTDefs.phi_flip_layer(self.l0_layer)
            ROOTDefs.phi_flip_layer(self.l1_layer)
            ROOTDefs.phi_flip_layer(self.l2_layer)
            ROOTDefs.phi_flip_layer(self.l3_layer)
            ROOTDefs.phi_flip_layer(self.had_layer)

            # Recalculate reconstructed Et
            if self.reco_et_yn == 1:
                self.reco_et = ROOTDefs.calc_reco_et(self)

            # Recalculate FCore
            if self.fcore_yn == 1:
                self.l1l2_combined_layer = Layer(self.l1_layer.cell_et + self.l2_layer.cell_et, self.l1_layer.eta_dim, self.l1_layer.phi_dim)
                self.fcore = calculate_fcore(self.l1l2_combined_layer, self.fcore_def[0], self.fcore_def[1])

            # Set flag indicating the event is now phi oriented
            self.phi_oriented = 1

    # Search the L2 layer in the given eta/phi range for the cell with greatest Et and set coordinates to variables
    def set_adjacent_eta(self, adjacent_eta_cells):
        self.adjacent_eta_cells = adjacent_eta_cells
        self.adjacent_eta_direction = self.adjacent_eta_cells[self.seed_eta]


class Tree:
    '''
    A tree of events

    :param tree: A TTree object
    '''
    def __init__(self, ttree):
        self.root_ttree = ttree
        self.entries = self.root_ttree.GetEntries()
        self.layer_dim_keys = {0 : [3, 3], 1 : [13, 3], 2 : [13, 3], 3 : [3, 3], 4 : [3, 3]}
        self.reco_et_def = [[1, 2], [5, 2], [5, 2], [3, 2], [3, 2]]
        self.seed_region_def = [[4, 8], [1, 1]]
        self.adjacent_eta_cells = { 4: -1, 5: 0, 6: 0, 7: 0, 8: 1 }
        self.fcore_def = [[3, 2], [13, 3]]
        self.reco_et_layer_weights = [1, 1, 1, 1, 1]
        self.reco_et_shift = 0

    # Modify the reconstructed Et definition for the Tree
    def set_reco_et_def(self, new_reco_et_def):
        self.reco_et_def = new_reco_et_def

    # Modify the layer dimensions if they are something other than the above default values
    def set_layer_dim(self, layer_key, eta_dim, phi_dim):
        self.layer_dim_keys[layer_key] = [eta_dim, phi_dim]

    # Modify the region whose max Et cell will be chosen as the seed
    def set_seed_region(self, min_eta, max_eta, min_phi, max_phi):
        self.seed_region_def = [[min_eta, max_eta], [min_phi, max_phi]]

    # Modify the eta values of the seed cell that will require including an adjacent eta cell in 3x3 layers
    def set_adjacent_eta_cells(self, new_adjacent_eta_cells):
        self.adjacent_eta_cells = new_adjacent_eta_cells

    # Modify the FCore core and isolation region definitions
    def set_fcore_def(self, new_fcore_def):
        self.fcore_def = new_fcore_def

    # Modify the values of layer weights used to calculate reconstructed Et
    def set_reco_et_layer_weights(self, new_layer_weights):
        self.reco_et_layer_weights = new_layer_weights

    # Modify the value of shift Et used to calculate reconstructed Et
    def set_reco_et_shift(self, new_shift_et):
        self.reco_et_shift = new_shift_et

    # Call the GetEntry() method of the input tree and assign all values to class variables of the same names
    def get_entry(self, i):
        self.root_ttree.GetEntry(i)
        self.L0CellEt = self.root_ttree.L0CellEt
        self.L1CellEt = self.root_ttree.L1CellEt
        self.L2CellEt = self.root_ttree.L2CellEt
        self.L3CellEt = self.root_ttree.L3CellEt
        self.HadCellEt = self.root_ttree.HadCellEt
        if hasattr(self.root_ttree, 'mc_visibleTau'):
            self.mctau = self.root_ttree.mc_visibleTau
            self.true_tau_pt = self.root_ttree.mc_visibleTau.Pt()
        if hasattr(self.root_ttree, 'true_tau_pt'):
            self.true_tau_pt = self.root_ttree.true_tau_pt
        if hasattr(self.root_ttree, 'true_tau_charged'):
            self.true_tau_charged = self.root_ttree.true_tau_charged
        if hasattr(self.root_ttree, 'true_tau_neutral'):
            self.true_tau_neutral = self.root_ttree.true_tau_neutral
