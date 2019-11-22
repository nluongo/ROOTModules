from ROOTDefs import load_root_layer_to_class
from ROOTClassDefs import Event

# Format cell Et arrays for 3x3 layers in such a way as to maintain [eta][phi] setup with my existing code
def po_3x3_cells_to_array(formatted_array, po_vector):
    formatted_array[0] = po_vector[0]
    formatted_array[3] = po_vector[1]
    formatted_array[6] = po_vector[2]
    formatted_array[1] = po_vector[3]
    formatted_array[4] = po_vector[4]
    formatted_array[7] = po_vector[5]
    formatted_array[2] = po_vector[6]
    formatted_array[5] = po_vector[7]
    formatted_array[8] = po_vector[8]

# Format cell Et arrays for 12x3 layers in such a way as to maintain [eta][phi] setup with my existing code
def po_12x3_cells_to_array(formatted_array, po_vector):
    formatted_array[0] = po_vector[0]
    formatted_array[3] = po_vector[1]
    formatted_array[6] = po_vector[2]
    formatted_array[9] = po_vector[3]
    formatted_array[12] = po_vector[4]
    formatted_array[15] = po_vector[5]
    formatted_array[18] = po_vector[6]
    formatted_array[21] = po_vector[7]
    formatted_array[24] = po_vector[8]
    formatted_array[27] = po_vector[9]
    formatted_array[30] = po_vector[10]
    formatted_array[33] = po_vector[11]
    formatted_array[1] = po_vector[12]
    formatted_array[4] = po_vector[13]
    formatted_array[7] = po_vector[14]
    formatted_array[10] = po_vector[15]
    formatted_array[13] = po_vector[16]
    formatted_array[16] = po_vector[17]
    formatted_array[19] = po_vector[18]
    formatted_array[22] = po_vector[19]
    formatted_array[25] = po_vector[20]
    formatted_array[28] = po_vector[21]
    formatted_array[31] = po_vector[22]
    formatted_array[34] = po_vector[23]
    formatted_array[2] = po_vector[24]
    formatted_array[5] = po_vector[25]
    formatted_array[8] = po_vector[26]
    formatted_array[11] = po_vector[27]
    formatted_array[14] = po_vector[28]
    formatted_array[17] = po_vector[29]
    formatted_array[20] = po_vector[30]
    formatted_array[23] = po_vector[31]
    formatted_array[26] = po_vector[32]
    formatted_array[29] = po_vector[33]
    formatted_array[32] = po_vector[34]
    formatted_array[35] = po_vector[35]

# Fill ET maps
def createCellLists(myTOB):
  '''
  Load cell energies from TOB into array

  :param myTOB: TOB whose energies will be loaded
  
  :return: 12x3 array
  '''
  layerOffset = [1, 4, 4, 1, 1]
  layerCells = [i*3 for i in layerOffset]

  allCells = []

  # Iterate over layer
  for l in range(5):
    myLayer = []
    # Iterate over phi
    for i in range(3):
      # Iterate over eta
      for j in range(layerCells[l]):
        myLayer += [myTOB.getEnergy(l+2, j+layerOffset[l], i+1)]
    allCells += [ myLayer ]
  return allCells

def isLocalMax(cells, candidateEta, candidatePhi):
  '''
  Determines if the given cells contain a local energy max at the given eta and phi coordinates. Follows the logic of the eFEX firmware, so cells to the left/top must be strictly greater while those to the right/bottom need only be less than or equal. This is to choose a seed if adjacent cells have the same energy.

  :param cells: 12x3 array of values holding cell Ets
  :param candidateEta: Eta index of the cell being checked as a local max
  :param candidatePhi: Phi index of the cell being checked as a local max

  :return: Truth value denoting whether the candidate cell passes the logic as a local max
  '''
  # Loop over all cells adjacent to the potential seed
  for i in range(3):
    for j in range(3):
      # Get particular adjacent cell 
      adjacentEta = candidateEta - 1 + i
      adjacentPhi = candidatePhi - 1 + j

      # Potential seed cell cannot be on the edge of the region
      if adjacentEta not in range(12) or adjacentPhi not in range(3):
        raise ValueError('Adjacent cells outside of window')

      # Don't need to compare cell with itself 
      if i == 1 and j == 1:
        continue
      # From eFEX definition, cells to the right and below can have equal energy to handle edge case
      elif ((i == 2 and j == 1) or (i == 1 and j == 0)):
        if cells[candidateEta][candidatePhi] < cells[adjacentEta][adjacentPhi]:
          return False
      # All other cells must have strictly greater energy
      else:
        if cells[candidateEta][candidatePhi] <= cells[adjacentEta][adjacentPhi]:
          return False
  return True

# Find seed cell of TOB if one exists
def getSeedCell(cellEt):
  '''
  Given an array of cell energies, find the seed cell according the eFEX firmware definition

  :param cellEt: 2-D array containing cell energies
  
  :return: Eta and phi of seed found, if no seed found variables return None
  '''
  maxSeedEta = None
  maxSeedEt = None
  phi = 1
  for i in range(4):
    eta = 4 + i
    if isLocalMax(cellEt, eta, phi) and cellEt[eta][phi] > maxSeedEt:
      maxSeedEta = eta
      maxSeedEt = cellEt[eta][phi]
  return maxSeedEta, phi 

def getTowerEt(event, eta, phi):
    '''
    Get the Et of a tower in a TOB by summing up all cells in that tower.

    :param tob: TOB object
    :param eta: Eta index of the tower in the context of a 3x3 layer i.e. 0-2
    :param phi: Phi index of the tower in the context of a 3x3 layer i.e. 0-2
    :return: Et value for the tower
    '''
    towerEt = event.l0_layer.cell_et[eta][phi] + event.l3_layer.cell_et[eta][phi] + event.had_layer.cell_et[eta][phi]
    for i in range(4):
        supercell_eta = i + eta * 4
        towerEt += event.l1_layer.cell_et[supercell_eta][phi]
        towerEt += event.l2_layer.cell_et[supercell_eta][phi]

    return towerEt

# Determine if central tower of TOB is local max
def isCentralTowerSeed(event):
    '''
    Determine if the central tower of the given event has a max in the central tower 
    '''
    # Get Et of central tower
    centralTowerEt = getTowerEt(event, 1, 1)
    # Check towers in bottom left corner which must be greater than or equal per central tower seeding
    if centralTowerEt < getTowerEt(event, 0, 1) or centralTowerEt < getTowerEt(event, 0, 0) or centralTowerEt < getTowerEt(event, 1, 0):
        return 0
    # Check all other towers which must be strictly greater than per central tower seeding
    elif centralTowerEt <= getTowerEt(event, 0, 2) or centralTowerEt <= getTowerEt(event, 1, 2) or centralTowerEt <= getTowerEt(event, 2, 0) or centralTowerEt <= getTowerEt(event, 2, 1) or centralTowerEt <= getTowerEt(event, 2, 2):
        return 0
    else:
        return 1 

# Truth-match taus to TOBs
def eventTruthMatchedTOBs(event, run, tree=None):
    '''
    Return TOBs that have been matched to true taus. Only TOBs that pass Run-II seeding will be matched

    :param event: Event in ROOT TTree holding truth, reco, and TOB information
    :param run: Which run paradigm we are matching for, currently accepts 'Run2' or 'Run3'
    :param tree: Custom Tree class instance holding event settings

    :return: List of truth-matched TOBs, the first element of the list is the TOB and the second is the Pt of the true tau it has been matched to.
    '''
    from ROOT import TVector3

    matchedTOBs = []

    # Loop over true taus in event
    for i in range(event.truth_SelectedTau_n):
        # Build truth vector
        trueVector = TVector3(0, 0, 0)
        trueVector.SetPtEtaPhi(event.truth_SelectedTau_ETVisible[i], event.truth_SelectedTau_EtaVisible[i], event.truth_SelectedTau_PhiVisible[i])

        # Find reco tau with minimum dR from truth tau
        minTrueRecodR = float('inf')
        trueRecodR = float('inf')
        closestReco = None
        closestRecoNum = -1
        for j in range(event.reco_SelectedTau_n):
            # Build reco vector
            recoVector = TVector3(0, 0, 0)
            recoVector.SetPtEtaPhi(event.reco_SelectedTau_ETCalo[j], event.reco_SelectedTau_tlv_eta[j], event.reco_SelectedTau_tlv_phi[j])
            trueRecodR = trueVector.DeltaR(recoVector)

            if trueRecodR < minTrueRecodR:
                minTrueRecodR = trueRecodR
                closestReco = recoVector
                closestRecoNum = j

        if minTrueRecodR > 0.3:
            matchedTOBs.append([-1, trueVector.Pt(), trueVector.Eta(), -1, -1, i, -1, -1])
            continue

        # Find TOB with minimum dR from reconstructed tau
        minRecoTOBdR = float('inf')
        recoTOBdR = float('inf')
        closestTOB = None
        closestTOBNum = -1
        for k, tob in enumerate(event.efex_AllTOBs):

            if run == 'Run2':
                # 3.99 to be consistent with Josefina's code
                if not tob.ppmIsMaxCore(3.99):
                    continue

            elif run == 'Run3':
                tob_event = event_from_tob(tree, tob)
                if not isCentralTowerSeed(tob_event):
                    continue

            else:
                print('Invalid run parameter in eventTruthMatchedTOBs, quitting...')
                exit()

            tobVector = TVector3(0, 0, 0)
            tobVector.SetPtEtaPhi(tob.largeTauClus(), tob.eta(), tob.phi())
            recoTOBdR = closestReco.DeltaR(tobVector) 
    
            if recoTOBdR < minRecoTOBdR:
                minRecoTOBdR = recoTOBdR
                closestTOB = tob
                closestTOBNum = k

        # If closest TOB is too far from reco the load partial entry
        if minRecoTOBdR > 0.3:
            matchedTOBs.append([-1, trueVector.Pt(), trueVector.Eta(), closestReco.Pt(), closestReco.Eta(), i, -1, -1])
            continue

        matchedTOBs.append([closestTOB, trueVector.Pt(), trueVector.Eta(), closestReco.Pt(), closestReco.Eta(), i, closestRecoNum, closestTOBNum])
      
    return matchedTOBs

def event_from_tob(tree, tob):
    '''
    Build a custom Event class instance from a TOB. Intended for formatting Et from TOB into 2D layers

    :param tree: Custom Tree class instance holding TOBs
    :param tob: TOB object
    :return: Custom Event class instance
    ''' 
    cells = createCellLists(tob)

    l0_cells = [0]*9
    l1_cells = [0]*36
    l2_cells = [0]*36
    l3_cells = [0]*9
    had_cells = [0]*9

    po_3x3_cells_to_array(l0_cells, cells[0])
    po_12x3_cells_to_array(l1_cells, cells[1])
    po_12x3_cells_to_array(l2_cells, cells[2])
    po_3x3_cells_to_array(l3_cells, cells[3])
    po_3x3_cells_to_array(had_cells, cells[4])

    l0_layer = load_root_layer_to_class(tree, l0_cells, 0)
    l1_layer = load_root_layer_to_class(tree, l1_cells, 1)
    l2_layer = load_root_layer_to_class(tree, l2_cells, 2)
    l3_layer = load_root_layer_to_class(tree, l3_cells, 3)
    had_layer = load_root_layer_to_class(tree, had_cells, 4)

    tob_event = Event(tree, l0_layer, l1_layer, l2_layer, l3_layer, had_layer, 0, 1, 0)

    return tob_event
