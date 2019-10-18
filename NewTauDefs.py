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

# Truth-match taus to TOBs
def eventTruthMatchedTOBs(event):
  '''
  Return TOBs that have been matched to true taus. TOBs are required to have a seed as defined by eFEX firmware and lie within eta of 1.4. True taus are required to have Pt > 20 GeV.

  :param event: Event in ROOT TTree holding truth, reco, and TOB information

  :return: List of truth-matched TOBs, the first element of the list is the TOB and the second is the Pt of the true tau it has been matched to.
  '''
  from ROOT import TVector3

  matchedTOBs = []

  # Loop over true taus in event
  for i in range(event.truth_SelectedTau_n):
    # Build truth vector
    trueVector = TVector3(0, 0, 0)
    trueVector.SetPtEtaPhi(event.truth_SelectedTau_tlv_pt[i], event.truth_SelectedTau_tlv_eta[i], event.truth_SelectedTau_tlv_phi[i])

    # Find TOB with minimum dR from reconstructed tau
    minTrueTOBdR = float('inf')
    for j, tob in enumerate(event.efex_AllTOBs):
      tobVector = TVector3(0, 0, 0)
      tobVector.SetPtEtaPhi(tob.largeTauClus(), tob.eta(), tob.phi())
      trueTOBdR = trueVector.DeltaR(tobVector) 
  
      if trueTOBdR < minTrueTOBdR:
        minTrueTOBdR = trueTOBdR
        closestTOB = tob
        closestj = j

    # If closest TOB is too far from truth then throw away true
    if minTrueTOBdR > 0.3:
      continue

    matchedTOBs.append([closestTOB, event.truth_SelectedTau_tlv_pt[i], closestj])
    
  return matchedTOBs
