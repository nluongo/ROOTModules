from ROOTDefs import prepare_event, get_signal_and_background_files, set_et_tree_parameters, layer_reco_et
import ROOTDefs

print(ROOTDefs)
print(ROOTDefs.__file__)

#tsig, fsig, tback, fback = get_signal_and_background_files()

tsig, fsig, tback, fback = get_signal_and_background_files('/eos/user/n/nicholas/SWAN_projects/NewTauSamples/dataFiles/ztt_Output_formatted.root', 
                                                           '/eos/user/n/nicholas/SWAN_projects/NewTauSamples/dataFiles/output_MB80_formatted.root')
set_et_tree_parameters(tback)

audit_event = 0

sig_or_back = 0

if sig_or_back == 1:
    tree = tsig
else:
    tree = tback

event = prepare_event(tree, audit_event, 0, 0, 1)

print('L2 Layer Cells')
print(event.l2_layer.cell_et)
print('FCore Definition')
print(event.fcore_def)
print('FCore Core Et')
print(layer_reco_et(event.l2_layer, event.fcore_def[0][0], event.fcore_def[0][1], event.seed_eta, event.seed_phi))
print('FCore Isolation Et')
print(layer_reco_et(event.l2_layer, event.fcore_def[1][0], event.fcore_def[1][1], event.seed_eta, event.seed_phi))
print('FCore')
print(event.fcore)
