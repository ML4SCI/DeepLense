import pandas as pd
import awkward as ak
import numpy as np
import math
import os
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser('Convert top benchmark h5 datasets to ROOT/awkd')
parser.add_argument('-i', '--inputdir', required=True, help='Directory that contains the input h5 file.')
parser.add_argument('-c', '--condition', default='all', choices=['train', 'val', 'test', 'all'], help='Create dataset for train/test/val/all.')
parser.add_argument('-m', '--mode', default='uproot', choices=['awkd', 'uproot', 'ROOT'], help='Mode to write ROOT files')
parser.add_argument('--max-event-size', type=int, default=100000, help='Maximum event size per output file.')
args = parser.parse_args()


def store_file_awkd(res_array_2d, res_array_1d, outpath):
    r"""Write .awkd files with awkward0
    """
    import awkward0 as ak0
    outpath += '.awkd'
    print('Saving to file', outpath, '...')
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    ak0.save(outpath, ak0.fromiter({**res_1d, **{'Part_'+k: res_2d[k] for k in res_2d}} for res_1d, res_2d in zip(res_array_1d, res_array_2d)), mode='w')
 

def store_file_uproot(res_array_2d, res_array_1d, outpath):
    r"""Write ROOT files with the latest feature in uproot(4)
    """
    import uproot
    def _check_uproot_version(uproot):
        v = uproot.__version__.split('.')
        v = int(v[0])*10000 + int(v[1])*100 + int(v[2])
        assert v >= 40104, "Uproot version should be >= 4.1.4 for the stable uproot-writing feature"

    _check_uproot_version(uproot)
    outpath += '.root'
    print('Saving to file', outpath, '...')
    ak_array2d = ak.from_iter(res_array_2d)
    ak_array1d = ak.from_iter(res_array_1d)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    with uproot.recreate(outpath, compression=uproot.LZ4(4)) as fw:
        # Note that 2D variable names prefixed with `Part_` due to uproot storing rule of jagged arrays
        fw['Events'] = {'Part': ak.zip({k:ak_array2d[k] for k in ak.fields(ak_array2d)}), **{k:ak_array1d[k] for k in ak.fields(ak_array1d) if k != 'nPart'}}
        fw['Events'].title = 'Events'


def store_file_ROOT(res_array_2d, res_array_1d, outpath):
    r"""Write ROOT files with PyROOT
    """
    import ROOT
    from array import array
    ROOT.ROOT.EnableImplicitMT(4)
    outpath += '.root'
    print('Saving to file', outpath, '...')
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    f = ROOT.TFile(outpath, 'recreate')
    f.SetCompressionAlgorithm(ROOT.kLZ4)
    f.SetCompressionLevel(4)
    try:
        tree = ROOT.TTree('Events', 'Events')
        # Reserve branches for in TTree
        dic = {}
        for var in res_array_1d[0]:
            vartype = 'i' if 'int' in str(type(res_array_1d[0][var])) else 'f'
            dic[var] = array(vartype, [1])
            tree.Branch(var, dic[var], f'{var}/{vartype.upper()}')
        for var in res_array_2d[0]:
            dic['Part_' + var] = ROOT.vector('float')()
            tree.Branch('Part_' + var, 'vector<float>', dic['Part_' + var])
        # Store variables
        for res_1d, res_2d in zip(res_array_1d, res_array_2d): # loop event by event
            for var in res_1d: # loop over variable names
                dic[var][0] = res_1d[var]
            for var in res_2d:
                dic['Part_' + var].clear()
                for v in res_2d[var]:
                    dic['Part_' + var].push_back(v)
            tree.Fill()
        f.Write()
    finally:
        f.Close()


def convert(input_files, output_file, store_file_func):
    #  List all 2D and 1D variables to store. Note that 2D variables will be prefixed with `Part_`
    varlist_2d = ['E', 'PX', 'PY', 'PZ']
    varlist_1d = ['truthE', 'truthPX', 'truthPY', 'truthPZ', 'ttv', 'is_signal_new']
    varlist_2d_new = ['E_log', 'P', 'P_log', 'Etarel', 'Phirel']
    varlist_1d_new = ['E_tot', 'PX_tot', 'PY_tot', 'PZ_tot', 'P_tot', 'Eta_tot', 'Phi_tot']
    idx, ifile = 0, 0
    res_array_2d, res_array_1d = [], []
    for filename in input_files:
        print('Reading table from:', filename, '...')
        with pd.HDFStore(os.path.join(args.inputdir, filename)) as store:
            df = store.select('table')

        print('Processing events ...')
        isfirst = True
        for origIdx, row in tqdm(df.iterrows()):

            if idx >= args.max_event_size:
                # Reach the max event limit per file. Store the current arrays into file
                store_file_func(res_array_2d, res_array_1d, os.path.join(args.inputdir, 'prep', f'{output_file}_{ifile}'))
                del res_array_2d, res_array_1d
                res_array_2d, res_array_1d = [], []
                ifile += 1
                idx = 0

            # First initiate 2d arrays
            res = {k:[] for k in varlist_2d + varlist_2d_new}
            nPart = 0
            for ipar in range(200):
                if row[f'E_{ipar}'] == 0.:
                    break
                for k in ['E', 'PX', 'PY', 'PZ']:
                    res[k].append(row[f'{k}_{ipar}'])
                res['E_log'].append(math.log(row[f'E_{ipar}']))
                res['P'].append(math.sqrt(row[f'PX_{ipar}']**2 + row[f'PY_{ipar}']**2 + row[f'PZ_{ipar}']**2))
                res['P_log'].append(math.log(res['P'][-1]))
                nPart += 1
            
            # Fill 1d arrays
            for k in varlist_1d:
                res[k] = row[k]
            res['E_tot'] = sum(res['E'])
            res['PX_tot'] = sum(res['PX'])
            res['PY_tot'] = sum(res['PY'])
            res['PZ_tot'] = sum(res['PZ'])
            res['P_tot'] = math.sqrt(res['PX_tot']**2 + res['PY_tot']**2 + res['PZ_tot']**2)
            res['Eta_tot'] = math.atanh(res['PZ_tot'] / res['P_tot'])
            res['Phi_tot'] = math.atan(res['PY_tot'] / res['PX_tot'])
            res['nPart'] = nPart
            res['origIdx'] = origIdx
            res['idx'] = idx
            
            # Calculate new 2d arrays
            for ipar in range(nPart):
                Eta = math.atanh(res['PZ'][ipar] / res['P'][ipar])
                Phi = math.atan(res['PY'][ipar] / res['PX'][ipar])
                res['Etarel'].append(np.sign(res['Eta_tot']) * (Eta - res['Eta_tot']))
                res['Phirel'].append(Phi - res['Phi_tot'] - 2 * math.pi * math.floor((Phi - res['Phi_tot']) / (2 * math.pi) + 0.5))

            # Store per event result
            res_array_2d.append({k:res[k] for k in varlist_2d + varlist_2d_new})
            res_array_1d.append({k:res[k] for k in res.keys() if k not in varlist_2d + varlist_2d_new})

            if isfirst:
                print(res)
                isfirst = False

            idx += 1

    # Save rest of events before finishing
    store_file_func(res_array_2d, res_array_1d, os.path.join(args.inputdir, 'prep', f'{output_file}_{ifile}'))


if __name__ == '__main__':
    store_file_func = store_file_awkd if args.mode == 'awkd' else \
        store_file_uproot if args.mode == 'uproot' else \
        store_file_ROOT if args.mode == 'ROOT' else None
    if args.condition == 'train':
        convert(input_files=['train.h5'], output_file='top_train', store_file_func=store_file_func)
    elif args.condition == 'val':
        convert(input_files=['val.h5'], output_file='top_val', store_file_func=store_file_func)
    elif args.condition == 'test':
        convert(input_files=['test.h5'], output_file='top_test', store_file_func=store_file_func)
    elif args.condition == 'all':
        convert(input_files=['train.h5'], output_file='top_train', store_file_func=store_file_func)
        convert(input_files=['val.h5'], output_file='top_val', store_file_func=store_file_func)
        convert(input_files=['test.h5'], output_file='top_test', store_file_func=store_file_func)
