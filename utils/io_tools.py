#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 17:23:38 2025

@author: paolos
"""
import os, sys
import glob
import numpy as np
from   pathlib import Path
import copy, re
import gc
import h5py
from   datetime import datetime

def natural_key(s):
    # divide in sequenze di numeri e non numeri
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def save_results(results, file, n_gb = 2, last_case = None):
    
        
    MAX_SIZE = n_gb * 1024**3
    
    pathfile = Path(file)
    parent   = str(pathfile.parent)
    stem     = str(pathfile.stem)

    # label = file.replace('.npz','')
    result_files = glob.glob(f"{parent}/{stem}*.npz")
    result_files = sorted(result_files, key=natural_key)    
    
    if len(result_files) != 0:
        file = result_files[-1]

        if os.path.getsize(file) > MAX_SIZE:

            filename =  "_".join(stem.split('_pt')[0:])
            if filename == '': filename = stem
            filename += f'_pt{len(result_files)+1}.npz'
            file = parent + '/' + filename

            np.savez_compressed(file, **results)
            print("Created file ",file)

        else:
            # If file already exists and with size less than MAX_SIZE, update it with new results
            saved_data = dict(np.load(file))
            for key, result in results.items():
                if key not in saved_data.keys():
                    # Add new case...
                    saved_data[key] = result
    
            np.savez_compressed(file, **saved_data)
            print("Updated file ",file)
            del(saved_data)
    else:
        np.savez_compressed(file, **results)
        print("Created file ",file)

    gc.collect()
    return

def remove_case_from_results(file, a = None, b = None):
    
    if a is None and b is None: sys.exit("Specify alpha or beta for the case to be removed")

    os.system(f"mv {file} {file}.bkp")    
    print(f"Created bkp file {file}.bkp.")

    a_keys = []
    b_keys = []
    
    data = dict(np.load(file+'.bkp'))
    if a is not None and b is not None:
        sys.exit("Not implemented yet!")

    elif a is not None:
        keys = [k for k in data.keys() if "a_{:.2f}_".format(a) in k ]

    elif b is not None:
        keys = [k for k in data.keys() if "b_{:.2f}".format(b) in k] 

    for k in keys:
        del(data[k])
        print(f"Deleted case {k}!")
        
    save_results(data, file)
    del(data)        
    
    return



def save_results_hdf5(data_dict, filename, attrs = None, version = 999):
    """
    Salva o appende i risultati di una simulazione in un file HDF5.

    Se il file o il dataset non esistono -> vengono creati.
    Se esistono -> i dati vengono appesi lungo l'ultimo asse.

    Parameters
    ----------
    filename : str
        Percorso del file .h5 su cui salvare.
    data_dict : dict
        Dizionario con {chiave: np.ndarray}.
        Tutti gli array devono essere compatibili per l'append sull'ultimo asse.
    """
    mode = 'a' if os.path.exists(filename) else 'w'

    with h5py.File(filename, mode) as f:
        
        if attrs is not None: 
            for k,v in attrs.items():
                f.attrs[k] = v
        f.attrs['edit_date'] = datetime.now().isoformat()
        f.attrs['version']   = version
        if mode == 'w':  f.attrs['craetion_date'] = datetime.now().isoformat()

        for key, arr in data_dict.items():
            arr = np.asarray(arr)

            if key not in f:
                # Crea un dataset nuovo, abilitando la possibilità di append
                maxshape = list(arr.shape)
                maxshape[-1] = None  # ultimo asse estensibile
                f.create_dataset(
                                    key, data=arr, maxshape=tuple(maxshape), compression="gzip"
                                )
            else:
                dset = f[key]
                # Controllo di compatibilità delle dimensioni (tutti tranne ultimo asse)
                
                if dset.shape[:-1] != arr.shape[:-1]:
                    raise ValueError(
                        f"Shape incompatibile per append: {key}: {dset.shape} vs {arr.shape}"
                    )

                # Estendi dataset sull’ultimo asse
                new_size = dset.shape[-1] + arr.shape[-1]
                dset.resize(new_size, axis=len(dset.shape)-1)
                dset[..., -arr.shape[-1]:] = arr

    print(f"Created file {filename}" if mode == 'w' else f"Updated file {filename}")
    return
