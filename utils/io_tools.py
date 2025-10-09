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


def save_to_netcdf(filename, data_dict):
    """
    Salva o appende risultati di simulazione in un file NetCDF compresso.
    La dimensione illimitata è sempre l'ultimo asse di ciascun array.

    Parameters
    ----------
    filename : str
        Nome del file NetCDF da creare o aggiornare.
    data_dict : dict[str, np.ndarray]
        Dizionario {nome_variabile: np.ndarray}.
        Tutti gli array devono avere la stessa ultima dimensione per poter essere appesi.
    """
    from netCDF4 import Dataset

    # controlla che tutte le ultime dimensioni coincidano
    last_dim = None
    for arr in data_dict.values():
        if last_dim is None:
            last_dim = arr.shape[-1]
        elif arr.shape[-1] != last_dim:
            raise ValueError("Tutti gli array devono avere la stessa ultima dimensione")

    if not os.path.exists(filename):
        # crea un nuovo file
        with Dataset(filename, "w", format="NETCDF4") as ds:
            for name, arr in data_dict.items():
                # definisci tutte le dimensioni, con l'ultima illimitata
                dims = []
                for i, size in enumerate(arr.shape):
                    dname = f"{name}_dim{i}"
                    if i == arr.ndim - 1:
                        ds.createDimension(dname, None)  # ultima = illimitata
                    else:
                        ds.createDimension(dname, size)
                    dims.append(dname)
                # crea variabile compressa
                var = ds.createVariable(name, arr.dtype, tuple(dims), zlib=True, complevel=4)
                var[...] = arr
    else:
        # append
        with Dataset(filename, "a") as ds:
            for name, arr in data_dict.items():
                if name not in ds.variables:
                    ds.createDimension(name, arr.size)

                    raise KeyError(f"La variabile {name} non esiste nel file")

                var = ds.variables[name]
                old_size = var.shape[-1]
                new_size = old_size + arr.shape[-1]

                # costruiamo slice per appendere sull'ultimo asse
                slc = (slice(None),) * (arr.ndim - 1) + (slice(old_size, new_size),)
                var[slc] = arr


def save_results_hdf5(data_dict, filename):
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
                dset.resize(new_size, axis=2)
                dset[:,:, -arr.shape[-1]:] = arr

    print(f"Created file {filename}" if mode == 'w' else f"Updated file {filename}")
    return
