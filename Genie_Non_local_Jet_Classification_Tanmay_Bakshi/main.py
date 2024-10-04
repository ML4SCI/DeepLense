import torch
import awkward
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py as hp

""" DATASET USED : Quark Gluon Tagging Reference Dataset
~ Kasieczka, Gregor;    
~ Plehn, Tilman;
~ Thompson, Jennifer;
~ Russel, Michael;

A set of MC simulated training/testing events for the evaluation of top quark tagging architectures.

In total 1.2M training events, 400k validation events and 400k test events. Use “train” for training, “val” for validation during the training and “test” for final testing and reporting results.

Description

* 14 TeV, hadronic tops for signal, qcd diets background, Delphes ATLAS detector card with Pythia8
* No MPI/pile-up included
* Clustering of  particle-flow entries (produced by Delphes E-flow) into anti-kT 0.8 jets in the pT range [550,650] GeV
* All top jets are matched to a parton-level top within ∆R = 0.8, and to all top decay partons within 0.8
* Jets are required to have |eta| < 2
* The leading 200 jet constituent four-momenta are stored, with zero-padding for jets with fewer than 200
* Constituents are sorted by pT, with the highest pT one first
* The truth top four-momentum is stored as truth_px etc.
* A flag (1 for top, 0 for QCD) is kept for each jet. It is called is_signal_new
* The variable "ttv" (= test/train/validation) is kept for each jet. It indicates to which dataset the jet belongs. It is redundant as the different sets are already distributed as different files.
"""




