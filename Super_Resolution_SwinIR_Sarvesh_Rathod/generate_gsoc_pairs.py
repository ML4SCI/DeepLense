import copy
import numpy as np
import h5py
import random
import os

from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from astropy.cosmology import FlatLambdaCDM

# Setup paths - Adjusted to be relative to where we run it (re-verify execution location)
# We will run this from d:\ML4SCI\DeepLense\Super_Resolution_SwinIR_Sarvesh_Rathod
# So the OUTPUT_DIR is just 'pairs'
# The DATA_DIR is relative to here: ../DeepLenseSim/data/Galaxy10_DECals.h5

OUTPUT_DIR = "pairs" 
DATA_FILE = r"../DeepLenseSim/data/Galaxy10_DECals.h5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cosmolgy
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)

# Instrument Configs
# G-band (VIS)
Euclid_g = Euclid(band='VIS', psf_type='GAUSSIAN', coadd_years=6)
kwargs_g_band = Euclid_g.kwargs_single_band()

def get_simulation_api(numpix, kwargs_band):
    kwargs_model = {'lens_model_list': ['SIE'], 
                    'lens_redshift_list': [0.5], 
                    'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'], 
                    'source_light_model_list': ['INTERPOL'], 
                    'source_redshift_list': [1.0], 
                    'cosmo': cosmo, 
                    'z_source_convention': 2.5, 
                    'z_source': 2.5}
    return SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model)

def simulate_pair(image_galaxy, sigma_v, source_pos_xx, source_pos_yy, source_ang, idx):
    # 1. Simulate HR (128x128) - "hsc" in our pipeline
    
    # HR Config
    kwargs_g_hr = copy.deepcopy(kwargs_g_band)
    kwargs_g_hr['pixel_scale'] = 0.05 # Half of Euclid (0.1) -> higher res
    sim_hr = get_simulation_api(128, kwargs_g_hr)
    imSim_hr = sim_hr.image_model_class({'point_source_supersampling_factor': 1})
    
    # LR Config
    kwargs_g_lr = copy.deepcopy(kwargs_g_band)
    kwargs_g_lr['pixel_scale'] = 0.1 # Standard Euclid
    sim_lr = get_simulation_api(64, kwargs_g_lr)
    imSim_lr = sim_lr.image_model_class({'point_source_supersampling_factor': 1})
    
    # Common Physics
    kwargs_mass = [{'sigma_v': sigma_v, 'center_x': 0, 'center_y': 0, 'e1': 0.0, 'e2': 0}]
    
    # Process Source Galaxy
    image_data = image_galaxy[:,:,0].astype(float) # Use G-band only for simplicity
    image_data -= np.median(image_data[:50, :50])
    
    # Source Light
    kwargs_source_mag = [{'magnitude': 22, 'image': image_data, 'scale': 0.0025, 'phi_G': source_ang, 'center_x': source_pos_xx, 'center_y': source_pos_yy}]
    
    # Lens Light
    kwargs_lens_light_mag = [{'magnitude': 17, 'R_sersic': 0.4, 'n_sersic': 2.3, 'e1': 0, 'e2': 0.05, 'center_x': 0, 'center_y': 0},
                             {'magnitude': 28, 'R_sersic': 1.5, 'n_sersic': 1.2, 'e1': 0, 'e2': 0.3, 'center_x': 0, 'center_y': 0}]

    # Conversion
    kwargs_lens_light_hr, kwargs_source_hr, _ = sim_hr.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag)
    kwargs_lens_light_lr, kwargs_source_lr, _ = sim_lr.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag)
    
    kwargs_lens = sim_hr.physical2lensing_conversion(kwargs_mass=kwargs_mass) # Same mass for both

    # Generate Images
    image_hr = imSim_hr.image(kwargs_lens, kwargs_source_hr, kwargs_lens_light_hr)
    image_lr = imSim_lr.image(kwargs_lens, kwargs_source_lr, kwargs_lens_light_lr)

    # Add Noise
    image_hr += sim_hr.noise_for_model(model=image_hr) * 0.5 
    image_lr += sim_lr.noise_for_model(model=image_lr)

    # Fix filenames to: {Index}_real_hsc.npy
    np.save(os.path.join(OUTPUT_DIR, f'{idx}_real_hsc.npy'), image_hr) # hsc = HR
    np.save(os.path.join(OUTPUT_DIR, f'{idx}_real_hst.npy'), image_lr) # hst = LR

def main():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found at {DATA_FILE}")
        return

    print("Loading Galaxy10 Data...")
    try:
        with h5py.File(DATA_FILE, 'r') as F:
            images = np.array(F['images'])
            typ = np.array(F['ans'])
            z = np.array(F['redshift'])
    except Exception as e:
        print(f"Failed to read H5 file: {e}")
        return

    # Filter Spirals
    unbarred_spiral = np.where(typ == 6)
    images_ref = images[unbarred_spiral]
    z_ref = z[unbarred_spiral]
    
    indx_img_zp1 = np.where(z_ref < 0.02)
    img_zp1 = images_ref[indx_img_zp1]
    
    # Valid indices 
    arr = [2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,54,55]
    
    # Generate smaller set for quick test, then user can increase
    num_sims = 2500 
    print(f"Generating {num_sims} pairs based on REAL galaxies...")
    
    for i in range(num_sims):
        index = np.random.randint(0, len(arr))
        # Rand params
        sigma_v = np.random.normal(260, 20)
        source_pos_xx = np.random.uniform(-0.3, 0.3)
        source_pos_yy = np.random.uniform(-0.3, 0.3)
        source_ang = np.random.uniform(-np.pi, np.pi)
        
        try:
            simulate_pair(img_zp1[arr[index]], sigma_v, source_pos_xx, source_pos_yy, source_ang, i)
        except Exception as e:
            print(f"Error generating pair {i}: {e}")
            continue
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_sims}")

    print("Done! Real simulated pairs saved.")

if __name__ == "__main__":
    main()
