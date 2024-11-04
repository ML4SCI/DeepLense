import matplotlib.pyplot as plt
import numpy as np
# import random
from tqdm import tqdm
import sys

from lens import DeepLens

num_sim = int(5e3)

directory = ''
if len(sys.argv) == 2: directory = sys.argv[1] + '/'
for i in tqdm(range(num_sim)):
    lens = DeepLens()
    lens.make_single_halo(1e12)
    lens.make_old_cdm()
    lens.set_instrument('hst')
    lens.make_source_light_mag()
    lens.simple_sim_2(64, 1)
    File = lens.image_real
    np.save(directory+'data_model_3/cdm/cdm_sim_%d'%i,File)
    lens.simple_sim_2(128,2)
    File = lens.image_real
    np.save(directory+'data_model_3/cdm_HR/cdm_HR_sim_%d'%i,File)
    lens.get_alpha(128,0.0505)
    np.save(directory+'data_model_3/cdm_HR/cdm_alpha_sim_%d'%i,lens.alpha)
