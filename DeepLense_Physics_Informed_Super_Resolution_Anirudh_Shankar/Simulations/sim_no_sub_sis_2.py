import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import sys

from lens import DeepLens


# Number of sims
num_sim = int(5e3)
directory = ''
if len(sys.argv) == 2: directory = sys.argv[1] + '/'

lens = DeepLens()
lens.make_single_halo(1e12)
lens.make_no_sub()
lens.set_instrument('Euclid')
lens.make_source_light_mag()
lens.simple_sim_2(64, 1)
lens.get_alpha(64, 0.101)
print(lens.alpha)
for i in tqdm(range(num_sim)):
    lens = DeepLens()
    lens.make_single_halo(1e12)
    lens.make_no_sub()
    lens.set_instrument('Euclid')
    lens.make_source_light_mag()
    lens.simple_sim_2(64, 1)
    File = lens.image_real
    np.save(directory+'data_model_2/no_sub/no_sub_sim_%d'%i,File)
    lens.simple_sim_2(128,2)
    File = lens.image_real
    np.save(directory+'data_model_2/no_sub_HR/no_sub_HR_sim_%d'%i,File)



if False:
    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.imshow(lens.image_real)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.sqrt(lens.image_real))
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(lens.poisson)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(lens.bkg)
    plt.colorbar()
    plt.show()