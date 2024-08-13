import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from lens import DeepLens


# Number of sims
num_sim = int(5e3)

for i in tqdm(range(num_sim)):
    lens = DeepLens()
    lens.make_single_halo_SIE(1e12)
    lens.make_no_sub()
    lens.set_instrument('Euclid')
    lens.make_source_light()
    lens.simple_sim(75,0.05)
    File = lens.image_real
    np.save('/home/anirudh/Documents/GSoC/LensSR/Simulations/data_model_1/no_sub_sie/no_sub_sie_sim_%d'%i,File)
    lens.simple_sim(150,0.025)
    File = lens.image_real
    np.save('/home/anirudh/Documents/GSoC/LensSR/Simulations/data_model_1/no_sub_sie_HR/no_sub_sie_HR_sim_%d'%i,File)
lens = DeepLens()
lens.make_single_halo_SIE(1e12)
lens.make_no_sub()
lens.set_instrument('Euclid')
lens.make_source_light()
lens.simple_sim(150,0.025)
lens.get_alpha(150,0.025)
print(lens.alpha.shape)
np.save('/home/anirudh/Documents/GSoC/LensSR/Simulations/data_model_1/no_sub_sie/alpha',lens.alpha)



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
