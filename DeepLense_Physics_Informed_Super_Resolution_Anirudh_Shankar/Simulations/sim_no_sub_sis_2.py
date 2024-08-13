import matplotlib.pyplot as plt
import numpy as np
import random

from lens import DeepLens


# Number of sims
num_sim = int(5e3)

for i in range(num_sim):
    lens = DeepLens()
    lens.make_single_halo(1e12)
    lens.make_no_sub()
    lens.set_instrument('Euclid')
    lens.make_source_light_mag()
    lens.simple_sim_2(75)
    File = lens.image_real
    break
    np.save('/users/mtoomey/scratch/deeplense/Model_II_test/no_sub/no_sub_sim_' + str(random.getrandbits(128)),File)



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