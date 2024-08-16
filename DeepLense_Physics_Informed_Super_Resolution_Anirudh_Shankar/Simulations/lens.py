import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import G, c, M_sun

# from pyHalo.preset_models import CDM


from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.SimulationAPI.sim_api import SimAPI


NUMPIX = 75

#TODO this is only valid for halo z=0.5
def axion_length_to_mass(length):
    """
        Converts de Broglie wavelenth to axion mass
    """
    return 0.06 * np.power(2*float(length),-1) * 1e-22

#TODO this is only valid for halo z=0.5
def axion_mass_to_length(mass):
    return axion_length_to_mass(mass)


"""
    lens - main class for DeepLense

"""
class DeepLens(object):

    """
        class constructor
    """
    def __init__(self,axion_mass=None,H0=70,Om0=0.3,Ob0=0.05,z_halo=0.5,z_gal=1.0):
        # Cosmology
        self.H0  = H0
        self.Om0 = Om0
        self.Ob0 = Ob0

        self.z_halo = z_halo
        self.z_gal  = z_gal

        # realizationsCDM = CDM(z_halo, z_gal,cone_opening_angle_arcsec=10)
        # self.astropy_instance = realizationsCDM.astropy_instance


        self.axion_mass = axion_mass

    def draw_old_cdm_sub_masses(self,m_sub_min=1e6,m_sub_max=1e10,n_sub=25,beta=-0.9):
        """
            Args:

            n_sub: mean number  of sub halos in FOV
            m_sub_min: Minimum mass of the sub halos (in solar mass)
            m_sub_max: Maximum mass of the sub halos (in solar mass)
            beta: slope (negative)

        """
        n_halos = np.random.poisson(n_sub)
        u = np.random.uniform(0, 1, size=n_sub)
        m_low_u, m_high_u = m_sub_min ** (beta + 1), m_sub_max ** (beta + 1)
        return (m_low_u + (m_high_u - m_low_u) * u) ** (1.0 / (beta + 1.0))


    def mass_to_radius(self,Mass,redshift_halo,redshift_gal):
        """
            mass_to_radius:
                Converts mass to Einstein radius

            Mass: Mass in solar masses

            redshift_halo: Redshift of the DM halo
            redshift_gal:  Redshift of the lensed galaxy
        """

        if redshift_gal < redshift_halo:
            raise Exception('Lensed galaxy must be at higher redshift than DM halo!')
            sys.exit()

        M_Halo = Mass * M_sun
        rad_to_arcsec = 206265

        # Choice of cosmology
        cosmo = FlatLambdaCDM(H0=self.H0,Om0=self.Om0,Ob0=self.Ob0)

        DL = cosmo.luminosity_distance(redshift_halo).to(u.m)

        # Luminosity distance to lensed galaxy
        DS = cosmo.luminosity_distance(redshift_gal).to(u.m)

        # Distance between halo and lensed galaxy
        DLS = DS - DL

        # Einstein radius
        theta = np.sqrt(4 * G * M_Halo/c**2 * DLS/(DL*DS))

        # Return radius in arcsecods
        radius_arcsec = theta * rad_to_arcsec

        return radius_arcsec.value
    
    def make_single_halo(self,mass):
        """
            Simply makes a single halo w/ all the goodies we need for lenstronomy
        """
        self.main_halo_mass = mass
        ER = self.mass_to_radius(mass,self.z_halo,self.z_gal)
        main_halo_type = 'SIS'  # You have many other possibilities available. Check out the SinglePlane class!
        kwargs_lens_main = {'theta_E': ER, 'center_x': 0.0, 'center_y': 0.0}
        self.lens_model_list = [main_halo_type]
        self.kwargs_lens_list = [kwargs_lens_main]
        self.lens_redshift_list = [self.z_halo]

    def make_single_halo_SIE(self, mass):
        self.main_halo_mass = mass
        ER = self.mass_to_radius(mass,self.z_halo,self.z_gal)
        main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
        kwargs_lens_main = {'theta_E': ER, 'e1': 0.1, 'e2': 0.0, 'center_x': 0.0, 'center_y': 0.0}
        # kwargs_shear = {'gamma1': 0.05, 'gamma2': 0}
        self.lens_model_list = [main_halo_type]
        self.kwargs_lens_list = [kwargs_lens_main]
        self.lens_redshift_list = [self.z_halo]

    def make_single_halo_SIE_shear(self,mass):
        """
            Simply makes a single halo w/ all the goodies we need for lenstronomy
        """
        self.main_halo_mass = mass
        ER = self.mass_to_radius(mass,self.z_halo,self.z_gal)
        main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
        kwargs_lens_main = {'theta_E': ER, 'e1': 0.1, 'e2': 0, 'center_x': 0, 'center_y': 0.0}
        kwargs_shear = {'gamma1': 0.05, 'gamma2': 0}
        self.lens_model_list = [main_halo_type, 'SHEAR']
        self.kwargs_lens_list = [kwargs_lens_main, kwargs_shear]
        self.lens_redshift_list = [self.z_halo,self.z_halo]

    def axion_length_to_mass(self,length):
        """
            Converts de Broglie wavelenth to axion mass
        """
        return 0.06 * np.power(2*length,-1) * 1e-22

    def make_no_sub(self):
        """
            Init. no substructure simulation
        """
        self.lens_model_class = LensModel(self.lens_model_list)

    def make_vortex(self,vort_mass,res=100):
        """
            make_vortex:
                Calculates info needed by lenstronomy to make a vortex
        """
        sub_element = vort_mass/res
        sub_array   = sub_element * np.ones(int(res))
        # Array of Einstein radius
        E_list = self.mass_to_radius(sub_array,self.z_halo,self.z_gal)
        # Convert axion mass to de Broglie wavelength
        de_Broglie = axion_mass_to_length(self.axion_mass)
        # Find temp coordinates for each element of vortex mass
        coords = np.linspace(-1*de_Broglie,de_Broglie,res)
        # Set center of vortex to center of image
        xx, yy = 0.0, 0.0
        # Set a random angle for vortex to rotate through
        ang = 2 * np.pi * np.random.random()

        # Now get list of rotated coordinates
        center_x_list = []
        center_y_list = []
        for i in range(len(E_list)):
            x = (coords[i] * np.cos(ang) - xx)
            center_x_list.append(x)
            y = (coords[i] * np.sin(ang) - yy)
            center_y_list.append(y)
        center_x_list = np.array(center_x_list)
        center_y_list = np.array(center_y_list)

        subhalo_type = "POINT_MASS"
        for i in range(len(E_list)):
            self.lens_model_list.append(subhalo_type)
            self.kwargs_lens_list.append({'theta_E':E_list[i], 'center_x':center_x_list[i], 'center_y':center_y_list[i]})
            self.lens_redshift_list.append(self.z_halo)

        self.lens_model_class = LensModel(self.lens_model_list)

    def make_old_cdm(self):
        """
            Makes old style of CDM substructure - i.e. point masses drawn form SHMD with beta = -1.9
        """
        sub_array = self.draw_old_cdm_sub_masses()
        E_list = self.mass_to_radius(sub_array,self.z_halo,self.z_gal)
       
        subhalo_type = "POINT_MASS" 
        for i in range(len(E_list)):
            self.lens_model_list.append(subhalo_type)
            #x and y position
            r, th = np.random.uniform(0.25,2.0), np.random.uniform(0,2*np.pi)
            x1,x2 = r*np.sin(th), r*np.cos(th)
            self.kwargs_lens_list.append({'theta_E':E_list[i], 'center_x':x1, 'center_y':x2})
            self.lens_redshift_list.append(self.z_halo)

        self.lens_model_class = LensModel(self.lens_model_list)




    def make_source_light(self):
        """
            Make light profile
        """
        center_x,center_y = np.random.uniform(-0.35,0.35,2)

        # Sersic parameters in the initial simulation for the source
        kwargs_sersic = {'amp': 20, 'R_sersic': 0.25, 'n_sersic': 1, 'e1': -0.1, 'e2': 0.1,
                        'center_x': center_x, 'center_y': center_y}
        self.source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic]
        self.source_redshift_list = [self.z_gal]
        self.source_model_class = LightModel(self.source_model_list)

    def make_source_light_mag(self):
        """
            Make light profile
        """
        center_x,center_y = np.random.uniform(-0.35,0.35,2)
        # Sersic parameters in the initial simulation for the source
        kwargs_sersic = {'magnitude': 20, 'R_sersic': 0.25, 'n_sersic': 1, 'e1': -0.1, 'e2': 0.1,
                        'center_x': center_x, 'center_y': center_y}
        self.source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic]
        self.source_redshift_list = [self.z_gal]
        self.source_model_class = LightModel(self.source_model_list)

        

    def simple_sim(self):
        # import main simulation class of lenstronomy
        from lenstronomy.Util import util
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        import lenstronomy.Util.image_util as image_util
        from lenstronomy.ImSim.image_model import ImageModel


        # data specifics
        background_rms = 1.e-2  #  background noise per pixel
        exp_time = 10**np.random.uniform(3,3.5)  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = NUMPIX  #  cutout pixel size per axis
        deltaPix = 0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.087  # full width at half maximum of PSF
        psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # generate the coordinate grid and image properties (we only read out the relevant lines we need)
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, center_ra=0, center_dec=0, subgrid_res=1, inverse=False)


        kwargs_data = {'background_rms': background_rms,  # rms of background noise
               'exposure_time': exp_time,  # exposure time (or a map per pixel)
               'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
               'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel 
               'transform_pix2angle': Mpix2coord,  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
               'image_data': np.zeros((numPix, numPix))  # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.
              }

        data_class = ImageData(**kwargs_data)
        # generate the psf variables
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}

        # if you are using a PSF estimate from e.g. a star in the FoV of your exposure, you can set
        #kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': deltaPix, 'kernel_point_source': 'odd numbered 2d grid with centered star/PSF model'}


        psf_class = PSF(**kwargs_psf)
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

        imageModel = ImageModel(data_class, psf_class, lens_model_class=self.lens_model_class, 
                        source_model_class=self.source_model_class,kwargs_numerics=kwargs_numerics,
                        lens_light_model_class=None)

        # generate image
        image_model = imageModel.image(self.kwargs_lens_list,self.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)

        poisson = image_util.add_poisson(image_model, exp_time=exp_time)
        bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
        # image_real = exp_time * (image_model + poisson + bkg)
        image_real = exp_time * (image_model)

        self.image_real = np.random.poisson(image_real.clip(min=0))#print(np.isnan(image_real).any())
        self.image_model = image_model
        self.poisson = poisson
        self.bkg = bkg


        data_class.update_data(image_real)
        kwargs_data['image_data'] = image_real

    def simple_sim_2(self):
        """
            Same structure as simple_sim but with Euclid resolution
        """
        from lenstronomy.SimulationAPI.sim_api import SimAPI

        kwargs_model_physical = {'lens_model_list': self.lens_model_list,  # list of lens models to be used
                              'lens_redshift_list': self.lens_redshift_list,  # list of redshift of the deflections
                              'source_light_model_list': self.source_model_list,  # list of extended source models to be used
                              'source_redshift_list': self.source_redshift_list,  # list of redshfits of the sources in same order as source_light_model_list
                              'cosmo': self.astropy_instance,  # astropy.cosmology instance
                              'z_source_convention': 2.5,
                              'z_source': 1.0,} 

        
        #######################################################################
        numpix = NUMPIX  # number of pixels per axis of the image to be modelled
    
        # here we define the numerical options used in the ImSim module. 
        # Have a look at the ImageNumerics class for detailed descriptions.
        # If not further specified, the default settings are used.
        kwargs_numerics = {'point_source_supersampling_factor': 1}

        #######################################################################
        sim = SimAPI(numpix=numpix, kwargs_single_band=self.kwargs_single_band, kwargs_model=kwargs_model_physical)
        imSim = sim.image_model_class(kwargs_numerics)
                   
        _, kwargs_source, _ = sim.magnitude2amplitude(None,self.kwargs_source)


        image = imSim.image(self.kwargs_lens_list,kwargs_source,None)

        self.image_model = image
        self.poisson = sim.noise_for_model(model=image)
        self.image_real = self.image_model + self.image_model

    def set_instrument(self,inst_name):
        """
	        set_instrument
			
        		Method which sets up the specifics for a given instrument.

        """
        if inst_name == None:
            pass
        # Note .lower() here is just to make string lower case
        elif inst_name.lower() == 'euclid':
            from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
            Euc = Euclid(band='VIS',psf_type='GAUSSIAN',coadd_years=6)
            self.kwargs_single_band = Euc.kwargs_single_band()
        else:
            pass
    
    def get_lensing_potential(self):
        """
        Compute the lensing potential
        """
        if not hasattr(self, 'lens_model_class'):
            raise AttributeError("Lens model is not initialized. Call make_no_sub, make_vortex, make_old_cdm, or make_single_halo first.")
        
        # Compute the lensing potential

        # euclid_delta = 0.1
        model_1_delta = 0.05
        arcsec_bound = model_1_delta*NUMPIX/2
        x = np.linspace(-arcsec_bound, arcsec_bound, NUMPIX)
        y = np.linspace(-arcsec_bound, arcsec_bound, NUMPIX)
        xx, yy = np.meshgrid(x, y)
        x_coords = xx.ravel()
        y_coords = yy.ravel()

        # Get the lensing potential
        self.potential = self.lens_model_class.potential(x_coords, y_coords, self.kwargs_lens_list)

        # Reshape potential to match the grid shape
        self.potential = self.potential.reshape((NUMPIX, NUMPIX))

    def get_alpha(self):
        """
        Compute the lensing potential
        """
        if not hasattr(self, 'lens_model_class'):
            raise AttributeError("Lens model is not initialized. Call make_no_sub, make_vortex, make_old_cdm, or make_single_halo first.")
        
        # Compute the lensing potential

        # euclid_delta = 0.1
        model_1_delta = 0.05
        arcsec_bound = model_1_delta*NUMPIX/2
        x = np.linspace(-arcsec_bound, arcsec_bound, NUMPIX)
        y = np.linspace(-arcsec_bound, arcsec_bound, NUMPIX)
        xx, yy = np.meshgrid(x, y)
        x_coords = xx.ravel()
        y_coords = yy.ravel()

        # Get the lensing potential
        self.alpha_x, self.alpha_y = self.lens_model_class.alpha(x_coords, y_coords, self.kwargs_lens_list)

        # Reshape potential to match the grid shape
        self.alpha_x, self.alpha_y = self.alpha_x.reshape((NUMPIX, NUMPIX)), self.alpha_y.reshape((NUMPIX, NUMPIX))
        self.alpha = np.sqrt(self.alpha_x**2 + self.alpha_y**2)

        

if __name__ == "__main__":

    lens = DeepLens(axion_mass=1e-24)
    lens.make_single_halo(1e12)
    lens.make_vortex(1e10)
    lens.set_instrument('Euclid')
    lens.make_source_light_mag()
    lens.simple_sim_2()

    print(lens.kwargs_single_band)

    
    import matplotlib.pyplot as plt

    plt.imshow(lens.image_real);plt.show()
    """
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
    """
