# calculations
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from datetime import datetime
# from numba import njit

# plotting
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from scipy.interpolate.ndgriddata import NearestNDInterpolator
from scipy.spatial.transform import Rotation as rot

# geometry
import trimesh as tm
from trimesh.ray.ray_pyembree import RayMeshIntersector # FASTER
# from trimesh.ray.ray_triangle import RayMeshIntersector # SLOWER
from trimesh.triangles import closest_point, points_to_barycentric 

# AI
from sklearn.preprocessing import MinMaxScaler

# other
import sys
import os
import copy
from functools import partial
from itertools import repeat
import time
import gc

from classes.Constants import Constants
# from classes.Jitted    import *

np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=np.nan)

# matplotlib.use('Qt5Agg')

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class with information about the particles contained in the domain.

#   TO DO
#   - Add option to apply heat flux as boundary condition
#   

class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry, phonon):
        super(Population, self).__init__()

        self.args = arguments
        
        self.lookup    = bool(int(self.args.lookup[0]))
        self.damping   = float(self.args.lookup[1])
        self.conv_crit = float(self.args.conv_crit[0])
        

        self.norm = self.args.energy_normal[0]
        
        self.n_of_subvols   = geometry.n_of_subvols

        self.empty_subvols      = self.args.empty_subvols
        self.n_of_empty_subvols = len(self.empty_subvols)
        
        self.particle_type = self.args.particles[0]

        if self.particle_type == 'pmps':
            self.particles_pmps   = float(self.args.particles[1])
            self.N_p              = int(np.ceil(self.particles_pmps*phonon.number_of_active_modes*self.n_of_subvols))
            self.particle_density = self.N_p/geometry.volume
        elif self.particle_type == 'total':
            self.N_p              = int(np.ceil(float(self.args.particles[1])))
            self.particles_pmps   = self.N_p/(phonon.number_of_active_modes*self.n_of_subvols)
            self.particle_density = self.N_p/geometry.volume
        elif self.particle_type == 'pv':
            self.particle_density = float(self.args.particles[1])
            self.N_p              = int(np.ceil(self.particle_density*geometry.volume))
            self.particles_pmps   = self.N_p/(phonon.number_of_active_modes*(self.n_of_subvols - self.n_of_empty_subvols))
        
        self.dt = float(self.args.timestep[0])   # ps
        self.t  = 0.0
        
        if geometry.subvol_type == 'slice':
            self.slice_axis    = geometry.slice_axis
            self.slice_length  = geometry.slice_length  # angstrom
        
        if geometry.subvol_type == 'sphere':
            self.repeat_lim = 2*self.particles_pmps+1
        else:
            self.repeat_lim = 5*self.particles_pmps
        
        self.subvol_volume  = geometry.subvol_volume  # angstrom³

        self.bound_cond          = geometry.bound_cond
        
        self.rough_facets        = geometry.rough_facets        # which facets have roughness as BC
        self.rough_facets_values = geometry.rough_facets_values # their values of roughness and correlation length        
        
        self.connected_facets = np.array(self.args.connect_facets).reshape(-1, 2)

        self.offset = float(self.args.offset[0]) # offset margin to avoid problems with trimesh collision detection - standard 2*tm.tol.merge = 2e-8
        
        self.T_distribution   = self.args.temp_dist[0]
        self.T_reference      = float(self.args.reference_temp[0])

        self.colormap = self.args.colormap[0]
        self.fig_plot = self.args.fig_plot
        self.rt_plot  = self.args.rt_plot

        self.current_timestep = 0

        print('Initialising reservoirs...')
        # number of reservoirs
        self.n_of_reservoirs = (self.bound_cond == 'T').sum()+(self.bound_cond == 'F').sum()
        if self.n_of_reservoirs > 0:
            self.initialise_reservoirs(geometry, phonon)            # initialise reservoirs
        else:
            print('No reservoir to be initialised.')

        print('Initialising population...')

        self.initialise_all_particles(geometry, phonon) # initialising particles

        self.n_timesteps, self.collision_facets, self.collision_positions = self.timesteps_to_boundary(self.positions, self.group_vel, geometry) # calculating timesteps to boundary

        self.residue   = np.ones(self.occupation_lookup.shape)*2*self.conv_crit

        self.results_folder_name = self.args.results_folder

        print('Creating convergence file...')
        self.open_convergence()
        self.write_convergence()

        if len(self.rt_plot) > 0:
            print('Starting real-time plot...')
            self.rt_graph, self.rt_fig = self.init_plot_real_time(geometry, phonon)
        
        

        print('Initialisation done!')
        
    def initialise_modes(self, phonon):
        '''Generate first modes.'''

        # creating unique mode matrix
        self.unique_modes = np.vstack(np.where(~phonon.inactive_modes_mask)).T

        if self.particle_type == 'pmps':
            # if particles per mode, per subvolume are defined, use tiling
            modes = np.tile( self.unique_modes,  (int(np.ceil(self.particles_pmps*(self.n_of_subvols-self.n_of_empty_subvols))), 1) )
            modes = modes[:self.N_p, :]
        else:
            # if not, generate randomly
            modes = np.random.randint(low = 0, high = phonon.number_of_active_modes , size = self.N_p)
            modes = self.unique_modes[modes, :]

        return modes.astype(int)

    def enter_probability(self, geometry, phonon):
        '''Calculates the probability of a particle with the modes of a given material (phonon)
        to enter the facets of a given geometry (geometry) with imposed boundary conditions.'''

        vel = np.transpose(phonon.group_vel, (0, 2, 1))           # shape = (Q, 3, J) - Group velocities of each mode
        normals = -geometry.mesh.facets_normal[self.res_facet, :] # shape = (R, 3)    - unit normals of each facet with boundary conditions
                                                                  # OBS: normals are reversed to point inwards (in the direction of entering particles).
        group_vel_parallel = np.dot(normals, vel)                 # shape = (R, Q, J) - dot product = projection of velocity over normal
        
        # Probability of a particle entering the domain
        enter_prob = group_vel_parallel*self.dt/self.bound_thickness.reshape(-1, 1, 1)   # shape = (F, Q, J)
        enter_prob = np.where(enter_prob < 0, 0, enter_prob)

        return enter_prob

    def generate_positions(self, number_of_particles, mesh, key):
        '''Initialise positions of a given number of particles'''
        
        if key == 'random':
            positions = tm.sample.volume_mesh(mesh, number_of_particles)
            in_mesh = np.where( tm.proximity.signed_distance(mesh, positions) >= 2*self.offset )[0]
            positions = positions[in_mesh, :]
            
            while positions.shape[0]<number_of_particles:
                new_positions = tm.sample.volume_mesh(mesh, number_of_particles-positions.shape[0]) # this function produces UP TO n points. So it could be 0. Go figure...
                if new_positions.shape[0]>0:
                    in_mesh = np.where( tm.proximity.signed_distance(mesh, new_positions) >= 2*self.offset )[0]
                    new_positions = new_positions[in_mesh, :]
                    positions = np.vstack((positions, new_positions))

        elif key == 'center':
            center = mesh.center_mass
            positions = np.ones( (number_of_particles, 3) )*center # Generate all points at the center of the bounding box, positions in angstrom

        return positions

    def initialise_all_particles(self, geometry, phonon):
        '''Uses Population atributes to generate all particles.'''
        
        # initialising positions one mode at a time (slower but uses less memory)
        self.positions = np.zeros((0, 3))

        if self.args.part_dist[0] == 'random_domain':
            number_of_particles = self.N_p
            self.positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'random')
        
        elif self.args.part_dist[0] == 'center_domain':
            number_of_particles = self.N_p
            self.positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'center')
        
        # elif self.args.part_dist[0] == 'random_subvol':
        #     counter = 0

        #     indexes                     = np.ones(self.n_of_subvols).astype(bool)
        #     indexes[self.empty_subvols] = False

        #     filled_volume = geometry.subvol_volume[indexes].sum()
        #     for i in range(self.n_of_subvols):
        #         if i not in self.empty_subvols:
                    
        #             number_of_particles = int(np.ceil(self.N_p*(geometry.subvol_volume[i]/filled_volume)))

        #             if counter + number_of_particles > self.N_p:
        #                 number_of_particles = self.N_p - counter
        #                 counter = self.N_p
        #             else:
        #                 counter += number_of_particles

        #             new_positions  = self.generate_positions(number_of_particles, geometry.subvol_meshes[i], key = 'random')
        #             self.positions = np.vstack((self.positions, new_positions))

        elif self.args.part_dist[0] == 'random_subvol':
            counter = np.zeros(self.n_of_subvols, dtype = int)

            n = self.N_p*geometry.subvol_volume/(geometry.subvol_volume.sum()-geometry.subvol_volume[self.empty_subvols].sum()) # number of particles for each subvolume
            n = np.ceil(n).astype(int)
            n[self.empty_subvols] = 0

            x = [np.zeros((0, 3)) for _ in range(self.n_of_subvols)]

            while np.any(counter < n):
                print(counter, n - counter)
                x_new = self.generate_positions(n.sum() - counter.sum(), geometry.mesh, key = 'random')

                sv_id = geometry.subvol_classifier.predict(geometry.scale_positions(x_new))

                x_new_list = [x_new[sv_id[:, i], :] for i in range(self.n_of_subvols) if i not in self.empty_subvols]

                for i in range(self.n_of_subvols):
                    if i not in self.empty_subvols:
                        
                        N = n[i] - counter[i]
                        x[i] = np.vstack((x[i], x_new_list[i][:N, :]))

                        counter[i] = x[i].shape[0]
            
            self.positions = np.vstack(x)
            self.positions = self.positions[:self.N_p, :]

        elif self.args.part_dist[0] == 'center_subvol':
            counter = 0
            
            indexes                     = np.ones(self.n_of_subvols).astype(bool)
            indexes[self.empty_subvols] = False

            filled_volume = geometry.subvol_volume[indexes].sum()

            for i in range(self.n_of_subvols):
                if i not in self.empty_subvols:

                    number_of_particles = int(np.ceil(self.N_p*(geometry.subvol_volume[i]/filled_volume)))
                    
                    if counter + number_of_particles > self.N_p:
                        number_of_particles = self.N_p - counter
                        counter = self.N_p
                    else:
                        counter += number_of_particles

                    new_positions = self.generate_positions(number_of_particles, geometry.subvol_meshes[i], key = 'center')
                    self.positions = np.vstack((self.positions, new_positions))

        # assigning properties from Phonon
        self.modes = self.initialise_modes(phonon) # getting modes
        self.omega, self.group_vel, self.wavevectors = self.assign_properties(self.modes, phonon)
        
        # initialising slice id
        self.subvol_id = self.get_subvol_id(geometry)

        # assign temperatures
        self.temperatures, self.subvol_temperature = self.assign_temperatures(self.positions, geometry)

        # occupation considering reference
        self.occupation_lookup = self.initialise_occupation_lookup(phonon)
        self.occupation        = phonon.calculate_occupation(self.temperatures, self.omega, reference = True)
        self.energies          = self.hbar*self.omega*self.occupation
        self.momentum          = self.hbar*self.wavevectors*self.occupation.reshape(-1, 1)
        
        # getting scattering arrays
        self.n_timesteps, self.collision_facets, self.collision_positions = self.timesteps_to_boundary(self.positions, self.group_vel, geometry)

        self.collision_cond = self.get_collision_condition(self.collision_facets)

        self.calculate_energy(geometry, phonon, lookup = self.lookup)
        self.calculate_heat_flux(geometry, phonon, lookup = self.lookup)
        self.calculate_momentum(geometry, phonon, lookup = self.lookup)

    def initialise_reservoirs(self, geometry, phonon):
        
        # which facets contain attached reservoirs
        self.res_facet = geometry.res_facets
        self.res_bound_values = geometry.res_values
        self.res_bound_cond   = geometry.res_bound_cond

        mask_temp = geometry.res_bound_cond == 'T' # which RESERVOIRS (res1, res2, res3...) have imposed temperature, boolean
        mask_flux = geometry.res_bound_cond == 'F' # which RESERVOIRS (res1, res2, res3...) have imposed heat flux  , boolean

        # setting temperatures
        facets_temp = geometry.res_values[mask_temp] # imposed temperatures

        # initialising the array that defines the temperature of each reservoir
        self.res_facet_temperature            = np.ones(self.n_of_reservoirs) # initialise array with None, len = n_of_reservoirs
        self.res_facet_temperature[:]         = None
        self.res_facet_temperature[mask_temp] = facets_temp                   # attribute imposed temperatures
        self.res_facet_temperature[mask_flux] = facets_temp.mean()            # initialising with the mean of the imposed temperatures

        # The virtual volume of the reservoir is considered a extrusion of the facet.
        # The extrusion thickness is calculated such that one particle per mode is created,
        # obbeying the density of particles generated (i.e. thickness = number_of_modes/(particle_density*facet_area) );
        
        self.bound_thickness = phonon.number_of_active_modes/(self.particle_density*geometry.mesh.facets_area[self.res_facet])

        self.enter_prob = self.enter_probability(geometry, phonon)

        self.res_energy_balance   = np.zeros(self.n_of_reservoirs)
        self.res_heat_flux        = np.zeros((self.n_of_reservoirs, 3))
        self.res_momentum_balance = np.zeros((self.n_of_reservoirs, 3))

    def fill_reservoirs(self, geometry, phonon):

        # generate random numbers
        dice = np.random.rand(self.n_of_reservoirs, phonon.number_of_qpoints, phonon.number_of_branches)

        # check if particles entered the domain comparing with their probability
        fixed_np  = np.where(self.enter_prob < 0, 0, np.floor(self.enter_prob)) # number of particles that will enter every iteration
        
        in_modes_mask = dice <= (self.enter_prob - fixed_np) # shape = (R, Q, J)
        
        in_modes_np   = np.floor(self.enter_prob)+in_modes_mask.astype(int)

        # calculate how many particles entered each facet
        N_p_facet = in_modes_np.astype(int).sum(axis = (1, 2))    # shape = (R,)

        # initialise new arrays
        self.res_positions   = np.zeros((0, 3))
        self.res_modes       = np.zeros((0, 2))
        self.res_facet_id    = np.zeros(0)
        self.res_dt_in       = np.zeros(0)

        for i in range(self.n_of_reservoirs):                         # for each reservoir
            n      = N_p_facet[i]                                     # the number of particles on that facet
            if n > 0:
                facet  = self.res_facet[i]                                # get its facet index
                mesh   = geometry.res_meshes[i]                           # select boundary
                
                # adding fixed particles
                c = in_modes_np[i, :, :].max()  # gets the maximum number of particles of a single mode to be generated
                while c > 0:
                    c_modes = np.vstack(np.where(in_modes_np[i, :, :] == c)).T
                    if c == 1:
                        c_dt_in = self.dt*(1-(dice[i, c_modes[:, 0], c_modes[:, 1]]/self.enter_prob[i, c_modes[:, 0], c_modes[:, 1]]))
                    else:
                        r = np.random.rand(c_modes.shape[0])
                        c_dt_in = self.dt*(1-(c-1+r)/self.enter_prob[i, c_modes[:, 0], c_modes[:, 1]])
                    
                    c -= 1
                    
                    self.res_dt_in = np.concatenate((self.res_dt_in, c_dt_in)) # add to the time drifted inside the domain
                    self.res_modes = np.vstack((self.res_modes, c_modes     )).astype(int) # add to the modes

                self.res_facet_id  = np.concatenate((self.res_facet_id, np.ones(n)*facet)).astype(int) # add to the reservoir id
                
                # get modes 
                r_modes = self.res_modes[np.where(self.res_facet_id == facet)[0], :]
                r_v = phonon.group_vel[r_modes[:, 0], r_modes[:, 1], :]

                # generate positions on boundary
                new_positions, _ = tm.sample.sample_surface(mesh, n)

                # check if they will have trouble colliding and resample if needed
                _, index_ray = geometry.mesh.ray.intersects_id(ray_origins      = new_positions ,
                                                               ray_directions   = r_v             ,
                                                               return_locations = False         ,
                                                               multiple_hits    = False         )

                r_nan = ~np.in1d(np.arange(n), index_ray) # find which particles don't have rays
                r_nan = np.where(r_nan)[0]                # int slicing is faster than bool slicing

                n_new = r_nan.shape[0] # how many need to be resampled

                while n_new > 0:

                    new_positions[r_nan, :], _ = tm.sample.sample_surface(mesh, n_new)

                    # check if they will have trouble colliding and resample if needed
                    _, index_ray = geometry.mesh.ray.intersects_id(ray_origins      = new_positions[r_nan, :],
                                                                   ray_directions   = r_v[r_nan, :]          ,
                                                                   return_locations = False                  ,
                                                                   multiple_hits    = False                  )

                    new_nan = ~np.in1d(np.arange(n)[r_nan], index_ray) # find which particles don't have rays
                    new_nan = np.where(new_nan)[0]

                    r_nan = r_nan[new_nan]

                    n_new = r_nan.shape[0] # how many need to be resampled

                # adding to the new population
                self.res_positions = np.vstack((self.res_positions , new_positions ))                  # add to the positions
                
        if self.res_modes.shape[0]>0:
            
            self.res_group_vel   = phonon.group_vel[self.res_modes[:, 0], self.res_modes[:, 1], :]        # retrieve velocities

            self.res_omega       = phonon.omega[self.res_modes[:, 0], self.res_modes[:, 1]]               # retrieve frequencies
            
            self.res_wavevectors = phonon.wavevectors[self.res_modes[:, 0], :]

        facets_flux = np.where(self.bound_cond == 'F')[0] # which FACETS have imposed heat flux  , indexes
        mask_flux = np.isin(self.res_facet, facets_flux)  # which RESERVOIRS (res1, res2, res3...) have imposed heat flux  , boolean
        
        self.res_facet_temperature[mask_flux] = np.array(list(map(self.calculate_temperature_for_flux       ,
                                                                  np.arange(self.n_of_reservoirs)[mask_flux],
                                                                  repeat(geometry)                          ,
                                                                  repeat(phonon)                          ))) # calculate temperature
        
        indexes = np.where(self.res_facet_id.reshape(-1, 1) == self.res_facet)[1]   # getting RESERVOIR indexes

        self.res_temperatures = self.res_facet_temperature[indexes] # impose temperature values to the right particles

        self.res_energy_balance   = np.zeros(self.n_of_reservoirs)
        self.res_heat_flux        = np.zeros((self.n_of_reservoirs, 3))
        self.res_momentum_balance = np.zeros((self.n_of_reservoirs, 3))

        if self.res_modes.shape[0]>0:
            self.res_occupation = phonon.calculate_occupation(self.res_temperatures, self.res_omega, reference = True)
            self.res_energies   = self.hbar*self.res_omega*self.res_occupation
            self.res_momentum   = self.hbar*self.res_wavevectors*self.res_occupation.reshape(-1, 1)

            for i in range(self.n_of_reservoirs):
                facet  = self.res_facet[i]

                indexes = self.res_facet_id == facet

                self.res_energy_balance[i]      = self.res_energies[indexes].sum()
                self.res_momentum_balance[i, :] = self.res_momentum[indexes, :].sum(axis = 0)
                self.res_heat_flux[i, :]        = (self.res_group_vel[indexes, :]*self.res_energies[indexes].reshape(-1, 1)).sum(axis = 0)
        
    def calculate_temperature_for_flux(self, reservoir, geometry, phonon):
        facet  = self.res_facet[reservoir]
        normal = -geometry.mesh.facets_normal[facet, :]
        
        indexes_out = np.logical_and(self.n_timesteps<=0, self.collision_facets == facet)                      # particles leaving through that facet

        print('Checking logic')
        print((self.n_timesteps<=0).sum(), (self.collision_facets == facet).sum())
        
        modes_out      = self.modes[indexes_out, :]
        omega_out      = self.omega[indexes_out]
        occupation_out = self.occupation[indexes_out]+phonon.reference_occupation[modes_out[:, 0], modes_out[:, 1]]
        
        energies_out  = self.hbar*omega_out*occupation_out
        flux_out     = (energies_out.reshape(-1, 1)*self.group_vel[indexes_out, :]*normal).sum() # eV angstrom/ps
        flux_out     = phonon.normalise_to_density(flux_out)                  # eV angstrom/ps

        print('e out shape', energies_out.shape)
        
        indexes_in  = self.res_facet_id == facet                # particles coming from that facet
        
        total_flux  = self.bound_values[facet]                  # imposed flux in W/m^2

        flux_in = (total_flux + flux_out)/self.eVpsa2_in_Wm2    # flux comming in, in eV/ps angs^2

        T_new = 300
        T_old = 10 # initial temperature
        
                
        # K1 = self.hbar/(flux_in*phonon.volume_unitcell)
        # K2 = (self.res_omega[indexes_in].reshape(-1, 1)*self.res_group_vel[indexes_in, :]*normal).sum(axis = 1)
        # K3 = np.exp(self.hbar*self.res_omega[indexes_in])

        K1 = self.hbar/self.kb
        K2 = self.res_omega[indexes_in]*(self.res_group_vel[indexes_in, :]*normal).sum(axis = 1) - flux_in
        K3 = (self.res_group_vel[indexes_in, :]*normal).sum(axis = 1)*np.exp(self.res_omega[indexes_in]*K1)

        print('Calculating flux reservoir temperature...')
        print('Flux out', flux_out)
        print('Flux in', flux_in)

        while np.abs((T_new-T_old)/T_old) > 1e-6:

            T_old = T_new

            occupation = phonon.calculate_occupation(T_old, self.res_omega[indexes_in], reference = False)

            dT = K1*((K2*occupation).sum()/((K3*np.log(occupation)).sum()*np.exp(-T_old)))

            T_new = T_old - dT
            
            print(T_old, T_new, dT)
        
        return T_new

    def add_reservoir_particles(self, geometry):
        '''Add the particles that came from the reservoir to the main population. Calculates flux balance for each reservoir.'''

        if self.res_modes.shape[0]>0:
            (self.res_n_timesteps        ,
             self.res_collision_facets   ,
             self.res_collision_positions) = self.timesteps_to_boundary(self.res_positions,
                                                                        self.res_group_vel,
                                                                        geometry          )
            
            self.res_n_timesteps -= self.res_dt_in/self.dt
            self.res_positions   += self.res_group_vel*self.res_dt_in.reshape(-1, 1)

            self.res_collision_cond = self.get_collision_condition(self.res_collision_facets)
            
            self.positions           = np.vstack((self.positions  , self.res_positions  ))
            self.modes               = np.vstack((self.modes      , self.res_modes      ))
            self.group_vel           = np.vstack((self.group_vel  , self.res_group_vel  ))
            self.wavevectors         = np.vstack((self.wavevectors, self.res_wavevectors))
            self.momentum            = np.vstack((self.momentum   , self.res_momentum   ))

            self.n_timesteps         = np.concatenate((self.n_timesteps        , self.res_n_timesteps        ))
            self.collision_facets    = np.concatenate((self.collision_facets   , self.res_collision_facets   ))
            self.collision_positions = np.concatenate((self.collision_positions, self.res_collision_positions))
            self.collision_cond      = np.concatenate((self.collision_cond     , self.res_collision_cond     ))
            self.temperatures        = np.concatenate((self.temperatures       , self.res_temperatures       ))
            self.omega               = np.concatenate((self.omega              , self.res_omega              ))
            self.occupation          = np.concatenate((self.occupation         , self.res_occupation         ))
            self.energies            = np.concatenate((self.energies           , self.res_energies           ))

    def assign_properties(self, modes, phonon):
        '''Get properties from the indexes.'''

        omega      = phonon.omega[ modes[:,0], modes[:,1] ]        # THz * rad
        group_vel  = phonon.group_vel[ modes[:,0], modes[:,1], : ] # THz * angstrom
        wavevector = phonon.wavevectors[modes[:, 0], :]

        return omega, group_vel, wavevector

    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Constant at T_cold unless specified otherwise.'''

        number_of_particles = positions.shape[0]
        key = self.T_distribution

        if key == 'custom':
            subvol_temperatures = np.array(self.args.subvol_temp)
            temperatures = (self.subvol_id*subvol_temperatures).sum(axis = 1)/self.subvol_id.sum(axis = 1)
        
        else:
            bound_T = self.res_bound_values[self.res_bound_cond == 'T'] # np.array([self.bound_values[i] for i in range(len(self.bound_values)) if self.bound_cond[i] == 'T'])
            
            if len(bound_T) == 0:
                bound_T = np.array([self.T_reference])

            temperatures = np.zeros(number_of_particles)    # initialise temperature array

            if key == 'linear':
                # calculates T at the center for each subvolume

                res_facet_index = self.res_facet[self.res_bound_cond == 'T']
                
                bound_positions = geometry.facet_centroid[res_facet_index, :] #np.array([geometry.res_centroid[i, :] for i in range(len(self.bound_values)) if self.bound_cond[i] == 'T'])

                if len(bound_T) > 2:
                    subvol_temperatures = LinearNDInterpolator(bound_positions, bound_T, fill = bound_T.mean())(geometry.subvol_center)
                elif len(bound_T) == 1:
                    subvol_temperatures = np.ones(self.n_of_subvols)*bound_T
                elif len(bound_T) == 2:
                    direction = bound_positions[1, :]-bound_positions[0, :]
                    K = ((geometry.subvol_center-bound_positions[0, :])*direction).sum(axis = 1)
                    alphas = K/(direction**2).sum()

                    subvol_temperatures = bound_T[0]+alphas*(bound_T[1]-bound_T[0])
                    temperatures = subvol_temperatures[self.subvol_id]

            elif key == 'random':
                temperatures        = np.random.rand(number_of_particles)*(bound_T.ptp() ) + bound_T.min()
            elif key == 'constant_hot':
                temperatures        = np.ones(number_of_particles)*bound_T.max()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.max()
            elif key == 'constant_cold':
                temperatures        = np.ones(number_of_particles)*bound_T.min()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.min()
            elif key == 'constant_mean':
                temperatures        = np.ones(number_of_particles)*bound_T.mean()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.mean()
        
        return temperatures, subvol_temperatures
    
    def get_collision_condition(self, collision_facets):
        
        collision_cond = np.empty(collision_facets.shape, dtype = str) # initialise as an empty string array
        
        collision_cond[ np.isnan(collision_facets)] = 'N'              # identify all nan facets with 'N'

        non_nan_facets = collision_facets[~np.isnan(collision_facets)].astype(int) # get non-nan facets
        
        collision_cond[~np.isnan(collision_facets)] = self.bound_cond[non_nan_facets] # save their condition

        return collision_cond

    def get_subvol_id(self, geometry):
        
        scaled_positions = geometry.scale_positions(self.positions) # scale positions between 0 and 1

        subvol_id = geometry.subvol_classifier.predict(scaled_positions) # get subvol_id from the model

        subvol_id = np.argmax(subvol_id, axis = 1)

        self.subvol_N_p = np.zeros(self.n_of_subvols)

        for sv in range(self.n_of_subvols):
            self.subvol_N_p[sv] = np.sum(subvol_id == sv)

        return subvol_id

    def initialise_occupation_lookup(self, phonon):
        
        T = self.subvol_temperature.reshape(-1, 1, 1)

        occupation = phonon.calculate_occupation(T, phonon.omega, reference = True)

        return occupation
    
    def save_occupation_lookup(self, phonon, damp = 0.5):

        old_occ = copy.copy(self.occupation_lookup)

        for sv in range(self.n_of_subvols):

            i = np.nonzero(self.subvol_id == sv)[0] # get with particles in the subvolume
            data = self.modes[i, :]                 # and their modes

            u, count = np.unique(data, axis = 0, return_counts = True) # unique modes in sv and where they are located in data
            
            occ_sum = np.zeros(phonon.omega.shape) # initialise the new occupation matrix
            for p in range(len(i)):
                occ_sum[data[p, 0], data[p, 1]] += self.occupation[i[p]] # sum for each present mode
            
            occ_sum[u[:, 0], u[:, 1]] /= count # average them

            occ_sum = np.where(occ_sum == 0, old_occ[sv, :, :], occ_sum)

            # self.occupation_lookup[sv, u[:, 0], u[:, 1]] = self.occupation_lookup[sv, u[:, 0], u[:, 1]]*damp + occ_sum[u[:, 0], u[:, 1]]*(1-damp)
            self.occupation_lookup[sv, :, :] = self.occupation_lookup[sv, :, :]*damp + occ_sum*(1-damp)

        diff = np.absolute(self.occupation_lookup - old_occ)

        self.residue = np.linalg.norm(diff, axis = (1, 2))
        
    def refresh_temperatures(self, geometry, phonon):
        '''Refresh energies and temperatures while enforcing boundary conditions as given by geometry.'''
        self.energies = self.occupation*self.omega*self.hbar    # eV
        
        self.subvol_id = self.get_subvol_id(geometry)

        self.calculate_energy(geometry, phonon, lookup = self.lookup)

        self.subvol_temperature = phonon.temperature_function(self.subvol_energy)

        self.temperatures = self.subvol_temperature[self.subvol_id]

    def calculate_energy(self, geometry, phonon, lookup = False):

        if lookup:
            self.save_occupation_lookup(phonon, damp = self.damping)
            self.subvol_energy = np.sum(self.occupation_lookup*phonon.omega*self.hbar, axis = (1, 2))
        else:
            self.subvol_energy = np.zeros(self.n_of_subvols)
            for sv in range(self.n_of_subvols):
                i = np.nonzero(self.subvol_id == sv)[0]
                self.subvol_energy[sv] = self.energies[i].sum()

            if self.norm == 'fixed':
                normalisation = phonon.number_of_active_modes/(self.particle_density*geometry.subvol_volume)
            elif self.norm == 'mean':
                normalisation = phonon.number_of_active_modes/self.subvol_N_p
                normalisation = np.where(np.isnan(normalisation), 0, normalisation)
            self.subvol_energy = self.subvol_energy*normalisation
        
        self.subvol_energy = phonon.normalise_to_density(self.subvol_energy)

        self.subvol_energy += phonon.reference_energy

    def calculate_heat_flux(self, geometry, phonon, lookup = False):
        
        if lookup:
            energy = self.occupation_lookup*self.hbar*phonon.omega # (SV, Q, J)
            heat_flux = phonon.group_vel*energy.reshape(energy.shape[0], energy.shape[1], energy.shape[2], 1) # (SV, Q, J, 3)
            heat_flux = np.sum(heat_flux, axis = (1, 2)) # SV, 3

        else:
            heat_flux = np.zeros((self.n_of_subvols, 3))

            for i in range(self.n_of_subvols):
                ind = np.where(self.subvol_id == i)[0] # 1d subvol id
                heat_flux[i, :] = np.sum(self.group_vel[ind, :]*self.energies[ind].reshape(-1, 1), axis = 0)

            if self.norm == 'fixed':
                normalisation = phonon.number_of_active_modes/(self.particle_density*geometry.subvol_volume.reshape(-1, 1))
            elif self.norm == 'mean':
                normalisation = phonon.number_of_active_modes/self.subvol_N_p.reshape(-1, 1)
            
            heat_flux = heat_flux*normalisation
        
        heat_flux = phonon.normalise_to_density(heat_flux)

        self.subvol_heat_flux = heat_flux*self.eVpsa2_in_Wm2

    def calculate_momentum(self, geometry, phonon, lookup = False):
        
        if lookup:
            n = self.occupation_lookup.reshape(self.occupation_lookup.shape[0], self.occupation_lookup.shape[1], self.occupation_lookup.shape[2], 1) # (SV, Q, J, 1)
            k = phonon.wavevectors.reshape(-1, 1, 3) # (Q, 1, 3)
            momentum = n*self.hbar*k # (SV, Q, J, 3)
            momentum = np.sum(momentum, axis = (1, 2)) # (SV, 3)
        else:
            momentum = np.zeros((self.n_of_subvols, 3))
            for i in range(self.n_of_subvols):
                ind = np.where(self.subvol_id == i)[0] # 1d subvol id
                momentum[i, :] = np.sum(self.hbar*self.wavevectors[ind, :]*self.occupation[ind].reshape(-1, 1), axis = 0)

            if self.norm == 'fixed':
                normalisation = phonon.number_of_active_modes/(self.particle_density*geometry.subvol_volume.reshape(-1, 1))
            elif self.norm == 'mean':
                normalisation = phonon.number_of_active_modes/self.subvol_N_p.reshape(-1, 1)
            
            momentum = momentum*normalisation

        momentum = phonon.normalise_to_density(momentum)

        momentum[:, 0] += phonon.reference_momentum_function_x(self.subvol_temperature)
        momentum[:, 1] += phonon.reference_momentum_function_y(self.subvol_temperature)
        momentum[:, 2] += phonon.reference_momentum_function_z(self.subvol_temperature)

        self.subvol_momentum = momentum*self.eVpsa_in_kgms
        
    def drift(self):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt # move forward by one v*dt step

        self.n_timesteps -= 1                    # -1 timestep in the counter to boundary scattering

    def find_boundary(self, positions, velocities, geometry):
        '''Finds which mesh triangle will be hit by the particle, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        # get where the particle will hit a wall, and which face (triangle)
        index_faces, index_ray, boundary_pos = geometry.mesh.ray.intersects_id(ray_origins      = positions ,
                                                                               ray_directions   = velocities,
                                                                               return_locations = True      ,
                                                                               multiple_hits    = False     )
        
        with_ray   = np.in1d(np.arange(positions.shape[0]), index_ray) # find which particles have rays
        stationary = (np.linalg.norm(velocities, axis = 1) == 0)       # find which particles have zero velocity

        all_boundary_pos                = np.zeros( (positions.shape[0], 3) )
        all_boundary_pos[stationary, :] = np.inf                       # those stationary have infinite collision position
        all_boundary_pos[with_ray  , :] = boundary_pos                 # those with ray have correspondent collision position

        all_faces = np.ones(with_ray.shape[0])*np.nan
        all_faces[with_ray] = index_faces

        nan_particles = np.isnan(all_faces)

        ################################# DEBUGGING ###################################
        # get which facet those faces are part of
        nan_faces    = np.isnan(all_faces)
        if nan_faces.any():
            sd = tm.proximity.signed_distance(geometry.mesh, positions)
            for i in range(nan_faces.shape[0]):
                if nan_faces[i]:
                    print(positions[i, :], velocities[i, :], sd[i])
            
            fig_nan, ax_nan = plt.subplots(nrows = 1, ncols = 1,
                                           subplot_kw={'projection': '3d'},
                                           figsize = (20, 20)             ,
                                           dpi = 100                      )
            ax_nan.scatter(positions[~nan_faces, 0],
                           positions[~nan_faces, 1],
                           positions[~nan_faces, 2], '.', s = 1, c = 'b', alpha = 0.2)
            ax_nan.scatter(positions[nan_faces, 0],
                           positions[nan_faces, 1],
                           positions[nan_faces, 2], '.', s = 1, c = 'r')

            if positions.shape[0]>1:
                nan_event = 0
            else:
                nan_event += 1

            plt.savefig(self.results_folder_name + 'nan{:d}.png'.format(nan_event))
            plt.close(fig_nan.number)
            # plt.show(block = True)
        #############################################################################

        index_facets = geometry.faces_to_facets(all_faces[~nan_faces])

        all_facets           = np.ones( positions.shape[0] )*np.nan  # all particles that do not collide do not have corresponding facets
        all_facets[~nan_particles] = index_facets

        # if self.current_timestep > 0:
        #     nan_particles[:]= True # FOR DEBUGGING

        if np.any(nan_particles):
            (all_boundary_pos[nan_particles, :],
             all_facets[nan_particles]         ) = self.find_collision_naive(positions[nan_particles, :] ,
                                                                             velocities[nan_particles, :],
                                                                             geometry                    )

        return all_boundary_pos, all_facets

    def find_collision_naive(self, x, v, geo):
        '''To be used when trimesh fails to find it with ray casting.
           It uses vector algebra to find the collision position in every facet,
           and pick the one that hits first between its vertices.'''
        # print('Naive!')

        n = -geo.mesh.facets_normal # normals - (F, 3)
        k = geo.facet_k

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            #          (N, F)                                    (F,)            (N,F)
            t = -(np.sum(x.reshape(-1, 1, 3)*n, axis = 2)+k)/np.sum(v.reshape(-1, 1, 3)*n, axis = 2)
        
        t_valid = t >= 0

        t = np.where(t_valid, t, np.inf) # update t - (N, F)
        
        xc = np.ones(x.shape)*np.nan    # (N, 3)
        fc = np.ones(x.shape[0])*np.nan # (N,)

        active = np.isnan(fc) # generate a follow up boolean mask - (N,)

        while np.any(active):
            t_min = np.amin(t[active, :], axis = 1, keepdims = True) # get minimum t

            cand_f = np.argmax(t[active, :] == t_min, axis = 1)      # which facet
            
            cand_xc = x[active, :]+t_min*v[active, :] # candidate collision positions based on t_min

            for f in range(len(geo.mesh.facets)): # for each facet
                f_particles = cand_f == f       # particles that may hit that facet
                if np.any(f_particles):
                    faces = geo.mesh.facets[f] # indexes of faces part of that facet
                    
                    n_fp = f_particles.sum()

                    in_facet = np.zeros(n_fp).astype(bool)
                    
                    for face in faces: # for each face in the facet
                        
                        tri = geo.mesh.vertices[geo.mesh.faces[face], :] # vertces of the face

                        tri = np.ones((n_fp, 3, 3))*tri # reshaping for barycentric
                        bar = tm.triangles.points_to_barycentric(tri, cand_xc[f_particles, :])

                        # check whether the points are between the vertices of the face
                        valid = np.all(np.logical_and(bar >= 0, bar <= 1), axis = 1)

                        in_facet[valid] = True # those valid are in facet
                    
                    indices = np.where(f_particles)[0][in_facet] # indices of the particles analysed for that facet that are really hitting

                    if len(indices) > 0:
                        xc[indices, :] = cand_xc[indices, :]
                        fc[indices] = f

            t_valid[active, cand_f] = False

            active = np.isnan(fc)

            t_valid[~active, :] = False
            t = np.where(t_valid, t, np.inf) # update t - (N, F)
        
        return xc, fc

    def timesteps_to_boundary(self, positions, velocities, geometry):
        '''Calculate how many timesteps to boundary scattering.'''

        if positions.shape[0] == 0:
            ts_to_boundary      = np.zeros(0)
            index_facets        = np.zeros(0)
            boundary_pos        = np.zeros((0, 3))
            
            return ts_to_boundary, index_facets, boundary_pos

        boundary_pos_t, index_facets = self.find_boundary(positions, velocities, geometry) # find collision in true boundary

        while np.any(np.isnan(index_facets)):
            nan_particles = np.isnan(index_facets)    # bool
            n_nan = np.sum(nan_particles.astype(int)) # int
            print('ts =', self.current_timestep, 'n_nan =', n_nan)
            positions[nan_particles, :] += (np.random.rand(n_nan, 3)-0.5)*2*1e-4# tm.tol.merge # nudge them
            boundary_pos_t[nan_particles, :], index_facets[nan_particles] = self.find_boundary(positions[nan_particles, :], velocities[nan_particles, :], geometry)

        normals = -geometry.mesh.facets_normal[index_facets.astype(int), :] # get the normals of the collision facets
        v_norm  = np.linalg.norm(velocities, axis = 1, keepdims = True)     # get the velocity norm

        offset_dist_norm = - v_norm*self.offset/np.sum(velocities*normals, axis = 1, keepdims = True) # ||d|| = ||v|| h / v . n
        offset_dist      = - velocities*offset_dist_norm/v_norm                                       # d = ||d|| v/||v||

        boundary_pos_p = boundary_pos_t + offset_dist # the corrected collision point

        # calculate distances for given particles
        boundary_dist = np.linalg.norm(boundary_pos_p - positions, axis = 1 ) # distance travelled up to collision

        ts_to_boundary = boundary_dist/( np.linalg.norm(velocities, axis = 1) * self.dt )  # such that particle hits the boundary when n_timesteps == 0 (crosses boundary)

        return ts_to_boundary, index_facets, boundary_pos_p
    
    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes'''

        self.positions           = np.delete(self.positions          , indexes, axis = 0)
        self.group_vel           = np.delete(self.group_vel          , indexes, axis = 0)
        self.wavevectors         = np.delete(self.wavevectors        , indexes, axis = 0)
        self.momentum            = np.delete(self.momentum           , indexes, axis = 0)
        self.omega               = np.delete(self.omega              , indexes, axis = 0)
        self.occupation          = np.delete(self.occupation         , indexes, axis = 0)
        self.energies            = np.delete(self.energies           , indexes, axis = 0)
        self.temperatures        = np.delete(self.temperatures       , indexes, axis = 0)
        self.n_timesteps         = np.delete(self.n_timesteps        , indexes, axis = 0)
        self.modes               = np.delete(self.modes              , indexes, axis = 0)
        self.collision_facets    = np.delete(self.collision_facets   , indexes, axis = 0)
        self.collision_positions = np.delete(self.collision_positions, indexes, axis = 0)
        self.collision_cond      = np.delete(self.collision_cond     , indexes, axis = 0)
    
    def calculate_specularity(self, facets, modes, geometry, phonon, return_cos = False):

        normals = -geometry.mesh.facets_normal[facets.astype(int)] # get their normals (outwards, theta < pi/2)

        q = phonon.q_points[modes[:, 0], :]
        q = np.where(q >   0.5, q - 1, q)
        q = np.where(q <= -0.5, q + 1, q)

        k = phonon.q_to_k(q)

        k_norm = np.sum(k**2, axis = 1)**0.5

        facets_indexes = np.array([np.where(f == self.rough_facets)[0][0] for f in facets])

        eta = self.rough_facets_values[facets_indexes, 0] # getting roughness

        v = phonon.group_vel[modes[:, 0], modes[:, 1], :]
        v_norm = np.sum(v**2, axis = -1)**0.5

        dot = np.sum(v*normals, axis = -1)
        cos_theta = dot/v_norm

        specularity = np.exp(-(2*eta*k_norm*cos_theta)**2)

        if return_cos:
            return specularity, cos_theta
        else:
            return specularity
    
    def get_random_mode(self, r, roulette):
        return np.argmax(r <= roulette)

    def select_reflected_modes(self, incident_modes, collision_facets, geometry, phonon):

        collision_facets = collision_facets.astype(int)

        n_g = -geometry.mesh.facets_normal[collision_facets, :] # get inward normals
        
        n_p = incident_modes.shape[0] # number of particles being reflected

        k = phonon.wavevectors[incident_modes[:, 0], :]  # get wavevectors

        # initialise new modes
        new_modes = np.zeros((n_p, 2), dtype = int)

        # see which particles are reflecting specularly            
        p = self.calculate_specularity(collision_facets, incident_modes, geometry, phonon)
        r = np.random.rand(n_p)
        indexes_spec = r <= p
        
        if np.any(indexes_spec): # specular reflection

            # reflect
            k_try = k[indexes_spec, :] - 2*n_g[indexes_spec, :]*(k[indexes_spec, :]*n_g[indexes_spec, :]).sum(axis = 1).reshape(-1, 1)
            
            # convert to q
            q_try = phonon.k_to_q(k_try)

            # get unique specular normals (so it is not calculated several times for the same facet)
            unique_facets, i_facet = np.unique(collision_facets[indexes_spec].astype(int), return_inverse = True)
            unique_n = -geometry.mesh.facets_normal[unique_facets, :]

            # sign of reflected waves
            v_all = phonon.group_vel # (Q, J, 3)
            v_dot_n = v_all.reshape(v_all.shape[0], v_all.shape[1], 1, v_all.shape[2])*unique_n # dot product V x n (Q, J, F_u, 3)

            v_dot_n = v_dot_n.sum(axis = -1) # (Q, J, F_u)
            
            sign = np.around(v_dot_n, decimals = 3) > 0    # where V agrees with normal (Q, J, F_u)

            for i_f in range(unique_facets.shape[0]): # for each facet
                
                facet = unique_facets[i_f]                      # get concerned facet
                i_p = collision_facets[indexes_spec] == facet   # get indexes of concerned particles

                modes = incident_modes[indexes_spec, :][i_p, :] # and their modes

                s = np.any(sign[:, :, i_f], axis = 1)           # get the mask of possible modes
                
                q_psbl = phonon.q_points[s, :]                  # get possible qpoints
                y = np.where(s)[0]                              # get their numbers
                
                f = NearestNDInterpolator(q_psbl, y)            # generate the interpolator
                new_qpoints = f(q_try[i_p, :]).astype(int)      # get new qpoints

                # calculate difference and make invalid branches to go to infinity
                omega_diff = np.absolute(phonon.omega[modes[:, 0], modes[:, 1]].reshape(-1, 1) -
                                         phonon.omega[new_qpoints, :          ])
                                        
                omega_diff = np.where(sign[new_qpoints, :, i_f], omega_diff, np.inf)

                branch_mask = (omega_diff == omega_diff.min(axis = 1).reshape(-1, 1)) # gets where are the closest valid branches

                new_branches = np.argmax(branch_mask, axis = 1) # save new branches

                i_modes = np.arange(new_modes.shape[0])[indexes_spec][i_p]

                new_modes[i_modes, 0] = copy.copy(new_qpoints)
                new_modes[i_modes, 1] = copy.copy(new_branches)
        
        # diffuse reflection
        indexes_diff = ~indexes_spec  # those which are not specularly reflected, are diffusely reflected

        # DAVIER'S FORMULATION ##########################################

        if np.any(indexes_diff):
            
            unique_facets, i_facet = np.unique(collision_facets[indexes_diff].astype(int), return_inverse = True)
            unique_n = -geometry.mesh.facets_normal[unique_facets, :]

            # sign of reflected waves
            v_all = phonon.group_vel # (Q, J, 3)
            v_dot_n = v_all.reshape(v_all.shape[0], v_all.shape[1], 1, v_all.shape[2])*unique_n # dot product V x n (Q, J, F_u, 3)

            v_dot_n = v_dot_n.sum(axis = -1) # (Q, J, F_u)
            
            # sign = np.around(v_dot_n, decimals = 3) > 0    # where V agrees with normal (Q, J, F_u)
            sign = v_dot_n > 0    # where V agrees with normal (Q, J, F_u)

            for i_f in range(unique_facets.shape[0]):
                
                # start = time.time()
                facet = unique_facets[i_f]                      # get concerned facet
                i_p = collision_facets[indexes_diff] == facet   # get indexes of concerned particles

                n_p = int(i_p.sum())                     # number of concerned particles

                modes = incident_modes[indexes_diff, :][i_p, :] # and their modes

                s = sign[:, :, i_f]           # get the mask of possible modes

                p_out, cos_theta_long = self.calculate_specularity(np.array([facet], dtype=int), phonon.unique_modes, geometry, phonon, return_cos = True) # specularity of every outward mode

                cos_theta = np.zeros(phonon.omega.shape)
                cos_theta[phonon.unique_modes[:, 0], phonon.unique_modes[:, 1]] = cos_theta_long
                
                prob = np.zeros(phonon.omega.shape)
                prob[phonon.unique_modes[:, 0], phonon.unique_modes[:, 1]] = (1 - p_out)

                prob = np.where(s, prob, 0)

                prob = prob*cos_theta*(np.sum(phonon.group_vel**2, axis = -1)**0.5) # calculate probability with lamberts law
                prob[np.isnan(prob)] = 0
                prob[np.isinf(prob)] = 0

                roulette = np.cumsum(prob)  # build roulette

                r = np.random.rand(n_p)*roulette.max() # draw random number

                func = partial(self.get_random_mode, roulette = roulette)

                mapped_modes = map(func, r)

                flat_i = np.fromiter(mapped_modes, dtype = int) # flatted array indexes

                new_qpoints  = np.floor(flat_i/phonon.number_of_branches).astype(int)
                new_branches = flat_i - new_qpoints*phonon.number_of_branches

                i_modes = np.arange(new_modes.shape[0])[indexes_diff][i_p]

                new_modes[i_modes, 0] = np.copy(new_qpoints)
                new_modes[i_modes, 1] = np.copy(new_branches)

        new_modes = new_modes.astype(int)

        f = open(self.results_folder_name+'scattering_diff.txt', 'a')
        for i in range(new_modes.shape[0]):
            if indexes_diff[i]:
                f.write('{:d}, {:d}, {:d}, {:d}, {:.1f}, {:.1f}, {:.1f} \n'.format(incident_modes[i, 0], incident_modes[i, 1],
                                                                        new_modes[i, 0]     , new_modes[i, 1]     ,
                                                                        n_g[i, 0], n_g[i, 1], n_g[i, 2]))
        f.close()

        return new_modes, indexes_spec

    def periodic_boundary_condition(self, positions, group_velocities, collision_facets, collision_positions, calculated_ts, geometry):
        
        collision_facets = collision_facets.astype(int) # ensuring collision facets are integers

        # getting which face is connected
        rows             = np.array([np.where(i == self.connected_facets)[0][0] for i in collision_facets])  
        mask             = self.connected_facets[rows, :] == collision_facets.reshape(-1, 1)
        connected_facets = (self.connected_facets[rows, :]*~mask).sum(axis = 1).astype(int)

        previous_positions        = copy.deepcopy(positions)               # the path of some particles is calculated from their current position
        first                     = np.nonzero(calculated_ts == 0)[0]      # but for those at the beginning of the timestep
        previous_positions[first, :] -= group_velocities[first, :]*self.dt # their starting position is the one before the collision

        L = (geometry.facet_centroid[connected_facets, :] - geometry.facet_centroid[collision_facets, :]) # get translation vector between connected facets
        normals = -geometry.mesh.facets_normal[connected_facets, :]                                       # get the normals of the new facets
        alpha = 1+2*self.offset/np.sum(L*normals, axis = 1, keepdims = True)                              # correct translation considering offsets
        new_positions = collision_positions + L*alpha                                                     # translate particles

        # finding next scattering event
        new_ts_to_boundary, new_collision_facets, new_collision_positions = self.timesteps_to_boundary(new_positions, group_velocities, geometry)

        calculated_ts += np.linalg.norm((collision_positions - previous_positions), axis = 1)/np.linalg.norm((group_velocities*self.dt), axis = 1)

        new_collision_facets = new_collision_facets.astype(int)
            
        new_collision_cond = self.bound_cond[new_collision_facets]

        return new_positions, new_ts_to_boundary, new_collision_facets, new_collision_positions, new_collision_cond, calculated_ts

    def roughness_boundary_condition(self, positions, group_velocities, incident_modes, occupation, collision_facets, collision_positions, calculated_ts, geometry, phonon, temperatures):

        # particles already scattered this timestep are calculated from their current position (at a boundary)
        previous_positions = copy.deepcopy(positions)

        # first scattering particles start from their position at the beginning of the timestep
        first = calculated_ts == 0
        previous_positions[first, :] -= group_velocities[first, :]*self.dt

        # the calculated timestep is up to the next scattering event
        dist = np.linalg.norm(collision_positions - previous_positions, axis = 1)
        vel = np.linalg.norm(group_velocities, axis = 1)
        
        new_calculated_ts = calculated_ts + dist/(vel*self.dt)

        # update particle positions to the collision posiiton
        new_positions = collision_positions
        
        # select new modes and get their properties
        new_modes, indexes_spec = self.select_reflected_modes(incident_modes, collision_facets, geometry, phonon)
        
        indexes_diff = ~indexes_spec

        # update v
        new_group_vel  = phonon.group_vel[new_modes[:, 0], new_modes[:, 1], :]

        # update k
        new_wavevectors = phonon.wavevectors[new_modes[:, 0], :]
        
        # update omega
        new_omega      = phonon.omega[     new_modes[:, 0],      new_modes[:, 1]]

        # update occupation
        new_occupation = copy.copy(occupation)
        new_occupation[indexes_diff] = phonon.calculate_occupation(temperatures[indexes_diff], new_omega[indexes_diff], reference = True) #

        # find next scattering event
        new_ts_to_boundary, new_collision_facets, new_collision_positions = self.timesteps_to_boundary(new_positions, new_group_vel, geometry)
        new_collision_cond = self.bound_cond[new_collision_facets.astype(int)]

        ############## SAVING SCATTERING DATA ##################
        f = open(self.results_folder_name+'scattering_spec.txt', 'a')
        for i in range(new_modes.shape[0]):
            if indexes_spec[i]:
                n_g = -geometry.mesh.facets_normal[int(collision_facets[i]), :]
                f.write('{:d}, {:d}, {:d}, {:d}, {:.1f}, {:.1f}, {:.1f} \n'.format(incident_modes[i, 0], incident_modes[i, 1],
                                                                                   new_modes[i, 0]     , new_modes[i, 1]     ,
                                                                                   n_g[0], n_g[1], n_g[2]))
        f.close()
        ###############################
        
        return new_modes, new_positions, new_group_vel, new_wavevectors, new_omega, new_occupation, new_ts_to_boundary, new_calculated_ts, new_collision_positions, new_collision_facets, new_collision_cond

    def boundary_scattering(self, geometry, phonon):
        '''Applies boundary scattering or other conditions to the particles where it happened, given their indexes.'''

        indexes_all = self.n_timesteps < 0                      # find scattering particles (N_p,)

        calculated_ts              = np.ones(indexes_all.shape) # start the tracking of calculation as 1 (N_p,)
        calculated_ts[indexes_all] = 0                          # set it to 0 for scattering particles   (N_p,)

        new_n_timesteps = copy.copy(self.n_timesteps) # (N_p,)

        # while there are any particles with the timestep not completely calculated
        while np.any(calculated_ts < 1):

            # I. Deleting particles entering reservoirs

            # identifying particles hitting rough facets
            indexes_del = np.logical_and(calculated_ts       <   1 ,
                                         np.in1d(self.collision_cond, ['T', 'F']))
            indexes_del = np.logical_and(indexes_del, 
                                         (1-calculated_ts) > new_n_timesteps)

            if np.any(indexes_del):
                if self.n_of_reservoirs > 0: # if there are any reservoirs
                    for i in range(self.n_of_reservoirs): # for each one

                        facet = self.res_facet[i]         # get their facet index

                        indexes_res = self.collision_facets[indexes_del] == facet # get which particles are leaving through it

                        # subtracting energy
                        energies = self.energies[indexes_del][indexes_res]
                        self.res_energy_balance[i] -= energies.sum()

                        # adding heat flux
                        hflux = energies.reshape(-1, 1)*self.group_vel[indexes_del, :][indexes_res, :]
                        self.res_heat_flux[i, :] += hflux.sum(axis = 0)

                        # adding momentum
                        momentum = self.momentum[indexes_del, :][indexes_res, :]
                        self.res_momentum_balance[i, :] += momentum.sum(axis = 0)

                self.delete_particles(indexes_del)
                calculated_ts   = calculated_ts[~indexes_del]
                new_n_timesteps = new_n_timesteps[~indexes_del]
                indexes_all     = indexes_all[~indexes_del]
            
            # II. Applying Periodicities:
            
            # identifying particles hitting facets with periodic boundary condition
            indexes_per = np.logical_and(calculated_ts       <   1 , # 
                                        self.collision_cond == 'P')
            indexes_per = np.logical_and(indexes_per               ,
                                         (1-calculated_ts) > new_n_timesteps)
            
            if np.any(indexes_per):

                # enforcing periodic boundary condition
                (self.positions[indexes_per, :]         ,
                new_n_timesteps[indexes_per]            ,
                self.collision_facets[indexes_per]      ,
                self.collision_positions[indexes_per, :],
                self.collision_cond[indexes_per]        ,
                calculated_ts[indexes_per]              ) = self.periodic_boundary_condition(self.positions[indexes_per, :]          ,
                                                                                             self.group_vel[indexes_per, :]          ,
                                                                                             self.collision_facets[indexes_per]      ,
                                                                                             self.collision_positions[indexes_per, :],
                                                                                             calculated_ts[indexes_per]              ,
                                                                                             geometry                                )

            # III. Performing scattering:

            # identifying particles hitting rough facets
            indexes_ref = np.logical_and(calculated_ts       <   1 ,
                                         self.collision_cond == 'R')
            indexes_ref = np.logical_and(indexes_ref               ,
                                         (1-calculated_ts) > new_n_timesteps)
            
            if np.any(indexes_ref):
                (self.modes[indexes_ref, :]              ,
                 self.positions[indexes_ref, :]          ,
                 self.group_vel[indexes_ref, :]          ,
                 self.wavevectors[indexes_ref, :]        ,
                 self.omega[indexes_ref]                 ,
                 self.occupation[indexes_ref]            ,
                 new_n_timesteps[indexes_ref]            ,
                 calculated_ts[indexes_ref]              ,
                 self.collision_positions[indexes_ref, :],
                 self.collision_facets[indexes_ref]      ,
                 self.collision_cond[indexes_ref]        ) = self.roughness_boundary_condition(self.positions[indexes_ref, :]          ,
                                                                                               self.group_vel[indexes_ref, :]          ,
                                                                                               self.modes[indexes_ref, :]              ,
                                                                                               self.occupation[indexes_ref]            ,
                                                                                               self.collision_facets[indexes_ref]      ,
                                                                                               self.collision_positions[indexes_ref, :],
                                                                                               calculated_ts[indexes_ref]              ,
                                                                                               geometry                                ,
                                                                                               phonon                                  ,
                                                                                               self.temperatures[indexes_ref])

            # IV. Drifting those who will not be scattered again in this timestep

            # identifying drifting particles
            indexes_drift = np.logical_and(calculated_ts < 1                  ,
                                           (1-calculated_ts) < new_n_timesteps)

            if np.any(indexes_drift):
                self.positions[indexes_drift, :] += self.group_vel[indexes_drift, :]*self.dt*(1-calculated_ts[indexes_drift]).reshape(-1, 1)

                new_n_timesteps[indexes_drift]   -= (1-calculated_ts[indexes_drift])
                calculated_ts[indexes_drift]     = 1
            
        self.n_timesteps = copy.copy(new_n_timesteps)

        if self.n_of_reservoirs > 0:
            
            self.res_energy_balance   = phonon.normalise_to_density(self.res_energy_balance)*self.ev_in_J
            self.res_heat_flux        = phonon.normalise_to_density(self.res_heat_flux)*self.eVpsa2_in_Wm2
            self.res_momentum_balance = phonon.normalise_to_density(self.res_momentum_balance)*self.eVpsa_in_kgms

    def lifetime_scattering(self, phonon):
        '''Performs lifetime scattering.'''

        # N_as = N_ad + dt/tau (N_BE(T*) - N_ad)
        
        Tqj = np.hstack((self.temperatures.reshape(-1, 1), self.modes))
        tau = phonon.lifetime_function(Tqj)

        occupation_ad = copy.deepcopy(self.occupation)             # getting current occupation
        occupation_BE = phonon.calculate_occupation(self.temperatures, self.omega, reference = True) # calculating Bose-Einstein occcupation

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            occupation_as = np.where(tau>0, occupation_ad + (self.dt/tau) *(occupation_BE - occupation_ad), occupation_ad)
            self.occupation = occupation_as
    
    def run_timestep(self, geometry, phonon):

        if  ( self.current_timestep % 100) == 0:
            self.write_final_state(geometry, phonon)

        self.drift()                                    # drift particles

        if self.n_of_reservoirs > 0:
            self.fill_reservoirs(geometry, phonon)          # refill reservoirs
            
            self.add_reservoir_particles(geometry)  # add reservoir particles that come in the domain
            
        self.boundary_scattering(geometry, phonon)      # perform boundary scattering/periodicity and particle deletion

        self.refresh_temperatures(geometry, phonon)     # refresh cell temperatures

        self.lifetime_scattering(phonon)                # perform lifetime scattering

        self.current_timestep += 1                      # +1 timestep index

        self.t = self.current_timestep*self.dt          # 1 dt passed
        
        if  ( self.current_timestep % 100) == 0:
            info ='Timestep {:>5d}, Residue = {:5.3e} ['.format( int(self.current_timestep), self.residue.max())
            for sv in range(self.n_of_subvols):
                info += ' {:>7.3f}'.format(self.subvol_temperature[sv])
            
            info += ' ]'

            print(info)
        
        if ( self.current_timestep % self.n_dt_to_conv) == 0:
            
            self.save_occupation_lookup(phonon, damp = self.damping)
            self.calculate_heat_flux(geometry, phonon, lookup = self.lookup)
            self.calculate_momentum(geometry, phonon, lookup = self.lookup)

            self.write_convergence()      # write data on file

        if (len(self.rt_plot) > 0) and (self.current_timestep % self.n_dt_to_plot == 0):
            self.plot_real_time(geometry)
        
        gc.collect() # garbage collector

    def init_plot_real_time(self, geometry, phonon):
        '''Initialises the real time plot to be updated for each timestep.'''

        n_dt_to_plot = np.floor( np.log10( self.args.iterations[0] ) ) - 1    # number of timesteps to save new animation frame
        n_dt_to_plot = int(10**n_dt_to_plot)
        n_dt_to_plot = max([1, n_dt_to_plot])

        self.n_dt_to_plot = 1 # n_dt_to_plot

        if self.rt_plot[0] in ['T', 'temperature']:
            colors = self.temperatures
            T_bound = self.res_bound_values[self.res_bound_cond == 'T']
            vmin = np.floor(self.T_bound.min()-10)
            vmax = np.ceil(self.T_bound.max()+10)
        elif self.rt_plot[0] in ['e', 'energy']:
            colors = self.energies
            T_bound = self.res_bound_values[self.res_bound_cond == 'T']
            vmin = phonon.calculate_energy(T_bound.min()-3, phonon.omega, reference = True).min()
            vmax = phonon.calculate_energy(T_bound.max()+3, phonon.omega, reference = True).max()
        elif self.rt_plot[0] in ['omega', 'angular_frequency']:
            colors = self.omega
            vmin = phonon.omega[phonon.omega>0].min()
            vmax = phonon.omega.max()
        elif self.rt_plot[0] in ['n', 'occupation']:
            colors = self.occupation
            order = [np.floor( np.log10( self.occupation.min()) ), np.floor( np.log10( self.occupation.max()) )]
            vmin = (10**order[0])*np.ceil(self.occupation.min()/(10**order[0]))
            vmax = (10**order[1])*np.ceil(self.occupation.max()/(10**order[1]))
        elif self.rt_plot[0] in ['qpoint']:
            colors = self.modes[:, 0]
            vmin = 0
            vmax = phonon.number_of_qpoints
        elif self.rt_plot[0] in ['branch']:
            colors = self.modes[:, 1]
            vmin = 0
            vmax = phonon.number_of_branches
        elif self.rt_plot[0] in ['ts_to_boundary']:
            colors = self.n_timesteps

            vel  = np.sqrt( (phonon.group_vel**2).sum(axis = 2) )
            min_vel = vel[vel>0].min()
            max_path = np.sqrt( (geometry.bounds[1, :]**2).sum() ).min()
            
            vmin = 0
            vmax = 100 # max_path/(min_vel*self.dt)
        elif self.rt_plot[0] in ['subvol']:
            colors = self.subvol_id
            vmin = 0
            vmax = self.n_of_subvols

        plt.ion()
        
        # Think about a way to make the animation "pretty" with a proper aspect ratio depending on dimensions of the domain

        # box_size = np.ptp(geometry.bounds, axis = 0)
        # figsize = np.array([box_size.max()/2, box_size.min()])
        # figsize = figsize*8/(box_size.max()/2)

        figsize = (15,15)

        fig = plt.figure(figsize = figsize, dpi = 100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
        ax.set_xlim(geometry.bounds[:, 0])
        ax.set_ylim(geometry.bounds[:, 1])
        ax.set_zlim(geometry.bounds[:, 2])

        graph = ax.scatter(self.positions[:, 0],
                           self.positions[:, 1],
                           self.positions[:, 2],
                           s    = 1            ,
                           vmin = vmin         ,
                           vmax = vmax         ,
                           c    = colors       ,
                           cmap = self.colormap)
        # fig.colorbar(graph)

        # graph.set_animated(True)

        plt.tight_layout()

        fig.canvas.draw()

        plt.show(block = False)

        self.plot_images = [np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape(fig.canvas.get_width_height()[::-1]+(3,))]

        plt.figure(fig.number)
        plt.savefig(self.results_folder_name+'last_anim_frame.png')

        return graph, fig

    def plot_real_time(self, geometry, property_plot = None, colormap = 'viridis'):
        '''Updates data on the real time plot at the end of the timestep.'''
        if property_plot == None:
            property_plot = self.rt_plot[0]
        
        if colormap == None:
            colormap = self.colormap
        
        if property_plot in ['T', 'temperature']:
            colors = self.temperatures
        elif property_plot in ['e', 'energy']:
            colors = self.energies
        elif property_plot in ['omega', 'angular_frequency']:
            colors = self.omega
        elif property_plot in ['n', 'occupation']:
            colors = self.occupation
        elif property_plot in ['qpoint']:
            colors = self.modes[:, 0]
        elif property_plot in ['branch']:
            colors = self.modes[:, 1]
        elif property_plot in ['ts_to_boundary']:
            colors = self.n_timesteps
        elif self.rt_plot[0] in ['subvol']:
            colors = self.subvol_id
        
        # uṕdating points
        # self.rt_graph.set_offsets(self.positions[:, [0, 1]])
        # self.rt_graph.set_3d_properties(self.positions[:, 2], 'z')
        # self.rt_graph.do_3d_projection(self.rt_fig._cachedRenderer)
        self.rt_graph._offsets3d = (self.positions[:,0], self.positions[:,1], self.positions[:,2])
        
        self.rt_graph.set_array(colors)

        self.rt_fig.canvas.draw()
        self.rt_fig.canvas.flush_events()

        self.plot_images += [np.fromstring(self.rt_fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape( self.rt_fig.canvas.get_width_height()[::-1]+(3,) ) ]

        plt.figure(self.rt_fig.number)
        plt.savefig(self.results_folder_name+'last_anim_frame.png')
        
        return

    def save_plot_real_time(self):
        if len(self.rt_plot) > 0:
            plt.close(fig = self.rt_fig)
            print('Saving animation...')
            imageio.mimsave(self.results_folder_name+'simulation.gif', self.plot_images, fps=10)
            print('Saved!')

    def plot_figures(self, geometry, property_plot=['temperature'], colormap = 'viridis'):

        fig = plt.figure( figsize = (8, 8), dpi = 150 )
        n = len(property_plot)

        for i in range(n):
            if property_plot[i] in ['T', 'temperature']:
                data = self.temperatures
                figname = 'temperature'
                title = 'Temperature [K]'
            elif property_plot[i] in ['omega', 'angular_frequency']:
                data = self.omega
                figname = 'angular_frequency'
                title = 'Angular Frequency [rad/s]'
            elif property_plot[i] in ['n', 'occupation']:
                data = np.log(self.occupation)
                figname = 'occupation'
                title = 'Occupation Number'
            elif property_plot[i] in ['e', 'energy']:
                data = self.energies
                figname = 'energy'
                title = 'Energy [eV]'

            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
            graph = ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], cmap = colormap, c = data, s = 1)
            fig.colorbar(graph, orientation='horizontal')
            plt.title(title, {'size': 15} )

            plt.tight_layout()
            
            plt.savefig(self.results_folder_name+figname+'.png')

    def open_convergence(self):

        n_dt_to_conv = np.floor( np.log10( self.args.iterations[0] ) ) - 2    # number of timesteps to save convergence data
        n_dt_to_conv = int(10**n_dt_to_conv)
        n_dt_to_conv = max([10, n_dt_to_conv])

        self.n_dt_to_conv = 10


        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        line = '# '
        line += 'Real Time                  '
        line += 'Timest. '
        line += 'Simul. Time '
        line += 'Total Energy '
        if self.n_of_reservoirs > 0:
            for i in range(self.n_of_reservoirs):
                line += 'En Bal Res {} '.format(i)
            for i in range(self.n_of_reservoirs):
                line += ' Hflux x Res {} '.format(i)
                line += ' Hflux y Res {} '.format(i)
                line += ' Hflux z Res {} '.format(i)
            for i in range(self.n_of_reservoirs):
                line += ' Momnt x Res {} '.format(i)
                line += ' Momnt y Res {} '.format(i)
                line += ' Momnt z Res {} '.format(i)
        line += ' No. Part. '
        for i in range(self.n_of_subvols):
            line += ' T Sv {:>3d} '.format(i)# temperature per subvol
        for i in range(self.n_of_subvols):
            line += ' Energ Sv {:>2d} '.format(i) 
        for i in range(self.n_of_subvols):
            line += ' Hflux x Sv {:>2d} '.format(i)
            line += ' Hflux y Sv {:>2d} '.format(i)
            line += ' Hflux z Sv {:>2d} '.format(i)
        for i in range(self.n_of_subvols):
            line += ' Momnt x Sv {:>2d} '.format(i)
            line += ' Momnt y Sv {:>2d} '.format(i)
            line += ' Momnt z Sv {:>2d} '.format(i)
        for i in range(self.n_of_subvols):
            line += ' Np Sv {:>2d}  '.format(i)
        
        line += '\n'

        self.f.write(line)
        self.f.close()

    def write_convergence(self):
        
        line = ''
        line += datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f ')  # real time in ISO format
        line += '{:>8d} '.format( int(self.current_timestep) )   # timestep
        line += '{:>12.5e} '.format( self.t )   # time
        if self.energies.shape[0]>0:
            line += '{:>12.5e} '.format( self.energies.sum() )  # average energy
        else:
            line += '{:>11.5e} '.format( 0 )
        
        if self.n_of_reservoirs>0:
            line += np.array2string(self.res_energy_balance, formatter = {'float_kind':'{:>12.5e}'.format}).strip('[]') + ' '
            for i in range(self.n_of_reservoirs):
                line += np.array2string(self.res_heat_flux[i, :], formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '
            for i in range(self.n_of_reservoirs):
                line += np.array2string(self.res_momentum_balance[i, :], formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '

        line += '{:>10d} '.format( self.omega.shape[0] )  # number of particles

        line += np.array2string(self.subvol_temperature, formatter = {'float_kind':'{:>9.3f}'.format} ).strip('[]') + ' '
        line += np.array2string(self.subvol_energy     , formatter = {'float_kind':'{:>12.5e}'.format}).strip('[]') + ' '
        
        for i in range(self.n_of_subvols):
            line += np.array2string(self.subvol_heat_flux[i, :], formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '
        for i in range(self.n_of_subvols):
            line += np.array2string(self.subvol_momentum[i, :] , formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '

        line += np.array2string(self.subvol_N_p        , formatter = {'int'       :'{:>10d}'.format}  ).strip('[]') + ' '
        
        line += '\n'

        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        self.f.writelines(line)

        self.f.close()

    def write_final_state(self, geometry, phonon):
        
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # saving final particle states: modes, positions, subvol id and occupation number.
        # Obs.: phonon properties of particles can be retrieved by mode information and subvol temperature
        
        # filename = self.results_folder_name + 'particle_data_{:.0f}ps.txt'.format(self.t)
        filename = self.results_folder_name + 'particle_data.txt'

        header ='Particles final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'q-point, branch, pos x [angs], pos y [angs], pos z [angs], subvol, occupation'

        data = np.hstack( (self.modes,
                           self.positions,
                           self.occupation.reshape(-1, 1)) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %d, %.3f, %.3f, %.3f, %.6e', delimiter = ',', header = header)
        
        # modes data
        # filename = self.results_folder_name + 'modes_data_{:.0f}ps.txt'.format(self.t)
        filename = self.results_folder_name + 'modes_data.txt'
        
        header ='Occupation data for each subvolume \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'T_ref = {} K \n'.format(self.T_reference) + \
                'Reshape data to proper shape when analysing (SV, Q, J).'
        
        data = self.occupation_lookup.reshape(-1, phonon.number_of_branches)

        np.savetxt(filename, data, '%.6e', delimiter = ',', header = header)

        # saving final subvol states: temperatures and heat flux

        filename = self.results_folder_name + 'subvol_data.txt'

        header ='subvols final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'subvol id, subvol volume, temperature [K], heat flux x, y, and z [W/m^2]'

        data = np.hstack( (np.arange(self.n_of_subvols).reshape(-1, 1),
                           geometry.subvol_center, 
                           self.subvol_volume.reshape(-1, 1),
                           self.subvol_temperature.reshape(-1, 1),
                           self.subvol_heat_flux) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %.3e, %.3e, %.3e, %.3e, %.3f, %.3e, %.3e, %.3e', delimiter = ',', header = header)

####################################################

#  ▄  █ ▄█ ▄█▄           ▄▄▄▄▄   ▄      ▄     ▄▄▄▄▀     ██▄   █▄▄▄▄ ██   ▄█▄    ████▄    ▄   ▄███▄     ▄▄▄▄▄   
# █   █ ██ █▀ ▀▄        █     ▀▄  █      █ ▀▀▀ █        █  █  █  ▄▀ █ █  █▀ ▀▄  █   █     █  █▀   ▀   █     ▀▄ 
# ██▀▀█ ██ █   ▀      ▄  ▀▀▀▀▄ █   █ ██   █    █        █   █ █▀▀▌  █▄▄█ █   ▀  █   █ ██   █ ██▄▄   ▄  ▀▀▀▀▄   
# █   █ ▐█ █▄  ▄▀      ▀▄▄▄▄▀  █   █ █ █  █   █         █  █  █  █  █  █ █▄  ▄▀ ▀████ █ █  █ █▄   ▄▀ ▀▄▄▄▄▀    
#    █   ▐ ▀███▀               █▄ ▄█ █  █ █  ▀          ███▀    █      █ ▀███▀        █  █ █ ▀███▀             
#   ▀                           ▀▀▀  █   ██                    ▀      █               █   ██                   
#                                                                    ▀                                               

# This is the edge of the code. Here beyond are methods that were coded, but substituted or outrightly excluded.
# I am a little reluctant abound deleting them though, since they can evenetually be useful in de future, or
# give some hint of some coding problem I may have. But they are really not being used.

################# 
    
    

    
    def print_sim_info(self, geometry, phonon):
        print('---------- o ----------- o ------------- o ------------')
        print('Simulation Info:')

        T_max = self.res_facet_temperature.max()

        Tqj = np.hstack((np.ones(phonon.number_of_active_modes).reshape(-1, 1)*T_max, phonon.unique_modes))
        tau = phonon.lifetime_function(Tqj)
        v = np.absolute(phonon.group_vel[phonon.unique_modes[:, 0], phonon.unique_modes[:, 1], self.slice_axis])

        kn = v*tau/self.slice_length

        maximum_n_slices = int(np.floor(geometry.bounds[:, self.slice_axis].ptp()/(v.max()*self.dt)))

        prob_enter = 100*v*self.dt/self.slice_length
        prob_enter = np.where(prob_enter>100, 100, prob_enter)

        print('Slice Knudsen number: Max = {:.3e}, Min = {:.3e}'.format(kn.max(), kn.min()))
        print('Maximum number of slices: {}'.format(maximum_n_slices))
        print('Probability of entering: Max = {:.3f}%, Min = {:.3f}%'.format(prob_enter.max(), prob_enter.min()))
        print('---------- o ----------- o ------------- o ------------')
