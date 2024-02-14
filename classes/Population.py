# calculations
import numpy as np
from datetime import datetime

# plotting
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate.ndgriddata import NearestNDInterpolator
from scipy.interpolate import interp1d, RBFInterpolator
from functools import partial

# other
import sys
import os
import copy
import gc

from classes.Constants     import Constants
from classes.Visualisation import Visualisation

np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=np.nan)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class with information about the particles contained in the domain.

#   TO DO
#   - Add option to apply heat flux as boundary condition
#   - Generalise for use with more than one material at once
#   - Add transmission on interface between materials

class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry, phonon):
        super(Population, self).__init__()

        self.args = arguments
        self.results_folder_name = self.args.results_folder
        
        self.n_dt_to_conv = 10 # number of timesteps for each convergence datapoint

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
        
        self.subvol_volume  = geometry.subvol_volume  # angstrom³

        self.bound_cond          = geometry.bound_cond
        self.res_gen   = self.args.reservoir_gen[0]
        
        self.rough_facets        = geometry.rough_facets        # which facets have roughness as BC
        self.rough_facets_values = geometry.rough_facets_values # their values of roughness and correlation length

        print('Calculating diffuse scattering probabilities...')
        self.calculate_fbz_specularity(geometry, phonon)
        self.find_specular_correspondences(geometry, phonon)
        self.diffuse_scat_probability(geometry, phonon)

        self.connected_facets = geometry.connected_facets

        self.T_distribution   = self.args.temp_dist[0]
        self.temp_interp_type = self.args.temp_interp[0]
        
        if self.args.reference_temp[0] != 'local':
            self.T_reference = float(self.args.reference_temp[0])
            self.reference_occupation = phonon.calculate_occupation(self.T_reference, phonon.omega)
            self.ref_en_density = phonon.crystal_energy_function(self.T_reference)
        else:
            self.T_reference = 'local'

        self.colormap = self.args.colormap[0]
        self.fig_plot = self.args.fig_plot

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

        self.conv_crit = float(self.args.conv_crit[0])
        self.conv_count_min = int(self.args.conv_crit[1])
        self.initialise_residue(geometry)
        
        print('Creating convergence file...')
        self.open_convergence(geometry)
        self.write_convergence(geometry)

        self.view = Visualisation(self.args, geometry, phonon, self) # initialising visualisation class
        self.plot_figures(geometry, phonon, property_plot = self.args.fig_plot, colormap = self.args.colormap[0])

        print('Initialisation done!')

    def initialise_modes(self, phonon):
        '''Generate first modes.'''

        print('Assigning modes...')

        # creating unique mode matrix
        self.unique_modes = np.vstack(np.where(~phonon.inactive_modes_mask)).T

        if self.particles_pmps >= 1:
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

        bound_thickness = phonon.number_of_active_modes/(self.particle_density*geometry.facets_area[self.res_facet])

        vel = np.transpose(phonon.group_vel, (0, 2, 1))           # shape = (Q, 3, J) - Group velocities of each mode
        normals = -geometry.facets_normal[self.res_facet, :] # shape = (R, 3)    - unit normals of each facet with boundary conditions
                                                                  # OBS: normals are reversed to point inwards (in the direction of entering particles).
        group_vel_parallel = np.dot(normals, vel)                 # shape = (R, Q, J) - dot product = projection of velocity over normal
        
        # Probability of a particle entering the domain
        enter_prob = group_vel_parallel*self.dt/bound_thickness.reshape(-1, 1, 1)   # shape = (R, Q, J)
        enter_prob = np.where(enter_prob < 0, 0, enter_prob)

        return enter_prob

    def generate_positions(self, number_of_particles, mesh, key):
        '''Initialise positions of a given number of particles'''
        
        if key == 'random':
            positions = mesh.sample_volume(number_of_particles)
            # in_mesh = mesh.closest_point(positions)[1] >= 0
            in_mesh = mesh.contains_naive(positions)
            positions = positions[in_mesh, :]
            
            while positions.shape[0]<number_of_particles:
                new_positions = mesh.sample_volume(number_of_particles-positions.shape[0])
                if new_positions.shape[0]>0:
                    # in_mesh = mesh.closest_point(new_positions)[1] >= 0
                    in_mesh = mesh.contains_naive(new_positions)
                    new_positions = new_positions[in_mesh, :]
                    positions = np.vstack((positions, new_positions))
            
        elif key == 'center':
            center = mesh.center_mass
            positions = np.ones( (number_of_particles, 3) )*center # Generate all points at the center of the bounding box, positions in angstrom
            
        return positions

    def initialise_all_particles(self, geometry, phonon):
        '''Uses Population atributes to generate all particles.'''
        
        if sys.stdout.isatty():
            # setup toolbar
            nchar = int(np.floor(np.log10(self.N_p)))+1
            bw = os.get_terminal_size(sys.stdout.fileno()).columns - (2*nchar + 24)
            message = "{0:{3}d} of {1:{3}d} particles - {2:s}   0%".format(0, self.N_p, " "*bw, nchar)
            sys.stdout.write(message)
            sys.stdout.flush()
        
        if self.args.part_dist[0] in ['random_domain', 'center_domain', 'random_subvol', 'center_subvol']:
            # initialising positions one mode at a time (slower but uses less memory)
            self.positions = np.zeros((0, 3))

            if self.args.part_dist[0] == 'random_domain':
                number_of_particles = self.N_p
                self.positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'random')

            elif self.args.part_dist[0] == 'center_domain':
                number_of_particles = self.N_p
                self.positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'center')
            
            elif self.args.part_dist[0] == 'random_subvol':
                
                counter = np.zeros(self.n_of_subvols, dtype = int)

                n = self.N_p*geometry.subvol_volume/(geometry.subvol_volume.sum()-geometry.subvol_volume[self.empty_subvols].sum()) # number of particles for each subvolume
                n = np.ceil(n).astype(int)
                n[self.empty_subvols] = 0

                x = [np.zeros((0, 3)) for _ in range(self.n_of_subvols)]

                while np.any(counter < n):

                    x_new = self.generate_positions(min(n.sum() - counter.sum(), int(1e4)), geometry.mesh, key = 'random')

                    sv_id = geometry.subvol_classifier.predict(x_new)

                    for i in range(self.n_of_subvols):
                        if i not in self.empty_subvols:
                            
                            N = n[i] - counter[i]
                            
                            ind = np.nonzero(sv_id == i)[0]
                            ind = ind[:N]

                            x[i] = np.vstack((x[i], x_new[ind, :]))

                            counter[i] = x[i].shape[0]
                    
                    if sys.stdout.isatty():
                        # update the bar
                        bw = os.get_terminal_size(sys.stdout.fileno()).columns - (2*nchar + 24)
                        l = int(np.floor(bw*counter.sum()/self.N_p))
                        message = "\r{0:{3}d} of {1:{3}d} particles - {2:s} {4:3.0f}%".format(counter.sum(), self.N_p, "\u2588"*l+" "*(bw-l), nchar, 100*counter.sum()/self.N_p)
                        sys.stdout.write(message)
                        sys.stdout.flush()
                
                sys.stdout.write("\n")
                self.positions = np.vstack(x)[:self.N_p, :]

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
            self.subvol_id = self.get_subvol_id(self.positions, geometry, verbose = True)

            # assign temperatures
            self.temperatures, self.subvol_temperature = self.assign_temperatures(self.positions, geometry)
            
            self.occupation = phonon.calculate_occupation(self.temperatures, self.omega)

            self.calculate_energy(geometry, phonon)
        else:
            try:
                data = np.loadtxt(self.args.part_dist[0], delimiter = ',', comments = '#', dtype = float)
            except:
                raise Exception('Wrong particle data file. Change the keyword or check whether the file exists.')
            
            self.modes = np.copy(data[:, [0, 1]]).astype(int) # need to add a check to see if the material is the same as the input
            self.positions = np.copy(data[:, [2, 3, 4]]) # need to normalise and rescale positions to avoid conflicts

            self.occupation = np.copy(data[:, 5])
            self.subvol_id = self.get_subvol_id(self.positions, geometry)

            self.omega, self.group_vel, self.wavevectors = self.assign_properties(self.modes, phonon)

            self.temperatures, self.subvol_temperature = self.assign_temperatures(self.positions, geometry)
            old_T = np.zeros(self.n_of_subvols)

            err = 1
            while err > 1e-6:
                self.refresh_temperatures(geometry, phonon)
                err = np.absolute((self.subvol_temperature - old_T)/self.subvol_temperature).max()
                old_T = np.copy(self.subvol_temperature)
                
            del(data)
        
        print('Getting first boundary collisions...')
        # getting scattering arrays
        (self.n_timesteps        ,
         self.collision_facets   ,
         self.collision_positions) = self.timesteps_to_boundary(self.positions,
                                                                self.group_vel,
                                                                geometry      )

        self.collision_cond = self.get_collision_condition(self.collision_facets)

        print('Initialising local quantities...')
        self.calculate_energy(geometry, phonon)
        self.subvol_heat_flux = self.calculate_heat_flux(geometry, phonon)
        self.calculate_kappa(geometry)

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

        self.enter_prob = self.enter_probability(geometry, phonon)
        self.res_counter = np.random.rand(*self.enter_prob.shape)
        self.N_leaving  = np.sum(self.enter_prob, axis = (1, 2)).round().astype(int)

        self.res_energy_balance   = np.zeros(self.n_of_reservoirs)
        self.res_heat_flux        = np.zeros((self.n_of_reservoirs, 3))

        self.res_norm = np.zeros(self.n_of_reservoirs)
        if self.norm == 'fixed':
            for i in range(self.n_of_reservoirs):
                self.res_norm[i] = np.absolute(np.sum(phonon.group_vel[1:, :, :]*geometry.facets_normal[self.res_facet[i]], axis = 2)).mean()

            self.res_norm *= self.dt*self.particle_density*geometry.facets_area[self.res_facet]*self.n_dt_to_conv

    def fill_reservoirs(self, geometry, phonon, n_leaving = None):

        if self.res_gen == 'constant':
            fixed_np      = np.floor(self.enter_prob).astype(int) # number of particles that will enter every iteration
            
            self.res_counter += self.enter_prob - fixed_np

            in_modes_mask = (self.res_counter >= 1).astype(int) # shape = (R, Q, J)

            self.res_counter -= in_modes_mask

            in_modes_np = fixed_np + in_modes_mask

            # calculate how many particles entered each facet
            N_p_facet = in_modes_np.sum(axis = (1, 2))    # shape = (R,)

            # initialise new arrays
            self.res_positions = np.zeros((0, 3))
            self.res_modes     = np.zeros((0, 2), dtype = int)
            self.res_facet_id  = np.zeros(0, dtype = int)
            self.res_dt_in     = np.zeros(0, dtype = int)

            for i in range(self.n_of_reservoirs):                         # for each reservoir
                n      = N_p_facet[i]                                     # the number of particles on that facet
                if n > 0:
                    facet  = self.res_facet[i]                                # get its facet index
                    # mesh   = geometry.res_meshes[i]                           # select boundary
                    
                    # adding fixed particles
                    c = in_modes_np[i, :, :].max()  # gets the maximum number of particles of a single mode to be generated
                    while c > 0:
                        
                        c_modes = np.vstack(np.where(in_modes_np[i, :, :] >= c)).T

                        if c == 1:
                            c_dt_in = self.dt*(1-(self.res_counter[i, c_modes[:, 0], c_modes[:, 1]]/self.enter_prob[i, c_modes[:, 0], c_modes[:, 1]]))
                        else:
                            r = np.random.rand(c_modes.shape[0])
                            c_dt_in = self.dt*(1-(c-1+r)/self.enter_prob[i, c_modes[:, 0], c_modes[:, 1]])
                        
                        c -= 1
                        
                        self.res_dt_in = np.concatenate((self.res_dt_in, c_dt_in)) # add to the time drifted inside the domain
                        self.res_modes = np.vstack((self.res_modes, c_modes.astype(int))) # add to the modes

                    self.res_facet_id  = np.concatenate((self.res_facet_id, (np.ones(n)*facet).astype(int))) # add to the reservoir id

                    # generate positions on boundary
                    new_positions = geometry.mesh.sample_surface(n, facets = facet)

                    self.res_positions = np.vstack((self.res_positions , new_positions ))                  # add to the positions

        elif self.res_gen == 'fixed_rate':
            # generate random numbers
            dice = np.random.rand(self.n_of_reservoirs, phonon.number_of_qpoints, phonon.number_of_branches)

            # check if particles entered the domain comparing with their probability
            fixed_np      = np.floor(self.enter_prob).astype(int) # number of particles that will enter every iteration
            
            in_modes_mask = (dice <= (self.enter_prob - fixed_np)).astype(int) # shape = (R, Q, J)
            
            in_modes_np   = fixed_np + in_modes_mask

            # calculate how many particles entered each facet
            N_p_facet = in_modes_np.sum(axis = (1, 2))    # shape = (R,)

            # initialise new arrays
            self.res_positions = np.zeros((0, 3))
            self.res_modes     = np.zeros((0, 2), dtype = int)
            self.res_facet_id  = np.zeros(0, dtype = int)
            self.res_dt_in     = np.zeros(0, dtype = int)

            for i in range(self.n_of_reservoirs):                         # for each reservoir
                n      = N_p_facet[i]                                     # the number of particles on that facet
                if n > 0:
                    facet  = self.res_facet[i]                                # get its facet index
                    
                    # adding fixed particles
                    c = in_modes_np[i, :, :].max()  # gets the maximum number of particles of a single mode to be generated
                    while c > 0:
                        
                        c_modes = np.vstack(np.where(in_modes_np[i, :, :] >= c)).T

                        if c == 1:
                            c_dt_in = self.dt*(1-(dice[i, c_modes[:, 0], c_modes[:, 1]]/self.enter_prob[i, c_modes[:, 0], c_modes[:, 1]]))
                        else:
                            r = np.random.rand(c_modes.shape[0])
                            c_dt_in = self.dt*(1-(c-1+r)/self.enter_prob[i, c_modes[:, 0], c_modes[:, 1]])
                        
                        c -= 1
                        
                        self.res_dt_in = np.concatenate((self.res_dt_in, c_dt_in)) # add to the time drifted inside the domain
                        self.res_modes = np.vstack((self.res_modes, c_modes.astype(int))) # add to the modes

                    self.res_facet_id  = np.concatenate((self.res_facet_id, (np.ones(n)*facet).astype(int))) # add to the reservoir id

                    # generate positions on boundary
                    new_positions = geometry.mesh.sample_surface(n, facets = facet)

                    self.res_positions = np.vstack((self.res_positions , new_positions ))                  # add to the positions
            
        elif self.res_gen == 'one_to_one':
            # initialise new arrays
            self.res_positions = np.zeros((0, 3))
            self.res_modes     = np.zeros((0, 2), dtype = int)
            self.res_facet_id  = np.zeros(0, dtype = int)
            self.res_dt_in     = np.zeros(0, dtype = int)

            for i in range(self.n_of_reservoirs):                  # for each reservoir
                facet  = self.res_facet[i]                         # get its facet index
                n      = n_leaving[i]                              # the number of particles on that facet
                if n > 0:
                    roulette = np.cumsum(self.enter_prob[i, :, :])
                    roulette /= roulette.max()

                    r = np.random.rand(n) # generating dies

                    flat_i = np.searchsorted(roulette, r) # searching for modes

                    # getting the new qpoints and branches
                    new_q  = np.floor(flat_i/phonon.number_of_branches).astype(int)
                    new_j = flat_i - new_q*phonon.number_of_branches

                    new_modes = np.vstack((new_q, new_j)).T

                    self.res_modes = np.vstack((self.res_modes, new_modes)) # storing
                    
                    self.res_dt_in = np.concatenate((self.res_dt_in, self.dt*np.random.rand(n))) # random generated time inside domain
                    self.res_facet_id = np.concatenate((self.res_facet_id, (np.ones(n)*facet).astype(int))) # add to the reservoir id

                    # generate positions on boundary
                    new_positions = geometry.mesh.sample_surface(n, facets = facet)

                    self.res_positions = np.vstack((self.res_positions , new_positions ))                  # add to the positions
        
        if self.res_modes.shape[0]>0:
            
            self.res_group_vel   = phonon.group_vel[self.res_modes[:, 0], self.res_modes[:, 1], :]        # retrieve velocities

            self.res_omega       = phonon.omega[self.res_modes[:, 0], self.res_modes[:, 1]]               # retrieve frequencies
            
            self.res_wavevectors = phonon.wavevectors[self.res_modes[:, 0], :]
        
        # ADD FLUX BOUNDARY CONDITION HERE

        indexes = np.where(self.res_facet_id.reshape(-1, 1) == self.res_facet)[1]   # getting RESERVOIR indexes

        self.res_temperatures = self.res_facet_temperature[indexes] # impose temperature values to the right particles

        if self.res_modes.shape[0]>0:
            self.res_occupation = phonon.calculate_occupation(self.res_temperatures, self.res_omega)
            if self.T_reference == 'local':
                self.res_energies = np.zeros(self.res_omega.shape)
                if self.norm == 'mean':
                    for i in range(self.n_of_reservoirs):
                        self.res_norm[i] += indexes.sum()
            else:
                dn = phonon.calculate_occupation(self.res_temperatures, self.res_omega) - self.reference_occupation[self.res_modes[:, 0], self.res_modes[:, 1]]

                self.res_energies = self.hbar*self.res_omega*dn
                for i in range(self.n_of_reservoirs):
                    facet   = self.res_facet[i]
                    indexes = self.res_facet_id == facet
                    self.res_energy_balance[i] = self.res_energies[indexes].sum()
                    self.res_heat_flux[i, :]   = (self.res_group_vel[indexes, :]*self.res_energies[indexes].reshape(-1, 1)).sum(axis = 0)

                    if self.norm == 'mean':
                        self.res_norm[i] += indexes.sum()

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

        print('Assigning properties...')

        omega      = phonon.omega[ modes[:,0], modes[:,1] ]        # THz * rad
        group_vel  = phonon.group_vel[ modes[:,0], modes[:,1], : ] # THz * angstrom
        wavevector = phonon.wavevectors[modes[:, 0], :]

        return omega, group_vel, wavevector

    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Constant at T_cold unless specified otherwise.'''

        print('Assigning temperatures...')

        if self.temp_interp_type in ['nearest', 'linear'] and geometry.subvol_type == 'slice':
            self.temp_interp = partial(interp1d, kind = self.temp_interp_type, fill_value = 'extrapolate')
        elif self.temp_interp_type == 'nearest':
            self.temp_interp = NearestNDInterpolator
        elif self.temp_interp_type in ['radial', 'linear']:
            if self.temp_interp_type == 'linear':
                print('Linear T interpolation is currently valid for slice subvolumes only. Defaulting to RBF interpolation to avoid extrapolation problems.')

            # NOTE: RBFInterpolator is kinda hard to make it work properly:
            #       - The "neighbours" keyword does not seem to work to reduce the influence.
            #       - Every point influences the result, so the distribution of temperature seems to cause a feedback loop that makes it hotter or colder. Need to confirm.
            #       - Other kernels need an aditional shape parameter that can be harder for the user to tune.
            #       
            #       An alternative is to use linear interpolator, but then the interpolator needs to include the points on the borders and corners
            #       of the geometry so that the convex hull of the interpolator corresponds to that of the geometry. That may be too difficult to do
            #       in general.
            #       Maybe a shape parameter identifier?

            self.temp_interp = partial(RBFInterpolator, kernel = 'cubic')
        else:
            raise Exception('Invalid T interpolator type.')

        number_of_particles = positions.shape[0]
        key = self.T_distribution

        if key == 'custom':
            subvol_temperatures = np.array(self.args.subvol_temp)
            temperatures = subvol_temperatures[self.subvol_id] # .sum(axis = -1)/self.subvol_id.sum(axis = -1)
        
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
                    d = (geometry.subvol_center - np.expand_dims(bound_positions, 1)) # (R, SV, 3)
                    d = (np.sum(d**2, axis = 2).T)**0.5                               # (SV, R)
                    
                    w = 1/d
                    w /= np.sum(w, axis = 1, keepdims = True)                        # (SV, R)

                    subvol_temperatures = np.sum(bound_T*w, axis = 1)
                    
                    temperatures = subvol_temperatures[self.subvol_id]

                elif len(bound_T) == 1:
                    subvol_temperatures = np.ones(self.n_of_subvols)*bound_T
                elif len(bound_T) == 2:
                    direction = bound_positions[1, :]-bound_positions[0, :]
                    K = ((geometry.subvol_center-bound_positions[0, :])*direction).sum(axis = 1)
                    alphas = K/(direction**2).sum()

                    subvol_temperatures = bound_T[0]+alphas*(bound_T[1]-bound_T[0])
                    temperatures = subvol_temperatures[self.subvol_id]
                
            elif key == 'random':
                subvol_temperatures = np.random.rand(self.n_of_subvols)*bound_T.ptp() + bound_T.min()
                temperatures = subvol_temperatures[self.subvol_id]
            elif key == 'hot':
                temperatures        = np.ones(number_of_particles)*bound_T.max()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.max()
            elif key == 'cold':
                temperatures        = np.ones(number_of_particles)*bound_T.min()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.min()
            elif key == 'mean':
                temperatures        = np.ones(number_of_particles)*bound_T.mean()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.mean()
        
        if geometry.subvol_type == 'slice' and self.temp_interp_type in ['linear', 'nearest']:
            self.temperature_interpolator = self.temp_interp(geometry.subvol_center[:, self.slice_axis], subvol_temperatures)
        elif geometry.subvol_type == 'grid' and np.any(geometry.grid == 1):
            self.temperature_interpolator = self.temp_interp(geometry.subvol_center[:, geometry.grid != 1], subvol_temperatures)
        else:
            self.temperature_interpolator = self.temp_interp(geometry.subvol_center, subvol_temperatures)
        
        return temperatures, subvol_temperatures
    
    def get_collision_condition(self, collision_facets):
        
        collision_cond = np.empty(collision_facets.shape, dtype = str) # initialise as an empty string array
        
        nan_i = np.isnan(collision_facets)

        collision_cond[nan_i] = 'N'              # identify all nan facets with 'N'

        non_nan_facets = collision_facets[~nan_i].astype(int) # get non-nan facets
        
        collision_cond[~nan_i] = self.bound_cond[non_nan_facets] # save their condition

        return collision_cond

    def get_subvol_id(self, positions, geometry, get_np = True, verbose = False):
        
        if verbose:
            print('Identifying subvols...')

        subvol_id = geometry.subvol_classifier.predict(positions) # get subvol_id from the model

        if get_np:
            self.subvol_N_p = np.array([(subvol_id==i).sum(dtype = int) for i in range(self.n_of_subvols)])

            self.N_p = self.subvol_N_p.sum()

        return subvol_id

    def refresh_temperatures(self, geometry, phonon):
        '''Refresh energies and temperatures while enforcing boundary conditions as given by geometry.'''

        self.subvol_id = self.get_subvol_id(self.positions, geometry)

        self.calculate_energy(geometry, phonon)

        self.subvol_temperature = phonon.temperature_function(self.subvol_energy)

        if geometry.subvol_type == 'slice' and self.temp_interp_type in ['linear', 'nearest']:
            self.temperature_interpolator = self.temp_interp(geometry.subvol_center[:, self.slice_axis], self.subvol_temperature)
            self.temperatures = self.temperature_interpolator(self.positions[:, self.slice_axis])
        elif geometry.subvol_type == 'grid' and np.any(geometry.grid == 1):
            self.temperature_interpolator = self.temp_interp(geometry.subvol_center[:, geometry.grid != 1], self.subvol_temperature)
            self.temperatures = self.temperature_interpolator(self.positions[:, geometry.grid != 1])
        else:
            self.temperature_interpolator = self.temp_interp(geometry.subvol_center, self.subvol_temperature)
            self.temperatures = self.temperature_interpolator(self.positions)
        
    def calculate_energy(self, geometry, phonon):
        
        if self.T_reference == 'local':
            dn = self.occupation - phonon.calculate_occupation(self.subvol_temperature[self.subvol_id], self.omega)
            ref = phonon.crystal_energy_function(self.subvol_temperature)
        else:
            dn = self.occupation - self.reference_occupation[self.modes[:, 0], self.modes[:, 1]]
            ref = self.ref_en_density
        
        self.energies = self.hbar*self.omega*dn
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
        
        self.subvol_energy += ref

    def calculate_heat_flux(self, geometry, phonon):
        
        heat_flux = np.zeros((self.n_of_subvols, 3))

        for i in range(self.n_of_subvols):
            ind = np.nonzero(self.subvol_id == i)[0] # 1d subvol id
            heat_flux[i, :] = np.sum(self.group_vel[ind, :]*self.energies[ind].reshape(-1, 1), axis = 0)

        if self.norm == 'fixed':
            normalisation = phonon.number_of_active_modes/(self.particle_density*geometry.subvol_volume.reshape(-1, 1))
        elif self.norm == 'mean':
            normalisation = phonon.number_of_active_modes/self.subvol_N_p.reshape(-1, 1)
        
        heat_flux = heat_flux*normalisation
        
        heat_flux = phonon.normalise_to_density(heat_flux)
        
        return heat_flux*self.eVpsa2_in_Wm2 # subvol heat flux
        
    def calculate_kappa(self, geometry):
        if geometry.subvol_type == 'slice':
            # if the subvol type is slice, the simulation is basically 1d and
            # kappa can be calculated the usual way
            T = np.zeros(self.n_of_subvols+2)
            T[1:-1] = self.subvol_temperature
            T[[0, -1]] = self.res_facet_temperature

            phi = self.subvol_heat_flux[:, geometry.slice_axis]
            
            # local gradients
            dx = 2*geometry.bounds[:, geometry.slice_axis].ptp()*self.a_in_m/self.n_of_subvols
            dT = T[2:] - T[:-2]

            # total gradient
            DX = geometry.bounds[:, geometry.slice_axis].ptp()*self.a_in_m*(1+self.n_of_subvols)/self.n_of_subvols
            DT = T[-1] - T[0]
            
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                self.subvol_kappa = -phi*dx/dT
                self.kappa        = -np.sum(phi*self.subvol_N_p)*(DX/DT)/self.N_p
            
            self.subvol_kappa[np.absolute(self.subvol_kappa) == np.inf] = 0

        else:
            # renaming the indices to keep the code readable
            i = geometry.subvol_connections[:,0]
            j = geometry.subvol_connections[:,1]

            # local gradients
            dx = geometry.subvol_center[j, :] - geometry.subvol_center[i, :] # (C, 3)
            n  = dx/np.linalg.norm(dx, axis = 1, keepdims = True)              # normalised dx (C, 3)
            dT = self.subvol_temperature[j] - self.subvol_temperature[i]       # (C,)

            phi = (self.subvol_heat_flux[i, :] + self.subvol_heat_flux[j, :])/2 # (C,3)

            phi_dot_n = np.sum(phi*n, axis = 1)

            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                self.svcon_kappa = np.where(dT == 0, 0, -phi_dot_n*np.linalg.norm(dx, axis = 1)*self.a_in_m/dT)

    def drift(self):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt # move forward by one v*dt step

        self.n_timesteps -= 1                    # -1 timestep in the counter to boundary scattering

    def timesteps_to_boundary(self, positions, velocities, geometry, indexes_del_extra = None):
        '''Calculate how many timesteps to boundary scattering.'''

        if positions.shape[0] == 0:
            ts_to_boundary      = np.zeros(0)
            index_facets        = np.zeros(0)
            collision_pos        = np.zeros((0, 3))
            
            return ts_to_boundary, index_facets, collision_pos

        mesh_pos = np.zeros((0, 3))
        mesh_facets = np.zeros(0, dtype = int)
        ts_to_boundary = np.zeros(0, dtype = float)
        stride = int(1e6)
        start = 0
        while start < positions.shape[0]:
            new_mesh_pos, new_ts_to_boundary, new_mesh_facets = geometry.mesh.find_boundary(positions[start:start+stride, :], velocities[start:start+stride, :]) # find collision in true boundary
            
            new_ts_to_boundary /= self.dt

            mesh_pos = np.concatenate((mesh_pos, new_mesh_pos), axis = 0)
            mesh_facets = np.concatenate((mesh_facets, new_mesh_facets), axis = 0, dtype = int)
            
            ts_to_boundary = np.concatenate((ts_to_boundary, new_ts_to_boundary), axis = 0, dtype = float)

            start += stride

        index_facets = mesh_facets
        collision_pos = mesh_pos
        
        if indexes_del_extra is None:
            return ts_to_boundary, index_facets, collision_pos
        else:
            return ts_to_boundary, index_facets, collision_pos, indexes_del_extra

    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes.
           
           Arguments:
              indexes: (bool or int) indexes of the particles to be deleted.'''

        self.positions           = np.delete(self.positions          , indexes, axis = 0)
        self.group_vel           = np.delete(self.group_vel          , indexes, axis = 0)
        self.wavevectors         = np.delete(self.wavevectors        , indexes, axis = 0)
        self.omega               = np.delete(self.omega              , indexes, axis = 0)
        self.occupation          = np.delete(self.occupation         , indexes, axis = 0)
        self.energies            = np.delete(self.energies           , indexes, axis = 0)
        self.temperatures        = np.delete(self.temperatures       , indexes, axis = 0)
        self.n_timesteps         = np.delete(self.n_timesteps        , indexes, axis = 0)
        self.modes               = np.delete(self.modes              , indexes, axis = 0)
        self.collision_facets    = np.delete(self.collision_facets   , indexes, axis = 0)
        self.collision_positions = np.delete(self.collision_positions, indexes, axis = 0)
        self.collision_cond      = np.delete(self.collision_cond     , indexes, axis = 0)
        self.subvol_id           = np.delete(self.subvol_id          , indexes, axis = 0)
    
    def calculate_fbz_specularity(self, geometry, phonon):

        n = -geometry.facets_normal[self.rough_facets, :] # normals (F, 3)
        n = np.expand_dims(n, axis = (1, 2)) # (F, 1, 1, 3)

        k = phonon.wavevectors
        k_norm = np.sum(k**2, axis = 1)**0.5 # (Q,)

        eta = self.rough_facets_values # getting roughness (F,)

        v = phonon.group_vel                  # (Q, J, 3)
        v_norm = np.sum(v**2, axis = -1)**0.5 # (Q, J)

        dot = np.sum(v*n, axis = -1)        # (F, Q, J)
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'): # to stop it giving errors when ||v|| = 0
            self.incidence_cos = dot/v_norm     # (F, Q, J)

        eta    = np.expand_dims(eta   , axis = (1, 2)) # (F, 1, 1)
        k_norm = np.expand_dims(k_norm, axis = (0, 2)) # (1, Q, 1)

        self.specularity_constant = -(2*eta*self.incidence_cos)**2 # (F, Q, J)

        self.specularity = np.exp(self.specularity_constant*(k_norm**2)) # (F, Q, J)

        self.specularity[np.isnan(self.specularity)] = 0

    def diffuse_scat_probability(self, geometry, phonon):

        if self.rough_facets.shape[0] == 0:
            print("No rough facets to calculate.")
        else:
            n = -geometry.facets_normal[self.rough_facets, :] # (F, 3)

            # sign of reflected waves
            v_all = phonon.group_vel # (Q, J, 3)

            v_dot_n = np.expand_dims(v_all, 2)*n        # (Q, J, F, 3)
            v_dot_n = v_dot_n.sum(axis = -1)            # (Q, J, F)
            v_dot_n = np.transpose(v_dot_n, (2, 0, 1))  # (F, Q, J)

            self.creation_roulette = np.zeros((self.rough_facets.shape[0], phonon.number_of_qpoints*phonon.number_of_branches))

            C_total = np.where(v_dot_n > 0, v_dot_n, 0)  # total rate of creation    (F, Q, J)
            D_total = np.where(v_dot_n < 0, -v_dot_n, 0)  # total rate of destruction (F, Q, J)
            
            specular_D = D_total*self.specularity

            corr_n = self.correspondent_modes[:, :3]
            in_q  = self.correspondent_modes[:, 3].astype(int)
            in_j  = self.correspondent_modes[:, 4].astype(int)
            out_q = self.correspondent_modes[:, 5].astype(int)
            out_j = self.correspondent_modes[:, 6].astype(int)

            self.creation_rate = np.copy(C_total)                                                # assume all creation is diffuse

            self.creation_rate = np.where(np.isnan(self.creation_rate), 0, self.creation_rate)

            unique_n, inv_n = np.unique(n, axis = 0, return_inverse = True) # getting unique normals of rough facets

            for i_n, u_n in enumerate(unique_n):
                i_f = (inv_n == i_n).nonzero()[0] # facets with that normal
                corr_indices = (np.linalg.norm(corr_n - u_n, axis = 1) < 1e-10).nonzero()[0]

                # unique outward modes and counts to find repetitions
                corr = self.correspondent_modes[corr_indices, :]
                unique_out, inv_out, count_out = np.unique(corr[:, [0,1,2,5,6]], axis = 0, return_counts = True, return_inverse = True)
            
                once = np.isin(inv_out, (count_out == 1).nonzero()[0]).nonzero()[0] # modes originated only by one specular reflection
                more_unique = (count_out > 1).nonzero()[0] # modes originated by more than one specular reflection
            
                for ii_f in i_f:
                    self.creation_rate[ii_f, out_q[corr_indices][once], out_j[corr_indices][once]] -= specular_D[ii_f, in_q[corr_indices][once], in_j[corr_indices][once]] # remove the creation due to specular scattering

                    for mu in more_unique: # for each repeated outward mode
                        repeated = (inv_out == mu).nonzero()[0]
                        self.creation_rate[ii_f, unique_out[mu, 3].astype(int), unique_out[mu, 4].astype(int)] -= specular_D[ii_f, in_q[corr_indices][repeated], in_j[corr_indices][repeated]].sum()

            self.find_degeneracies(phonon)
            if self.scat_model in ['k', 'wavevector', 'wave_vector']:

                for i in range(self.degeneracies.shape[0]):
                    self.creation_rate[:, self.degeneracies[i, 0], self.degeneracies[i, [1, 2]]] = self.creation_rate[:, self.degeneracies[i, 0], self.degeneracies[i, [1, 2]]].mean(axis = -1, keepdims = True)

            self.creation_rate = np.around(self.creation_rate, decimals = 10) # DEBUG

            for i_f, _ in enumerate(self.rough_facets):
                self.creation_roulette[i_f, :] = np.cumsum(self.creation_rate[i_f, :, :])/np.cumsum(self.creation_rate[i_f, :, :]).max()
            
    def select_reflected_modes(self, in_modes, col_fac, col_pos, n_in, omega_in, geometry, phonon):
        
        col_fac = col_fac.astype(int)
        i_rough = np.array([np.nonzero(self.rough_facets == f)[0][0] for f in col_fac])
        true_spec = self.true_specular[i_rough, in_modes[:, 0], in_modes[:, 1]]
        p = self.specularity[i_rough, in_modes[:, 0], in_modes[:, 1]]

        n_p = in_modes.shape[0]
        r = np.random.rand(n_p)

        indexes_spec = np.logical_and(true_spec, r <= p)
        indexes_diff = ~indexes_spec

        out_modes = np.zeros(in_modes.shape, dtype = int)
        n_out     = copy.copy(n_in)
        omega_out = copy.copy(omega_in)

        if np.any(indexes_spec):
            a = np.hstack((-geometry.facets_normal[col_fac[indexes_spec], :], in_modes[indexes_spec, :]))

            out_spec_modes = self.specular_function(a).astype(int)
            
            if self.scat_model in ['k', 'wavevector', 'wave_vector']: # TESTE TESTE TESTE TESTE TESTE TESTE TESTE TESTE TESTE TESTE TESTE TESTE 
                # Distributing energy equally among degenerate modes
                deg_indices = self.degen_index[out_spec_modes[:, 0], out_spec_modes[:, 1]].astype(int)

                indexes_change = np.logical_and(deg_indices > -1, np.random.rand(deg_indices.shape[0]) >= 0.5) # random assignment of any branch with equal probability
                
                out_spec_modes[indexes_change, 1] = self.degeneracies[deg_indices[indexes_change], 2]

            out_modes[indexes_spec, :] = out_spec_modes

        if np.any(indexes_diff):
            out_modes[indexes_diff, :] = self.pick_diffuse_modes(col_fac[indexes_diff], phonon)
            
            omega_out[indexes_diff] = phonon.omega[out_modes[indexes_diff, 0], out_modes[indexes_diff, 1]]

            if geometry.subvol_type == 'slice' and self.temp_interp_type in ['nearest', 'linear']:
                T_diff = self.temperature_interpolator(col_pos[indexes_diff, self.slice_axis])
            elif geometry.subvol_type == 'grid' and np.any(geometry.grid == 1):
                x = col_pos[indexes_diff, :].reshape(-1, 3)
                T_diff = self.temperature_interpolator(x[:, geometry.grid != 1])
            else:
                T_diff = self.temperature_interpolator(col_pos[indexes_diff, :])

            n_out[indexes_diff] = phonon.calculate_occupation(T_diff, omega_out[indexes_diff])
        
        return out_modes, n_out, omega_out

    def pick_diffuse_modes(self, col_fac, phonon):
        
        n_p = col_fac.shape[0]
        new_modes = np.zeros((n_p, 2))

        u_facets = np.unique(col_fac)
        
        for facet in u_facets:

            i_p = np.arange(n_p)[col_fac == facet] # particles GENERAL indices, int

            i_f = np.nonzero(self.rough_facets == facet)[0][0] # get rough facet index
            
            r = np.random.rand(i_p.shape[0])*self.creation_roulette[i_f, -1]

            flat_i = np.searchsorted(self.creation_roulette[i_f, :], r)

            new_q  = np.floor(flat_i/phonon.number_of_branches).astype(int)
            new_j = flat_i - new_q*phonon.number_of_branches

            new_modes[i_p, 0] = np.copy(new_q)
            new_modes[i_p, 1] = np.copy(new_j)

        new_modes = new_modes.astype(int)

        return new_modes

    def find_degeneracies(self, phonon):
        
        possible_degen = np.absolute(np.transpose(np.expand_dims(phonon.omega, 2), (1, 0, 2)) - phonon.omega) < 1e-10 # same k and omega
        possible_degen[np.arange(phonon.number_of_branches), :, np.arange(phonon.number_of_branches)] = False         # removing self references

        deg_modes = np.vstack(possible_degen.nonzero()).T # getting q and j
        deg_modes = deg_modes[:, [1, 0, 2]]               # flipping to make it as [q, j1, j2]

        deg_modes = deg_modes[np.lexsort(keys = (deg_modes[:, 2], deg_modes[:, 1], deg_modes[:, 0])), :] # sorting just to keep it neat

        # removing duplicates
        i = 0
        n = deg_modes.shape[0]
        while i < n:
            deg_modes = deg_modes[~np.all(deg_modes == deg_modes[i, [0, 2, 1]], axis = 1), :]
            i+=1
            n = deg_modes.shape[0]
        
        self.degeneracies = np.copy(deg_modes) # saving unique degeneracies
        
        # saving a list of reference index so there's no need to search for them later
        self.degen_index = -np.ones(phonon.omega.shape)
        self.degen_index[deg_modes[:, 0], deg_modes[:, 1]] = np.arange(deg_modes.shape[0])
        self.degen_index[deg_modes[:, 0], deg_modes[:, 2]] = np.arange(deg_modes.shape[0])

    def find_specular_correspondences(self, geo, phonon):
        
        self.scat_model = self.args.bound_scat[0]

        facets = self.rough_facets
        normals = -np.round(geo.facets_normal[facets, :], decimals = 10)
        normals, inv_normals = np.unique(normals, axis = 0, return_inverse = True)

        k = phonon.wavevectors
        v = phonon.group_vel
        
        # array of INCOMING modes that CAN be specularly reflected to other mode - Initially all false
        true_spec = np.zeros((facets.shape[0], phonon.number_of_qpoints, phonon.number_of_branches), dtype = bool)

        correspondent_modes = np.zeros((0, 7))

        if self.scat_model in ['k', 'wavevector', 'wave_vector']:
            tol = phonon.q_to_k(np.absolute(1/(2*phonon.data_mesh)))

            near_k_func = NearestNDInterpolator(k, np.arange(phonon.number_of_qpoints)) # nearest interpolator function on K space
            
            for i_n, n in enumerate(normals):
                v_dot_n = np.sum(v*n, axis = 2) 
                s_in  = v_dot_n < 0                      # modes coming to the facet
                s_out = v_dot_n > 0                      # available modes going out of the facet

                active_k = np.any(s_in, axis = 1) # bool, (Q,) - wavevectors that can arrive to the facet

                k_try   = k[active_k, :] - 2*n*np.sum(k[active_k, :]*n, axis = 1, keepdims = True) # reflect them specularly
                _, disp = phonon.find_min_k(k_try, return_disp = True)                                             # check if thay stay in the FBZ

                active_k[active_k] = np.all(disp == 0, axis = 1) # normal processes can be specular
                
                # recalculating reflections of the ones that remained active (redundant but safe)
                k_try = k[active_k, :] - 2*n*np.sum(k[active_k, :]*n, axis = 1, keepdims = True) # (Qa, 3)

                # checking k availability
                q_near = near_k_func(k_try)             # nearest qpoint to k_try
                k_near = phonon.wavevectors[q_near, :]  # get the nearest k vector in relation to k_try
                k_dist = np.absolute(k_try - k_near)    # calculate the distance between the two in each dimension
                
                # it should be a wavevector with at least one valid velocity and within grid tolerance
                in_tol = np.logical_and(np.any(s_out[q_near, :], axis = 1), np.all(k_dist < tol, axis = 1))
                
                active_k[active_k] = in_tol # update the active to only those within tolerance
                
                # recalculating reflections of the ones that remained active (redundant but safe)
                k_try = k[active_k, :] - 2*n*np.sum(k[active_k, :]*n, axis = 1, keepdims = True) # (Qa, 3)
                
                out_q = near_k_func(k_try)                            # out qpoints (Qa,)
                in_q  = np.arange(phonon.number_of_qpoints)[active_k] # in  qpoints (Qa,)

                is_in = s_in[in_q, :]
                is_out = np.transpose(np.expand_dims(s_out[out_q, :], 0), (2, 1, 0))

                valid_v = np.logical_and(is_in, is_out)

                # valid reflections only with the same reflected velocity
                active_k[active_k] = np.any(valid_v, axis = (0,2))

                valid_v = valid_v[:, np.any(valid_v, axis = (0, 2)), :]

                # recalculating reflections of the ones that remained active (redundant but safe)
                k_try = k[active_k, :] - 2*n*np.sum(k[active_k, :]*n, axis = 1, keepdims = True) # (Qa, 3)
                out_q = near_k_func(k_try)                            # out qpoints (Qa,)
                in_q  = np.arange(phonon.number_of_qpoints)[active_k] # in  qpoints (Qa,)

                in_delta  = np.sum(np.absolute(v[ in_q, :, :])*np.expand_dims(tol, (0, 1)), axis = 2) # (Qa, J)
                out_delta = np.sum(np.absolute(v[out_q, :, :])*np.expand_dims(tol, (0, 1)), axis = 2) # (Qa, J)

                in_omega  = phonon.omega[ in_q, :]
                out_omega = phonon.omega[out_q, :]
                
                in_uplim    = in_omega  + in_delta
                in_downlim  = in_omega  - in_delta

                out_uplim   = out_omega + out_delta
                out_downlim = out_omega - out_delta

                out_uplim   = np.transpose(np.expand_dims(out_uplim  , 0), axes = (2, 1, 0)) # (J, Qa, 1)
                out_downlim = np.transpose(np.expand_dims(out_downlim, 0), axes = (2, 1, 0)) # (J, Qa, 1)

                overlap_range = np.where(in_uplim < out_uplim, in_uplim, out_uplim) - np.where(in_downlim > out_downlim, in_downlim, out_downlim) # (J, Qa, J)
                overlap = overlap_range > 0
                
                omega_diff = np.absolute((in_omega - np.transpose(np.expand_dims(out_omega  , 0), axes = (2, 1, 0)))/in_omega) # (J, Qa, J)
                omega_diff = np.where(overlap, omega_diff, np.inf) # (J, Qa, J)

                # valid_corresp = np.logical_and(overlap, same_v) # correspondences with valid velocities and frequency intervals
                valid_corresp = np.logical_and(overlap, valid_v) # correspondences with valid velocities and frequency intervals

                valid_k = np.any(valid_corresp, axis = (0, 2))

                active_k[active_k] = valid_k

                valid_corresp = valid_corresp[:, valid_k, :]

                omega_diff = omega_diff[:, valid_k, :]

                omega_diff = np.where(valid_corresp, omega_diff, np.inf) # updating valid differences

                min_diff = np.amin(omega_diff, axis = 0) # (Qa, J) - minimum relative omega differences

                min_diff_branch = np.where(np.any(valid_corresp, axis = 0), np.argmax(omega_diff == min_diff, axis = 0), -1).astype(int)

                k_try = k[active_k, :] - 2*n*np.sum(k[active_k, :]*n, axis = 1, keepdims = True) # (Qa, 3)
                
                in_q = np.arange(phonon.number_of_qpoints)[active_k] # in qpoints
                out_q = near_k_func(k_try)                           # out qpoints (Qa,)

                i_f = inv_normals == i_n

                for i, q in enumerate(in_q):
                    for j in range(phonon.number_of_branches):
                        if min_diff_branch[i, j] == -1:
                            true_spec[i_f, q, j] = False
                        else:
                            true_spec[i_f, q, j] = True
                
                i_q_in, spec_j_in = np.nonzero(min_diff_branch != -1) # specular active in q index, specular in branch
                spec_q_in = in_q[i_q_in]                              # specular in q (true q index)

                spec_q_out = out_q[i_q_in]                            # specular out q
                spec_j_out = min_diff_branch[i_q_in, spec_j_in]       # specular out j

                n_ts = i_q_in.shape[0]

                fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (15, 15), dpi = 300, sharex = 'row', sharey = 'row')
                
                b = np.eye(3)
                is_normal = 1 - np.sum(b*np.absolute(n), axis = 1) < 1e-3

                if np.any(is_normal):
                    b = b[~is_normal, :]

                # if b1 is parallel to b1
                b1 = b[0, :]
                b1 = b1 - n*np.sum(n*b1)  # make b1 orthogonal to the normal
                b1 = b1/np.sum(b1**2)**0.5          # normalise b1

                b2 = np.cross(n, b1)           # generate b2 = n x b1
                
                k_b1_in = np.sum(phonon.wavevectors[spec_q_in, :]*b1, axis = 1)
                k_b1_out = np.sum(phonon.wavevectors[spec_q_out, :]*b1, axis = 1)
                v_b1_in = np.sum(phonon.group_vel[spec_q_in, spec_j_in, :]*b1, axis = 1)
                v_b1_out = np.sum(phonon.group_vel[spec_q_out, spec_j_out, :]*b1, axis = 1)

                k_b2_in = np.sum(phonon.wavevectors[spec_q_in, :]*b2, axis = 1)
                k_b2_out = np.sum(phonon.wavevectors[spec_q_out, :]*b2, axis = 1)
                v_b2_in = np.sum(phonon.group_vel[spec_q_in, spec_j_in, :]*b2, axis = 1)
                v_b2_out = np.sum(phonon.group_vel[spec_q_out, spec_j_out, :]*b2, axis = 1)

                k_n_in = np.sum(phonon.wavevectors[spec_q_in, :]*n, axis = 1)
                k_n_out = np.sum(phonon.wavevectors[spec_q_out, :]*n, axis = 1)
                v_n_in = np.sum(phonon.group_vel[spec_q_in, spec_j_in, :]*n, axis = 1)
                v_n_out = np.sum(phonon.group_vel[spec_q_out, spec_j_out, :]*n, axis = 1)

                omega_in  = phonon.omega[spec_q_in , spec_j_in ]
                omega_out = phonon.omega[spec_q_out, spec_j_out]

                ax[0, 0].scatter(k_n_in, k_n_out, s = 1)
                ax[1, 0].scatter(v_n_in, v_n_out, s = 1)

                ax[0, 1].scatter(k_b1_in, k_b1_out, s = 1)
                ax[1, 1].scatter(v_b1_in, v_b1_out, s = 1)

                ax[0, 2].scatter(k_b2_in, k_b2_out, s = 1)
                ax[1, 2].scatter(v_b2_in, v_b2_out, s = 1)

                ax[2, 1].scatter(omega_in, omega_out, s = 1)

                v_list = [r'$\vec{{n}}$', r'$\vec{{b}}_{{1}}$', r'$\vec{{b}}_{{2}}$']

                for i in range(3):
                    ax[0, i].set_xlabel(r'$\vec{{k}}_{{in}}\cdot$' + v_list[i])
                    ax[0, i].set_ylabel(r'$\vec{{k}}_{{out}}\cdot$' + v_list[i])
                    ax[1, i].set_xlabel(r'$\vec{{v}}_{{in}}\cdot$' + v_list[i])
                    ax[1, i].set_ylabel(r'$\vec{{v}}_{{out}}\cdot$' + v_list[i])
                
                ax[2, 1].set_xlabel('$\omega_{in}$')
                ax[2, 1].set_ylabel('$\omega_{out}$')

                ax[2, 0].axis('off')
                ax[2, 2].axis('off')
                
                for a in ax.ravel():
                    a.yaxis.set_tick_params(labelleft=True)

                plt.suptitle('Normal = <{:.2f}, {:.2f}, {:.2f}>. Angles = {:.2f}, {:.2f}, {:.2f}'.format(n[0], n[1], n[2], 
                                                                                                    np.arccos(n[0])*180/np.pi,
                                                                                                    np.arccos(n[1])*180/np.pi,
                                                                                                    np.arccos(n[2])*180/np.pi))

                plt.tight_layout()
                plt.savefig(os.path.join(self.results_folder_name, f'spec{n}.png'))
                plt.close(fig)

                correspondent_modes = np.vstack((correspondent_modes, np.vstack((np.ones(n_ts)*n.reshape(-1, 1), spec_q_in, spec_j_in, spec_q_out, spec_j_out)).T))
        
        elif self.scat_model in ['v', 'vel', 'velocity', 'groupvel', 'group_vel']:
            
            k_grid = phonon.q_to_k(np.absolute(1/(2*phonon.data_mesh))) # size of the grid in each direction

            delta_omega = np.sum((phonon.group_vel*k_grid)**2, axis = 2)**0.5 # the acceptable variation of omega from the central value for each mode

            for i_n, n in enumerate(normals):
                gc.collect()
                v_dot_n = np.sum(v*n, axis = 2) 
                s_in  = v_dot_n < 0                      # modes coming to the facet
                s_out = v_dot_n > 0                      # available modes going out of the facet

                in_modes = np.vstack(s_in.nonzero()).T
                out_modes = np.vstack(s_out.nonzero()).T
                del s_in, s_out
                
                omega_in  = phonon.omega[ in_modes[:, 0],  in_modes[:, 1]]
                omega_out = phonon.omega[out_modes[:, 0], out_modes[:, 1]]

                delta_omega_in  = delta_omega[ in_modes[:, 0],  in_modes[:, 1]]
                delta_omega_out = delta_omega[out_modes[:, 0], out_modes[:, 1]]

                v_in = phonon.group_vel[in_modes[:, 0], in_modes[:, 1], :]
                v_in = v_in - 2*n*(np.sum(v_in*n, axis = 1, keepdims = True))

                v_out = phonon.group_vel[out_modes[:, 0], out_modes[:, 1], :]

                v_in_norm = np.linalg.norm(v_in, axis = 1)
                v_out_norm = np.linalg.norm(v_out, axis = 1)

                crit = 1e-3
                
                # sorted vx
                sorted_i_out = np.argsort(v_out[:, 0])
                sorted_vx_out = v_out[sorted_i_out, 0]
                
                sorted_i_in = np.argsort(v_in[:, 0])
                sorted_vx_in = v_in[sorted_i_in, 0]

                sorted_norm_out = v_out_norm[sorted_i_out]
                sorted_norm_in = v_in_norm[sorted_i_in]

                right_i = np.searchsorted(sorted_vx_out, sorted_vx_in, side = 'right')
                # going to the right
                far_right = right_i == sorted_vx_out.shape[0]

                norm = np.fmax(sorted_norm_out[right_i[~far_right]], sorted_norm_in[~far_right])

                far_right[~far_right] = (sorted_vx_out[right_i[~far_right]] - sorted_vx_in[~far_right])/norm > crit
                done = np.copy(far_right)
                while not np.all(done):
                    done[~done] = right_i[~done] == sorted_vx_out.shape[0]-1 # check the ones not done are at the border and mark them as done

                    norm = np.fmax(sorted_norm_out[right_i[~done] + 1], sorted_norm_in[~done])
                    
                    take_step = (sorted_vx_out[right_i[~done] + 1] - sorted_vx_in[~done])/norm < crit # check if the ones not done can take a step
                    
                    ind = (~done).nonzero()[0][take_step] # indices of the ones that will walk

                    done[~done] = ~take_step # the ones that won't walk are done

                    right_i[ind] += 1
                
                left_i = np.searchsorted(sorted_vx_out, sorted_vx_in, side = 'left')-1
                # going to the left
                far_left = left_i == -1

                norm = np.fmax(sorted_norm_out[left_i[~far_left]-1], sorted_norm_in[~far_left])

                far_left[~far_left] = (sorted_vx_in[~far_left] - sorted_vx_out[left_i[~far_left]])/norm > crit
                done = np.copy(far_left)
                while not np.all(done):
                    done[~done] = left_i[~done] == 0 # check the ones not done are at the border and mark them as done

                    norm = np.fmax(sorted_norm_out[left_i[~done]-1], sorted_norm_in[~done])
                    
                    take_step = (sorted_vx_in[~done] - sorted_vx_out[left_i[~done] - 1])/norm < crit # check if the ones not done can take a step
                    
                    ind = (~done).nonzero()[0][take_step] # indices of the ones that will walk

                    done[~done] = ~take_step # the ones that won't walk are done

                    left_i[ind] -= 1

                # saving the possible reflections in vx
                right_i[far_right] -= 1
                left_i[far_left] += 1
                
                L = right_i - left_i + 1 # how many possible reflections for each incoming mode
                
                possible_reflections = np.zeros((L.sum(), 2), dtype = int) # initialise possible reflections array
                N = 0 # initialise position
                for i in range(sorted_vx_in.shape[0]): # for each SORTED vx_in
                    if L[i] > 0:
                        possible_reflections[N:N+L[i], 0] = sorted_i_in[i]
                        possible_reflections[N:N+L[i], 1] = sorted_i_out[left_i[i]:right_i[i]+1]
                        N += L[i]
                
                ref_norm = np.fmax(v_in_norm[possible_reflections[:, 0]], v_out_norm[possible_reflections[:, 1]])
                # DIRECTION Y
                diff = np.absolute(v_in[possible_reflections[:, 0], 1] - v_out[possible_reflections[:, 1], 1])
                possible_reflections = possible_reflections[diff/ref_norm < crit, :]
                ref_norm = ref_norm[diff/ref_norm < crit]
                
                # DIRECTION Z
                diff = np.absolute(v_in[possible_reflections[:, 0], 2] - v_out[possible_reflections[:, 1], 2])
                possible_reflections = possible_reflections[diff/ref_norm < crit, :]

                diff = np.absolute(omega_in[possible_reflections[:, 0]] - omega_out[possible_reflections[:, 1]])
                delta = delta_omega_in[possible_reflections[:, 0]] + delta_omega_out[possible_reflections[:, 1]]

                possible_reflections = possible_reflections[diff < delta, :]

                in_modes  =  in_modes[possible_reflections[:, 0], :]
                out_modes = out_modes[possible_reflections[:, 1], :]

                v_in  = v[ in_modes[:, 0],  in_modes[:, 1], :]
                v_out = v[out_modes[:, 0], out_modes[:, 1], :]

                v_in  /= np.linalg.norm(v_in , axis = 1, keepdims = True)
                v_out /= np.linalg.norm(v_out, axis = 1, keepdims = True)
                
                v_try = v_in - 2*n*np.sum(v_in*n, axis = 1, keepdims = True) # reflect

                angle = np.arccos(np.sum(v_try*v_out, axis = 1))
                angle[np.isnan(angle)] = np.pi

                in_modes  =  in_modes[angle < crit, :]
                out_modes = out_modes[angle < crit, :]
                
                spec_q_in = in_modes[:, 0]           # in qpoints
                spec_q_out = out_modes[:, 0] # out qpoints (Qa,)

                spec_j_in = in_modes[:, 1]           # in qpoints
                spec_j_out = out_modes[:, 1] # out qpoints (Qa,)

                i_f = inv_normals == i_n

                for i in in_modes:
                    true_spec[i_f, i[0], i[1]] = True
                
                n_ts = spec_q_in.shape[0]

                fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (15, 15), dpi = 100, sharex = 'row', sharey = 'row')
                
                b = np.eye(3)
                is_normal = 1 - np.sum(b*np.absolute(n), axis = 1) < 1e-3

                if np.any(is_normal):
                    b = b[~is_normal, :]

                # if b1 is parallel to b1
                b1 = b[0, :]
                b1 = b1 - n*np.sum(n*b1)  # make b1 orthogonal to the normal
                b1 = b1/np.sum(b1**2)**0.5          # normalise b1

                b2 = np.cross(n, b1)           # generate b2 = n x b1
                
                k_b1_in = np.sum(phonon.wavevectors[spec_q_in, :]*b1, axis = 1)
                k_b1_out = np.sum(phonon.wavevectors[spec_q_out, :]*b1, axis = 1)
                v_b1_in = np.sum(phonon.group_vel[spec_q_in, spec_j_in, :]*b1, axis = 1)
                v_b1_out = np.sum(phonon.group_vel[spec_q_out, spec_j_out, :]*b1, axis = 1)

                k_b2_in = np.sum(phonon.wavevectors[spec_q_in, :]*b2, axis = 1)
                k_b2_out = np.sum(phonon.wavevectors[spec_q_out, :]*b2, axis = 1)
                v_b2_in = np.sum(phonon.group_vel[spec_q_in, spec_j_in, :]*b2, axis = 1)
                v_b2_out = np.sum(phonon.group_vel[spec_q_out, spec_j_out, :]*b2, axis = 1)

                k_n_in = np.sum(phonon.wavevectors[spec_q_in, :]*n, axis = 1)
                k_n_out = np.sum(phonon.wavevectors[spec_q_out, :]*n, axis = 1)
                v_n_in = np.sum(phonon.group_vel[spec_q_in, spec_j_in, :]*n, axis = 1)
                v_n_out = np.sum(phonon.group_vel[spec_q_out, spec_j_out, :]*n, axis = 1)

                omega_in  = phonon.omega[spec_q_in , spec_j_in ]
                omega_out = phonon.omega[spec_q_out, spec_j_out]

                ax[0, 0].scatter(k_n_in, k_n_out, s = 1)
                ax[1, 0].scatter(v_n_in, v_n_out, s = 1)

                ax[0, 1].scatter(k_b1_in, k_b1_out, s = 1)
                ax[1, 1].scatter(v_b1_in, v_b1_out, s = 1)

                ax[0, 2].scatter(k_b2_in, k_b2_out, s = 1)
                ax[1, 2].scatter(v_b2_in, v_b2_out, s = 1)

                ax[2, 1].scatter(omega_in, omega_out, s = 1)

                v_list = [r'$\vec{{n}}$', r'$\vec{{b}}_{{1}}$', r'$\vec{{b}}_{{2}}$']

                for i in range(3):
                    ax[0, i].set_xlabel(r'$\vec{{k}}_{{in}}\cdot$' + v_list[i])
                    ax[0, i].set_ylabel(r'$\vec{{k}}_{{out}}\cdot$' + v_list[i])
                    ax[1, i].set_xlabel(r'$\vec{{v}}_{{in}}\cdot$' + v_list[i])
                    ax[1, i].set_ylabel(r'$\vec{{v}}_{{out}}\cdot$' + v_list[i])
                
                ax[2, 1].set_xlabel('$\omega_{in}$')
                ax[2, 1].set_ylabel('$\omega_{out}$')

                ax[2, 0].axis('off')
                ax[2, 2].axis('off')

                for a in ax.ravel():
                    a.yaxis.set_tick_params(labelleft=True)

                plt.suptitle('Normal = <{:.2f}, {:.2f}, {:.2f}>. Angles = {:.2f}, {:.2f}, {:.2f}'.format(n[0], n[1], n[2], 
                                                                                                    np.arccos(n[0])*180/np.pi,
                                                                                                    np.arccos(n[1])*180/np.pi,
                                                                                                    np.arccos(n[2])*180/np.pi))

                plt.tight_layout()
                plt.savefig(os.path.join(self.results_folder_name, f'spec{n}.png'))
                plt.close(fig)

                correspondent_modes = np.vstack((correspondent_modes, np.vstack((np.ones(n_ts)*n.reshape(-1, 1), spec_q_in, spec_j_in, spec_q_out, spec_j_out)).T))

        self.correspondent_modes = correspondent_modes
        self.specular_function   = NearestNDInterpolator(self.correspondent_modes[:, :-2], self.correspondent_modes[:, -2:])
        self.true_specular       = copy.copy(true_spec)
        self.specularity         = self.true_specular.astype(int)*self.specularity

        np.savetxt(os.path.join(self.results_folder_name, 'specular_correspondences.txt'), self.correspondent_modes, fmt = '%.3f %.3f %.3f %d %d %d %d')

    def periodic_boundary_condition(self, positions, group_velocities, collision_facets, collision_positions, calculated_ts, geometry, indexes_del_extra):
        
        collision_facets = collision_facets.astype(int) # ensuring collision facets are integers

        # getting which face is connected
        rows             = np.array([np.where(i == self.connected_facets)[0][0] for i in collision_facets])  
        mask             = self.connected_facets[rows, :] == collision_facets.reshape(-1, 1)
        connected_facets = (self.connected_facets[rows, :]*~mask).sum(axis = 1).astype(int)

        previous_positions        = copy.deepcopy(positions)               # the path of some particles is calculated from their current position
        first                     = np.nonzero(calculated_ts == 0)[0]      # but for those at the beginning of the timestep
        previous_positions[first, :] -= group_velocities[first, :]*self.dt # their starting position is the one before the collision

        L = (geometry.facet_centroid[connected_facets, :] - geometry.facet_centroid[collision_facets, :]) # get translation vector between connected facets
        new_positions = collision_positions + L                                                     # translate particles

        # finding next scattering event
        new_ts_to_boundary, new_collision_facets, new_collision_positions, indexes_del_extra = self.timesteps_to_boundary(new_positions, group_velocities, geometry, indexes_del_extra)

        calculated_ts += np.linalg.norm((collision_positions - previous_positions), axis = 1)/np.linalg.norm((group_velocities*self.dt), axis = 1)

        new_collision_facets[indexes_del_extra] = 0
        new_collision_facets = new_collision_facets.astype(int)
            
        new_collision_cond = self.bound_cond[new_collision_facets]

        return new_positions, new_ts_to_boundary, new_collision_facets, new_collision_positions, new_collision_cond, calculated_ts, indexes_del_extra

    def roughness_boundary_condition(self, positions,
                                           group_velocities,
                                           collision_facets,
                                           collision_positions,
                                           calculated_ts,
                                           in_modes, 
                                           occupation_in,
                                           omega_in,
                                           geometry,
                                           phonon,
                                           indexes_del_extra):
        
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
        new_modes, new_occupation, new_omega = self.select_reflected_modes(in_modes, collision_facets, collision_positions, occupation_in, omega_in, geometry, phonon)

        # update v
        new_group_vel  = phonon.group_vel[new_modes[:, 0], new_modes[:, 1], :]

        # find next scattering event
        (new_ts_to_boundary     ,
         new_collision_facets   ,
         new_collision_positions,
         indexes_del_extra      ) = self.timesteps_to_boundary(new_positions, new_group_vel, geometry, indexes_del_extra)
        
        new_collision_cond = np.empty(new_collision_facets.shape, dtype = str)
        new_collision_cond[~indexes_del_extra] = self.bound_cond[new_collision_facets[~indexes_del_extra].astype(int)]
        
        return (new_modes,
                new_positions,
                new_group_vel,
                new_omega,
                new_occupation,
                new_ts_to_boundary,
                new_calculated_ts,
                new_collision_positions,
                new_collision_facets,
                new_collision_cond,
                indexes_del_extra)

    def boundary_scattering(self, geometry, phonon):
        '''Applies boundary scattering or other conditions to the particles where it happened, given their indexes.'''

        self.subvol_id = self.get_subvol_id(self.positions, geometry) # getting subvol

        indexes_all = self.n_timesteps < 0                      # find scattering particles (N_p,)

        indexes_del_extra = np.zeros(indexes_all.shape, dtype = bool) # particles to be deleted due to errors and such

        calculated_ts              = np.ones(indexes_all.shape) # start the tracking of calculation as 1 (N_p,)
        calculated_ts[indexes_all] = 0                          # set it to 0 for scattering particles   (N_p,)

        new_n_timesteps = copy.copy(self.n_timesteps) # (N_p,)

        self.N_leaving = np.zeros(self.n_of_reservoirs, dtype = int)

        # while there are any particles with the timestep not completely calculated
        while np.any(calculated_ts < 1):

            # I. Deleting particles entering reservoirs

            # identifying particles hitting rough facets
            indexes_del = np.logical_and(calculated_ts       <   1 ,
                                         np.in1d(self.collision_cond, ['T', 'F']))
            indexes_del = np.logical_and(indexes_del, 
                                         (1-calculated_ts) > new_n_timesteps)
            indexes_del = np.logical_or(indexes_del, indexes_del_extra)

            if np.any(indexes_del):
                rand_res_index = np.random.rand(0, self.n_of_reservoirs, indexes_del_extra.sum()) # indexes_del_extra will be regenerated randomly on the reservoirs

                if self.n_of_reservoirs > 0: # if there are any reservoirs
                    
                    for i in range(self.n_of_reservoirs): # for each one

                        facet = self.res_facet[i]         # get their facet index

                        indexes_res = self.collision_facets[indexes_del] == facet # get which particles are leaving through it

                        self.N_leaving[i] += int(indexes_res.sum())
                        self.N_leaving[i] += int((rand_res_index == i).sum())

                        # subtracting energy
                        if self.T_reference == 'local':
                            dn = self.occupation[indexes_del][indexes_res] - phonon.calculate_occupation(self.res_facet_temperature[i], self.omega[indexes_del][indexes_res])
                        else:
                            dn = self.occupation[indexes_del][indexes_res] - self.reference_occupation[self.modes[indexes_del, 0][indexes_res], self.modes[indexes_del, 1][indexes_res]]
                        
                        energies = self.hbar*self.omega[indexes_del][indexes_res]*dn
                        self.res_energy_balance[i] -= energies.sum()

                        if self.norm == 'mean':
                            self.res_norm[i] += indexes_res.sum()

                        # adding heat flux
                        hflux = energies.reshape(-1, 1)*self.group_vel[indexes_del, :][indexes_res, :]/np.sum(self.group_vel[indexes_del, :][indexes_res, :]*geometry.facets_normal[facet, :], axis = 1, keepdims = True)
                        self.res_heat_flux[i, :] += hflux.sum(axis = 0)

                self.delete_particles(indexes_del)
                calculated_ts     = calculated_ts[~indexes_del]
                new_n_timesteps   = new_n_timesteps[~indexes_del]
                indexes_all       = indexes_all[~indexes_del]
                indexes_del_extra = indexes_del_extra[~indexes_del]
            
            # II. Applying Periodicities:
            
            # identifying particles hitting facets with periodic boundary condition
            indexes_per = np.logical_and(calculated_ts       <   1 , # 
                                        self.collision_cond == 'P')
            indexes_per = np.logical_and(indexes_per               ,
                                         (1-calculated_ts) > new_n_timesteps)
            indexes_per = np.logical_and(indexes_per, ~indexes_del_extra)

            if np.any(indexes_per):

                # enforcing periodic boundary condition
                (self.positions[indexes_per, :]         ,
                new_n_timesteps[indexes_per]            ,
                self.collision_facets[indexes_per]      ,
                self.collision_positions[indexes_per, :],
                self.collision_cond[indexes_per]        ,
                calculated_ts[indexes_per]              ,
                indexes_del_extra[indexes_per]          ) = self.periodic_boundary_condition(self.positions[indexes_per, :]          ,
                                                                                             self.group_vel[indexes_per, :]          ,
                                                                                             self.collision_facets[indexes_per]      ,
                                                                                             self.collision_positions[indexes_per, :],
                                                                                             calculated_ts[indexes_per]              ,
                                                                                             geometry                                ,
                                                                                             indexes_del_extra[indexes_per]          )

            # III. Performing scattering:

            # identifying particles hitting rough facets
            indexes_ref = np.logical_and(calculated_ts       <   1 ,
                                         self.collision_cond == 'R')
            # print('ref', indexes_ref.sum())
            indexes_ref = np.logical_and(indexes_ref               ,
                                         (1-calculated_ts) > new_n_timesteps)
            # print('ref', indexes_ref.sum())
            indexes_ref = np.logical_and(indexes_ref, ~indexes_del_extra)

            if np.any(indexes_ref):
                (self.modes[indexes_ref, :]              ,
                 self.positions[indexes_ref, :]          ,
                 self.group_vel[indexes_ref, :]          ,
                 self.omega[indexes_ref]                 ,
                 self.occupation[indexes_ref]            ,
                 new_n_timesteps[indexes_ref]            ,
                 calculated_ts[indexes_ref]              ,
                 self.collision_positions[indexes_ref, :],
                 self.collision_facets[indexes_ref]      ,
                 self.collision_cond[indexes_ref]        ,
                 indexes_del_extra[indexes_ref]          ) = self.roughness_boundary_condition(self.positions[indexes_ref, :]          ,
                                                                                               self.group_vel[indexes_ref, :]          ,
                                                                                               self.collision_facets[indexes_ref]      ,
                                                                                               self.collision_positions[indexes_ref, :],
                                                                                               calculated_ts[indexes_ref]              ,
                                                                                               self.modes[indexes_ref, :]              , 
                                                                                               self.occupation[indexes_ref]            ,
                                                                                               self.omega[indexes_ref]                 ,
                                                                                               geometry                                ,
                                                                                               phonon                                  ,
                                                                                               indexes_del_extra[indexes_ref])

            # IV. Drifting those who will not be scattered again in this timestep

            # identifying drifting particles
            indexes_drift = np.logical_and(calculated_ts < 1                  ,
                                           (1-calculated_ts) < new_n_timesteps)
            indexes_drift = np.logical_and(indexes_drift, ~indexes_del_extra)

            if np.any(indexes_drift):
                self.positions[indexes_drift, :] += self.group_vel[indexes_drift, :]*self.dt*(1-calculated_ts[indexes_drift]).reshape(-1, 1)

                new_n_timesteps[indexes_drift]   -= (1-calculated_ts[indexes_drift])
                calculated_ts[indexes_drift]     = 1
            
        self.n_timesteps = copy.copy(new_n_timesteps)
    
    def adjust_reservoir_balance(self, geometry, phonon):
        if self.n_of_reservoirs > 0:
            
            self.res_heat_flux *= phonon.number_of_active_modes/(self.particle_density*self.dt*self.n_dt_to_conv*geometry.facets_area[self.res_facet].reshape(-1, 1))
            self.res_heat_flux  = phonon.normalise_to_density(self.res_heat_flux)
            self.res_heat_flux *= self.eVpsa2_in_Wm2 # convert eV/a²ps to W/m²
                       
            self.res_energy_balance *= phonon.number_of_active_modes/(self.particle_density*self.dt*self.n_dt_to_conv)
            self.res_energy_balance = phonon.normalise_to_density(self.res_energy_balance) # eV/ps            

    def restart_reservoir_balance(self):
        self.res_heat_flux = np.zeros((self.n_of_reservoirs, 3))
        self.res_energy_balance = np.zeros(self.n_of_reservoirs)
        if self.norm == 'mean':
            self.res_norm = np.zeros(self.n_of_reservoirs)

    def lifetime_scattering(self, phonon):
        '''Performs lifetime scattering.'''
        
        Tqj = np.hstack((self.temperatures.reshape(-1, 1), self.modes))
        tau = phonon.lifetime_function(Tqj)

        n0 = phonon.calculate_occupation(self.temperatures, self.omega) # calculating Bose-Einstein occcupation
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'): # this case is treated inside np.where, but the warning still keeps showing up
            self.occupation = np.where(tau > 0, n0 + (self.occupation - n0)*np.exp(-self.dt/tau), n0) # n = n0 + (n - n0(T)) e^(-dt/tau(T))

    def contains_check(self, geometry):
        '''Eventually, some particles will escape the geometry because of numerical uncertainties.
           This tries to put them back on the geometry.'''

        out = np.logical_or(np.any(self.positions < geometry.bounds[0, :]-1e-10, axis = 1),
                            np.any(self.positions > geometry.bounds[1, :]+1e-10, axis = 1)).nonzero()[0]

        if out.shape[0]>0:
            self.positions[out, :] = geometry.mesh.sample_volume(out.shape[0])
            self.n_timesteps[out], self.collision_facets[out], self.collision_positions[out] = self.timesteps_to_boundary(self.positions[out, :], self.group_vel[out, :], geometry)
            self.collision_cond[out] = self.get_collision_condition(self.collision_facets[out])

    def run_timestep(self, geometry, phonon):
        
        if self.current_timestep == 0:
            print('Simulating...')

        if (self.current_timestep % 100) == 0:
            self.write_final_state(geometry)
            self.view.update_population(self, verbose = False)
            self.view.postprocess(verbose = False)
            self.update_residue(geometry)
            self.contains_check(geometry)
            self.plot_figures(geometry, phonon, property_plot = self.args.fig_plot, colormap = self.args.colormap[0])

            info ='Timestep {:>5d} - max residue: {:>9.3e} ({:<9s}) ['.format(int(self.current_timestep), self.max_residue, self.max_residue_qt)
            for sv in range(self.n_of_subvols):
                info += ' {:>7.3f}'.format(self.subvol_temperature[sv])
            info += ' ]'
            print(info)

        self.drift()                                    # drift particles

        if self.n_of_reservoirs > 0:
            if self.res_gen in ['fixed_rate', 'constant']:
                self.fill_reservoirs(geometry, phonon)          # refill reservoirs
            elif self.res_gen == 'one_to_one':
                self.fill_reservoirs(geometry, phonon, n_leaving = self.N_leaving)          # refill reservoirs
            self.add_reservoir_particles(geometry)  # add reservoir particles that come in the domain

        self.boundary_scattering(geometry, phonon)      # perform boundary scattering/periodicity and particle deletion

        self.refresh_temperatures(geometry, phonon)     # refresh cell temperatures
        
        self.lifetime_scattering(phonon)                # perform lifetime scattering

        self.current_timestep += 1                      # +1 timestep index

        self.t = self.current_timestep*self.dt          # 1 dt passed
        
        if ( self.current_timestep % self.n_dt_to_conv) == 0:
            self.subvol_heat_flux = self.calculate_heat_flux(geometry, phonon)
            self.calculate_kappa(geometry)
            self.adjust_reservoir_balance(geometry, phonon)
            self.write_convergence(geometry)      # write data on file
            self.restart_reservoir_balance()

        gc.collect() # garbage collector

    def initialise_residue(self, geo):
        if geo.subvol_type == 'slice':
            self.old_mean_large = np.ones(3*self.n_of_subvols+self.n_of_reservoirs)
            self.old_std_large  = np.ones(3*self.n_of_subvols+self.n_of_reservoirs)
        else:
            self.old_mean_large = np.ones(4*self.n_of_subvols+self.n_of_reservoirs+geo.n_of_subvol_con)
            self.old_std_large  = np.ones(4*self.n_of_subvols+self.n_of_reservoirs+geo.n_of_subvol_con)

        # self.residue_all = np.ones(8)
        self.conv_count = 0
        self.finish_sim = False
        self.max_residue    = 1
        self.max_residue_qt = 'none'

        if geo.subvol_type == 'slice':
            ax_str = ['x', 'y', 'z'][self.slice_axis]
            self.residue_qts   = ['T_{:d}'.format(i) for i in range(self.n_of_subvols)] + \
                                 ['phi_{:s}_{:d}'.format(ax_str, j) for j in range(self.n_of_subvols)] + \
                                 ['en_res_{:d}'.format(i) for i in range(self.n_of_reservoirs)] +\
                                 ['k_{:d}'.format(i) for i in range(self.n_of_subvols)]
        else:
            self.residue_qts   = ['T_{:d}'.format(i) for i in range(self.n_of_subvols)] + \
                                 ['phi_{:s}_{:d}'.format(i, j) for j in range(self.n_of_subvols) for i in ['x', 'y', 'z']] + \
                                 ['en_res_{:d}'.format(i) for i in range(self.n_of_reservoirs)] +\
                                 ['k_{:d}'.format(i) for i in range(geo.n_of_subvol_con)]

    def update_residue(self, geo):
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

            if geo.subvol_type == 'slice':
                new_mean_large = np.concatenate((self.view.mean_T, self.view.mean_sv_phi[3*np.arange(self.n_of_subvols)+self.slice_axis], self.view.mean_en_res, self.view.mean_sv_k))
                new_std_large  = np.concatenate((self.view.std_T , self.view.std_sv_phi[3*np.arange(self.n_of_subvols)+self.slice_axis] , self.view.std_en_res , self.view.std_sv_k ))
            else:
                new_mean_large = np.concatenate((self.view.mean_T, self.view.mean_sv_phi, self.view.mean_en_res, self.view.mean_con_k))
                new_std_large  = np.concatenate((self.view.std_T , self.view.std_sv_phi , self.view.std_en_res , self.view.std_con_k ))
            
            residue_mean = np.absolute((new_mean_large - self.old_mean_large)/self.old_mean_large)

            # residue_std = np.absolute((new_std_large - self.old_std_large)/self.old_std_large)
        
        self.residue_all = np.where(new_std_large > np.absolute(new_mean_large), 0, residue_mean)

        self.max_residue = np.nanmax(self.residue_all)
        
        index = np.nonzero(self.residue_all == self.max_residue)[0][0]
        self.max_residue_qt = self.residue_qts[index]

         # if any is larger than criterion, not converged
        if self.max_residue < self.conv_crit:
            self.conv_count += 1
        else:
            self.conv_count = 0

        if self.conv_count >= self.conv_count_min:
            self.finish_sim = True

        # update previous values
        self.old_mean_large = new_mean_large
        self.old_std_large = new_std_large

        s = ''
        # for i in np.concatenate((residue_mean, residue_std)):
        for i in self.residue_all:
            s+= '{:9.3e} '.format(i)
        s += '\n'

        with open(os.path.join(self.results_folder_name,'residue.txt'), 'a+') as f:
            f.writelines(s)

    def plot_figures(self, geometry, phonon, property_plot=['energy'], colormap = 'jet'):
        
        fig, ax = geometry.mesh.plot_facet_boundaries(l_color = self.view.ax_style['axiscolor'])
        n = len(property_plot)

        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
        # ax.view_init(elev = 90, azim = -90)
        
        graph = ax.scatter(self.positions[:, 0],
                           self.positions[:, 1],
                           self.positions[:, 2],
                           cmap = colormap     ,
                           c = np.random.rand(self.positions.shape[0]),
                           s = 1               )

        for i in range(n):
            if property_plot[i] in ['T', 'temperature', 'temperatures']:
                figname = 'fig_temperature'
                colors = self.temperatures
                if self.n_of_reservoirs > 0:
                    T = np.concatenate((self.res_bound_values[self.res_bound_cond == 'T'], self.subvol_temperature))
                else:
                    T = self.subvol_temperature
                vmin = np.floor(T.min()-1)
                vmax = np.ceil(T.max()+1)
                label = 'Temperature [K]'
                format = '{:.1f}'
            elif property_plot[i] in ['omega', 'angular_frequency', 'frequency']:
                figname = 'fig_omega'
                colors = self.omega
                order = np.ceil(np.log10(phonon.omega.max()))
                vmin = 0
                vmax = (10**order)*np.ceil(phonon.omega.max()/(10**order))
                label = r'Angular frequency $\omega$ [THz$\cdot$rad]'
                format = '{:.2e}'
            elif property_plot[i] in ['n', 'occupation']:
                figname = 'fig_occupation'
                T = self.subvol_temperature
                dn = self.occupation - phonon.calculate_occupation(T.mean(), self.omega)
                colors = dn

                # using symlog scale
                N_order = 10
                B = np.floor(np.log10(np.absolute(dn).max()))       # B in ( A x 10^B )
                A = np.ceil(np.absolute(dn).max()/(10**B)) # A in ( A x 10^B )
                y_interp = np.linspace(-N_order, N_order, 2*N_order+1)

                x_interp = np.zeros(y_interp.shape[0])
                x_interp[:N_order] = -A*10**(B-np.arange(N_order))
                x_interp[-N_order:] = A*10**(B-np.flip(np.arange(N_order)))

                f = interp1d(x_interp, y_interp, kind = 'cubic', fill_value = (-N_order, N_order), bounds_error=False)

                colors = f(dn)
                vmin = N_order
                vmax = -N_order
                label = r'Occupation number deviation $\delta n$ [phonons]'
                format = '{:.2e}'
            elif property_plot[i] in ['e', 'energy', 'energies']:
                figname = 'fig_energy'
                
                T = self.subvol_temperature
                
                dn = self.occupation - phonon.calculate_occupation(T.mean(), self.omega)
                colors = self.hbar*self.omega*dn

                emin = (phonon.calculate_energy(T.min()-1, phonon.omega) - phonon.calculate_energy(T.mean(), phonon.omega)).min()
                emax = (phonon.calculate_energy(T.max()+1, phonon.omega) - phonon.calculate_energy(T.mean(), phonon.omega)).max()

                order = [np.floor( np.log10(np.absolute(emin)) ), np.floor( np.log10(np.absolute(emax)) )]
                vmin = (10**order[0])*np.ceil(emin/(10**order[0]))
                vmax = (10**order[1])*np.ceil(emax/(10**order[1]))

                label = r'Energy density deviation $\hbar \omega \delta n$ [eV/angstrom$^3$]'
                format = '{:.2e}'
            elif property_plot[i] in ['sv', 'subvolumes', 'subvolume', 'subvol', 'subvols']:
                figname = 'subvolumes'
                colors = self.subvol_id
                vmin = 0
                vmax = self.n_of_subvols
                label = 'Subvolume index [-]'
                format = '{:d}'

            graph.set_array(colors)
            if colors.shape[0] > 0:
                graph.autoscale()

            if i == 0:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap = self.colormap),
                                ax = ax,
                                location = 'bottom',
                                orientation = 'horizontal',
                                fraction = 0.1,
                                aspect = 30,
                                shrink = 0.8,
                                pad = 0.1,
                                format = format)
                cax = cb.ax
            else:
                cax.clear()

                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

                cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap = self.colormap),
                                cax = cax,
                                orientation = 'horizontal',
                                format = format)

            figcolor = self.view.ax_style['figcolor']
            facecolor = self.view.ax_style['facecolor']
            linecolor = self.view.ax_style['axiscolor']

            cb.set_label(label = label, size = 'small', color = linecolor)
            ticks = np.array(cb.get_ticks())
            if property_plot[i] in ['sv', 'subvolumes', 'subvolume', 'subvol', 'subvols']:
                ticks = ticks.astype(int)
            cb.set_ticks(ticks)
            cb.set_ticklabels([format.format(i) for i in ticks], size = 'x-small', color = linecolor)

            ax.set_facecolor(figcolor)
            fig.patch.set_facecolor(figcolor)

            ax.xaxis.label.set_color(linecolor)
            ax.yaxis.label.set_color(linecolor)
            ax.zaxis.label.set_color(linecolor)
            cb.outline.set_edgecolor(linecolor)

            ax.set_xticks(ax.get_xticks())
            ax.set_yticks(ax.get_yticks())
            ax.set_zticks(ax.get_zticks())

            ax.set_xticklabels(ax.get_xticklabels(), fontdict = {'color':linecolor})
            ax.set_yticklabels(ax.get_yticklabels(), fontdict = {'color':linecolor})
            ax.set_zticklabels(ax.get_zticklabels(), fontdict = {'color':linecolor})

            plt.savefig(os.path.join(self.results_folder_name, f'{figname}.png'))

        plt.close(fig)

    def open_convergence(self, geometry):

        n_dt_to_conv = np.floor( np.log10( self.args.iterations[0] ) ) - 2    # number of timesteps to save convergence data
        n_dt_to_conv = int(10**n_dt_to_conv)
        n_dt_to_conv = max([10, n_dt_to_conv])

        filename = os.path.join(self.results_folder_name,'convergence.txt')

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
            line += ' Np Sv {:>3d} '.format(i)
        if geometry.subvol_type == 'slice':
            for i in range(self.n_of_subvols):
                line += ' Kappa Sv {:>2d} '.format(i)
            line += ' Kappa total  '
        else:
            for _, svc in enumerate(geometry.subvol_connections):
                line += ' K Con {:>3d}-{:>3d} '.format(svc[0], svc[1])

        line += '\n'

        self.f.write(line)
        self.f.close()

    def write_convergence(self, geometry):
        
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

        line += '{:>10d} '.format( self.N_p )  # number of particles

        line += np.array2string(self.subvol_temperature, formatter = {'float_kind':'{:>9.3f}'.format} ).strip('[]') + ' '
        line += np.array2string(self.subvol_energy     , formatter = {'float_kind':'{:>12.5e}'.format}).strip('[]') + ' '
        
        for i in range(self.n_of_subvols):
            line += np.array2string(self.subvol_heat_flux[i, :], formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '

        line += np.array2string(self.subvol_N_p.astype(int), formatter = {'int':'{:>10d}'.format}  ).strip('[]') + ' '

        if geometry.subvol_type == 'slice':

            line += np.array2string(self.subvol_kappa, formatter = {'float_kind':'{:>12.5e}'.format}  ).strip('[]') + ' '

            line += '{:>13.6e} '.format(self.kappa)
        else:
            line += np.array2string(self.svcon_kappa, formatter = {'float_kind':'{:>14.7e}'.format}  ).strip('[]') + ' '

        line += '\n'

        filename = os.path.join(self.results_folder_name,'convergence.txt')

        self.f = open(filename, 'a+')

        self.f.writelines(line)

        self.f.close()

    def write_final_state(self, geometry):
        
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # saving final particle states: modes, positions and occupation number.
        # Obs.: phonon properties of particles can be retrieved by mode information and subvol temperature calculated at initialisation
        
        #### PARTICLE DATA ####
        filename = os.path.join(self.results_folder_name, 'particle_data.txt')

        header ='Particles final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'q-point, branch, pos x [angs], pos y [angs], pos z [angs], occupation'

        data = np.hstack( (self.modes,
                           self.positions,
                           self.occupation.reshape(-1, 1)) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %d, %.3f, %.3f, %.3f, %.6e', delimiter = ',', header = header)
        
        #### MEAN AND STDEV QUANTITIES ####
        if self.current_timestep > 0:
            filename = os.path.join(self.results_folder_name,'subvolumes.txt')

            if geometry.subvol_type == 'slice':

                header ='subvols final state data \n' + \
                        'Date and time: {}\n'.format(time) + \
                        'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                        'subvol id, subvol x, subvol y, subvol z, subvol volume, T [K], sigma T [K], HF x [W/m^2], HF y [W/m^2], HF z [W/m^2], sigma HF x [W/m^2], sigma HF y [W/m^2], sigma HF z [W/m^2], kappa [W/m K], sigma kappa [W/m K]'

                data = np.hstack((np.arange(self.n_of_subvols).reshape(-1, 1),
                                  geometry.subvol_center                     ,
                                  self.subvol_volume.reshape(-1, 1)          ,
                                  self.view.mean_T.reshape(-1, 1)            ,
                                  self.view.std_T.reshape(-1, 1)             ,
                                  self.view.mean_sv_phi.reshape(-1, 3)       ,
                                  self.view.std_sv_phi.reshape(-1, 3)        ,
                                  self.view.mean_sv_k.reshape(-1, 1)         ,
                                  self.view.std_sv_k.reshape(-1, 1)          ))
                
                # comma separated
                np.savetxt(filename, data, '%d, %.3e, %.3e, %.3e, %.3e, %.3f, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e', delimiter = ',', header = header)
            
            else:
                header ='subvols final state data \n' + \
                        'Date and time: {}\n'.format(time) + \
                        'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                        'subvol id, subvol position, subvol volume, T [K], sigma T [K], HF x [W/m^2], HF y [W/m^2], HF z [W/m^2], sigma HF x [W/m^2], sigma HF y [W/m^2], sigma HF z [W/m^2]'

                data = np.hstack((np.arange(self.n_of_subvols).reshape(-1, 1),
                                geometry.subvol_center,
                                self.subvol_volume.reshape(-1, 1),
                                self.view.mean_T.reshape(-1, 1),
                                self.view.std_T.reshape(-1, 1),
                                self.view.mean_sv_phi.reshape(-1, 3) ,
                                self.view.std_sv_phi.reshape(-1, 3)))
                
                # comma separated
                np.savetxt(filename, data, '%d, %.3e, %.3e, %.3e, %.3e, %.3f, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e', delimiter = ',', header = header)

                # connections data
                filename = os.path.join(self.results_folder_name, 'subvol_connections.txt')

                header ='connections final state data \n' + \
                        'Date and time: {}\n'.format(time) + \
                        'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                        'connection id, sv 1, sv 2, con dx, con dy, con dz, dT [K], sigma dT [K], HF [W/m^2], sigma HF [W/m^2], kappa [W/m K], sigma kappa [W/m K]'

                data = np.hstack((np.arange(geometry.n_of_subvol_con).reshape(-1, 1),
                                  geometry.subvol_connections,
                                  geometry.subvol_con_vectors,
                                  self.view.mean_con_dT.reshape(-1, 1),
                                  self.view.std_con_dT.reshape(-1, 1),
                                  self.view.mean_con_phi.reshape(-1, 1),
                                  self.view.std_con_phi.reshape(-1, 1),
                                  self.view.mean_con_k.reshape(-1, 1),
                                  self.view.std_con_k.reshape(-1, 1)))
                np.savetxt(filename, data, '%d, %d, %d, %.3e, %.3e, %.3e, %.3f, %.3e, %.3e, %.3e, %.3e, %.3e', delimiter = ',', header = header)

                