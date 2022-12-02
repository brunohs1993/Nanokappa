# calculations
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime

# plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from scipy.interpolate.ndgriddata import NearestNDInterpolator

# geometry
import trimesh as tm
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector # FASTER
except:
    from trimesh.ray.ray_triangle import RayMeshIntersector # SLOWER
from trimesh.triangles import points_to_barycentric


# other
import sys
import os
import copy
from itertools import repeat
import time
import gc

from classes.Constants     import Constants
from classes.Visualisation import Visualisation

np.set_printoptions(precision=6, threshold=sys.maxsize, linewidth=np.nan)

matplotlib.use('Qt5Agg')

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
        self.results_folder_name = self.args.results_folder
        
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
        self.res_gen   = self.args.reservoir_gen[0]
        
        self.rough_facets        = geometry.rough_facets        # which facets have roughness as BC
        self.rough_facets_values = geometry.rough_facets_values # their values of roughness and correlation length
        
        print('Calculating diffuse scattering probabilities...')
        self.calculate_fbz_specularity(geometry, phonon)
        self.find_specular_correspondences(geometry, phonon)
        self.diffuse_scat_probability(geometry, phonon)
        
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

        self.conv_crit = float(self.args.conv_crit[0])
        self.conv_count_min = int(self.args.conv_crit[1])
        self.initialise_residue(geometry)
        
        self.plot_figures(geometry, phonon, property_plot = self.args.fig_plot, colormap = self.args.colormap[0])

        print('Creating convergence file...')
        self.open_convergence(geometry)
        self.write_convergence(geometry)

        self.view = Visualisation(self.args, geometry, phonon) # initialising visualisation class
        self.view.preprocess()

        if len(self.rt_plot) > 0:
            print('Starting real-time plot...')
            self.rt_graph, self.rt_fig = self.init_plot_real_time(geometry, phonon)
        
        print('Initialisation done!')

        # self.in_bins  = np.zeros(90)
        # self.out_bins = np.zeros(90)
        # self.var_bins = np.zeros(180)
        # self.spec_count = 0
        # self.spec_total = 0
        
    def initialise_modes(self, phonon):
        '''Generate first modes.'''

        print('Assigning modes...')

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
        enter_prob = group_vel_parallel*self.dt/self.bound_thickness.reshape(-1, 1, 1)   # shape = (R, Q, J)
        enter_prob = np.where(enter_prob < 0, 0, enter_prob)

        return enter_prob

    def generate_positions(self, number_of_particles, mesh, key):
        '''Initialise positions of a given number of particles'''
        
        if key == 'random':
            positions = tm.sample.volume_mesh(mesh, number_of_particles)
            while positions.shape[0] == 0:
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
        
        if sys.stdout.isatty():
            # setup toolbar
            nchar = int(np.floor(np.log10(self.N_p)))
            bw = os.get_terminal_size(sys.stdout.fileno()).columns - (2*nchar + 24)
            message = "{0:{3}d} of {1:{3}d} particles - {2:s}   0%".format(0, self.N_p, " "*bw, nchar)
            sys.stdout.write(message)
            sys.stdout.flush()

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

                sv_id = geometry.subvol_classifier.predict(geometry.scale_positions(x_new))

                for i in range(self.n_of_subvols):
                    if i not in self.empty_subvols:
                        
                        N = n[i] - counter[i]
                        
                        ind = np.nonzero(sv_id[:, i])[0]
                        ind = ind[:N]

                        x[i] = np.vstack((x[i], x_new[ind, :]))

                        counter[i] = x[i].shape[0]
                
                if sys.stdout.isatty():
                    time.sleep(0.01) # don't know why, but it was in the code. Need to check. Doesn't make much of a difference anyway.
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

        # occupation considering reference
        self.occupation        = phonon.calculate_occupation(self.temperatures, self.omega, reference = True)
        self.energies          = self.hbar*self.omega*self.occupation
        self.momentum          = self.hbar*self.wavevectors*self.occupation.reshape(-1, 1)
        
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
        self.calculate_momentum(geometry, phonon)

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
        self.N_leaving  = np.sum(self.enter_prob, axis = (1, 2)).round().astype(int)

        self.res_energy_balance   = np.zeros(self.n_of_reservoirs)
        self.res_heat_flux        = np.zeros((self.n_of_reservoirs, 3))
        self.res_momentum_balance = np.zeros((self.n_of_reservoirs, 3))

    def fill_reservoirs(self, geometry, phonon, n_leaving = None):

        if self.res_gen == 'fixed_rate':
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
                    mesh   = geometry.res_meshes[i]                           # select boundary
                    
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
                    new_positions, _ = tm.sample.sample_surface(mesh, n)

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
                    mesh   = geometry.res_meshes[i]                # select boundary
                    
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
                    new_positions, _ = tm.sample.sample_surface(mesh, n)

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

        print('Assigning properties...')

        omega      = phonon.omega[ modes[:,0], modes[:,1] ]        # THz * rad
        group_vel  = phonon.group_vel[ modes[:,0], modes[:,1], : ] # THz * angstrom
        wavevector = phonon.wavevectors[modes[:, 0], :]

        return omega, group_vel, wavevector

    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Constant at T_cold unless specified otherwise.'''

        print('Assigning temperatures...')

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
            elif key == 'hot':
                temperatures        = np.ones(number_of_particles)*bound_T.max()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.max()
            elif key == 'cold':
                temperatures        = np.ones(number_of_particles)*bound_T.min()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.min()
            elif key == 'mean':
                temperatures        = np.ones(number_of_particles)*bound_T.mean()
                subvol_temperatures = np.ones(  self.n_of_subvols)*bound_T.mean()
        
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

        scaled_positions = geometry.scale_positions(positions) # scale positions between 0 and 1

        subvol_id = geometry.subvol_classifier.predict(scaled_positions) # get subvol_id from the model

        if get_np:
            self.subvol_N_p = subvol_id.sum(axis = 0).astype(int)

            self.N_p = self.subvol_N_p.sum()

        subvol_id = np.argmax(subvol_id, axis = 1).astype(int)

        return subvol_id

    def refresh_temperatures(self, geometry, phonon):
        '''Refresh energies and temperatures while enforcing boundary conditions as given by geometry.'''
        self.energies = self.occupation*self.omega*self.hbar    # eV
        
        self.subvol_id = self.get_subvol_id(self.positions, geometry)

        self.calculate_energy(geometry, phonon)

        self.subvol_temperature = phonon.temperature_function(self.subvol_energy)

        self.temperatures = self.subvol_temperature[self.subvol_id]

    def calculate_energy(self, geometry, phonon):

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

    def calculate_momentum(self, geometry, phonon):
        
        momentum = np.zeros((self.n_of_subvols, 3))
        for i in range(self.n_of_subvols):
            ind = np.nonzero(self.subvol_id == i)[0] # 1d subvol id
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

        index_facets = geometry.faces_to_facets(all_faces[~nan_particles])

        all_facets                 = np.ones( positions.shape[0] )*np.nan  # all particles that do not collide do not have corresponding facets
        all_facets[~nan_particles] = index_facets

        if np.any(nan_particles):
            (all_boundary_pos[nan_particles, :],
             _,
             all_facets[nan_particles]         ) = geometry.find_boundary_naive(positions[nan_particles, :] ,
                                                                             velocities[nan_particles, :])
        
        return all_boundary_pos, all_facets

    def timesteps_to_boundary(self, positions, velocities, geometry, indexes_del_extra = None):
        '''Calculate how many timesteps to boundary scattering.'''

        if positions.shape[0] == 0:
            ts_to_boundary      = np.zeros(0)
            index_facets        = np.zeros(0)
            collision_pos        = np.zeros((0, 3))
            
            return ts_to_boundary, index_facets, collision_pos
        # mesh_pos, mesh_facets = self.find_boundary(positions, velocities, geometry) # find collision in true boundary
        mesh_pos, _, mesh_facets = geometry.find_boundary_naive(positions, velocities) # find collision in true boundary
        
        if indexes_del_extra is not None :
            # If there are any numerical errors with collision calculations, delete particles to avoid further problems
            if np.any(np.isnan(mesh_facets)):
                nan_particles = np.isnan(mesh_facets)  # bool
                indexes_del_extra[nan_particles] = True
            
            collision_pos = np.ones(positions.shape   )*np.nan
            index_facets  = np.ones(positions.shape[0])*np.nan

            collision_pos[~indexes_del_extra, :], index_facets[~indexes_del_extra] = self.apply_offset(mesh_pos[~indexes_del_extra, :], velocities[~indexes_del_extra, :], mesh_facets[~indexes_del_extra], geometry)
        else:
            collision_pos, index_facets = self.apply_offset(mesh_pos, velocities, mesh_facets, geometry)

        # calculate distances for given particles
        boundary_dist = np.linalg.norm(collision_pos - positions, axis = 1 ) # distance travelled up to collision

        ts_to_boundary = boundary_dist/( np.linalg.norm(velocities, axis = 1) * self.dt )  # such that particle hits the boundary when n_timesteps == 0 (crosses boundary)

        if indexes_del_extra is None:
            return ts_to_boundary, index_facets, collision_pos
        else:
            return ts_to_boundary, index_facets, collision_pos, indexes_del_extra

    def apply_offset(self, collision_pos, velocities, facets, geo):

        normals = -geo.mesh.facets_normal[facets.astype(int), :] # get the normals of the collision facets
        
        collision_pos += velocities*self.offset/np.sum(velocities*normals, axis = 1, keepdims = True) # ||d|| = ||v|| h / v . n

        _, close_distance, facets = geo.find_boundary_naive(collision_pos)

        too_close = np.absolute(self.offset - close_distance) > 1e-8 # check if there is any too close

        while np.any(too_close):
            extra_offset = np.where(too_close, self.offset - close_distance, 0).reshape(-1, 1) # calculate extra displacement

            collision_pos += extra_offset*(velocities/np.sum(velocities*-geo.mesh.facets_normal[facets, :], axis = 1, keepdims = True)) # adjust position

            _, close_distance, facets = geo.find_boundary_naive(collision_pos)

            too_close = np.absolute(self.offset - close_distance) > 1e-8 # check again if there is any too close
            
        return collision_pos, facets

    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes.
           
           Arguments:
              indexes: (bool or int) indexes of the particles to be deleted.'''

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
        self.subvol_id           = np.delete(self.subvol_id     , indexes, axis = 0)
    
    def calculate_fbz_specularity(self, geometry, phonon):

        n = -geometry.mesh.facets_normal[self.rough_facets, :] # normals (F, 3)
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

    def diffuse_scat_probability(self, geometry, phonon):

        if self.rough_facets.shape[0] == 0:
            print("No rough facets to calculate.")
        else:
            n = -geometry.mesh.facets_normal[self.rough_facets, :] # (F_u, 3)

            # sign of reflected waves
            v_all = phonon.group_vel # (Q, J, 3)

            v_dot_n = np.expand_dims(v_all, 2)*n        # (Q, J, F_u, 3)
            v_dot_n = v_dot_n.sum(axis = -1)            # (Q, J, F_u)
            v_dot_n = np.transpose(v_dot_n, (2, 0, 1))  # (F_u, Q, J)

            self.creation_roulette = np.zeros((self.rough_facets.shape[0], phonon.number_of_qpoints*phonon.number_of_branches))

            C_total = np.where(v_dot_n > 0, v_dot_n, 0)  # total rate of creation    (F_u, Q, J)
            # D_total = np.where(v_dot_n < 0, v_dot_n, 0)  # total rate of destruction (F_u, Q, J)

            self.creation_rate = C_total*(1-self.specularity)#/C_total.sum(axis = (1, 2), keepdims = True)

            self.creation_rate = np.where(np.isnan(self.creation_rate), 0, self.creation_rate)

            # self.destruction_rate = D_total#/D_total.sum(axis = (1, 2), keepdims = True)
            for i_f, _ in enumerate(self.rough_facets):
                self.creation_roulette[i_f, :] = np.cumsum(self.creation_rate[i_f, :, :])/np.cumsum(self.creation_rate[i_f, :, :]).max()

    def select_reflected_modes(self, in_modes, col_fac, col_pos, n_in, omega_in, geometry, phonon):
        
        col_fac = col_fac.astype(int)
        i_rough = np.array([np.nonzero(self.rough_facets == f)[0][0] for f in col_fac])
        true_spec = self.true_specular[i_rough, in_modes[:, 0], in_modes[:, 1]]
        p = self.specularity[i_rough, in_modes[:, 0], in_modes[:, 1]]

        n_p = in_modes.shape[0]
        r = np.random.rand(n_p)

        indexes_spec = np.logical_and(true_spec, r < p)
        indexes_diff = ~indexes_spec

        out_modes = np.zeros(in_modes.shape, dtype = int)
        n_out     = copy.copy(n_in)
        omega_out = copy.copy(omega_in)

        if np.any(indexes_spec):
            a = np.hstack((np.expand_dims(col_fac[indexes_spec], 1), in_modes[indexes_spec, :]))
            out_modes[indexes_spec, :] = self.specular_function(a)

        if np.any(indexes_diff):
            out_modes[indexes_diff, :] = self.pick_diffuse_modes(col_fac[indexes_diff], phonon)
            
            sv_diff = self.get_subvol_id(col_pos[indexes_diff, :], geometry, get_np = False)
            T_diff = self.subvol_temperature[sv_diff]
            omega_out[indexes_diff] = phonon.omega[out_modes[indexes_diff, 0], out_modes[indexes_diff, 1]]
            n_out[indexes_diff] = phonon.calculate_occupation(T_diff, omega_out[indexes_diff], reference = True)
        
        ##############
        # self.spec_count += indexes_spec.sum()
        # self.spec_total += indexes_spec.shape[0]
        # print('{:.3e}, {:.3e}, {:.3f}'.format(self.spec_count, self.spec_total, self.spec_count/self.spec_total))

        # normals = -geometry.mesh.facets_normal[col_fac, :]
        # v_in    = phonon.group_vel[ in_modes[:, 0],  in_modes[:, 1], :]
        # v_out   = phonon.group_vel[out_modes[:, 0], out_modes[:, 1], :]

        # in_angle  = np.arccos(np.sum(-normals*v_in, axis = 1)/np.linalg.norm(v_in, axis = 1))*180/np.pi
        # out_angle = np.arccos(np.sum(normals*v_out, axis = 1)/np.linalg.norm(v_out, axis = 1))*180/np.pi
        # var_angle = np.arccos(np.sum(-v_in*v_out, axis = 1)/(np.linalg.norm(v_in, axis = 1)*np.linalg.norm(v_out, axis = 1)))*180/np.pi

        # self.in_bins  += np.histogram( in_angle,  90, range = (0,  90))[0]
        # self.out_bins += np.histogram(out_angle,  90, range = (0,  90))[0]
        # self.var_bins += np.histogram(var_angle, 180, range = (0, 180))[0]

        # fig, ax = plt.subplots(nrows = 1, ncols = 3, dpi = 200, figsize = (20, 5))

        # ax[0].bar(np.arange(90), self.in_bins/self.in_bins.sum())
        # ax[1].bar(np.arange(90), self.out_bins/self.out_bins.sum())
        # ax[2].bar(np.arange(180), self.var_bins/self.var_bins.sum())

        # ax[0].set_title('In')
        # ax[1].set_title('Out')
        # ax[2].set_title('Var')

        # plt.suptitle('overall specularity: {:.3f}'.format(self.spec_count/self.spec_total))

        # plt.tight_layout()
        # plt.savefig(self.results_folder_name + 'specular_hist.png')
        # plt.close(fig)

        # saving scat data        
        # with open(self.results_folder_name+'scattering_diff.txt', 'a')as f: 
        #     for i in range(new_modes.shape[0]):
        #         if indexes_diff[i]:
        #             f.write('{:d}, {:d}, {:d}, {:d}, {:.1f}, {:.1f}, {:.1f} \n'.format(incident_modes[i, 0], incident_modes[i, 1],
        #                                                                     new_modes[i, 0]     , new_modes[i, 1]     ,
        #                                                                     n_g[i, 0], n_g[i, 1], n_g[i, 2]))

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

    def find_specular_correspondences(self, geo, phonon):
        facets = self.rough_facets
        n      = -np.round(geo.mesh.facets_normal[facets, :], decimals = 10)

        k = phonon.wavevectors
        v = phonon.group_vel
        tol = np.absolute(1/(2*phonon.data_mesh))

        # array of INCOMING modes that CAN be specularly reflected to other mode - Initially all false
        true_spec = np.zeros((facets.shape[0], phonon.number_of_qpoints, phonon.number_of_branches), dtype = bool)

        correspondent_modes = np.zeros((0, 5), dtype = int)

        _, inv_omega = np.unique(phonon.omega, return_inverse=True) # unique omega and inverse array
        inv_omega    = inv_omega.reshape(phonon.omega.shape)        # reshaping to (Q, J,)
        
        for i_f, f in enumerate(facets):
            v_dot_n = np.sum(v*n[i_f, :], axis = 2) 
            s_in  = v_dot_n < 0                      # modes coming to the facet
            s_out = v_dot_n > 0                      # available modes going out of the facet

            active_k = np.any(s_in, axis = 1) # bool, (Q,) - wavevectors that can arrive to the facet

            k_try   = k[active_k, :] - 2*n[i_f, :]*np.sum(k[active_k, :]*n[i_f, :], axis = 1, keepdims = True) # reflect them specularly
            _, disp = phonon.find_min_k(k_try, return_disp = True)                                             # check if thay stay in the FBZ

            active_k[active_k] = np.all(disp == 0, axis = 1) # normal processes can be specular

            # recalculating reflections of the ones that remained active (redundant but safe)
            k_try = k[active_k, :] - 2*n[i_f, :]*np.sum(k[active_k, :]*n[i_f, :], axis = 1, keepdims = True) # (Qa, 3)

            near_k_func = NearestNDInterpolator(k, np.arange(phonon.number_of_qpoints)) # nearest interpolator function on K space

            # checking k availability
            q_near = near_k_func(k_try)             # nearest qpoint to k_try
            k_near = phonon.wavevectors[q_near, :]  # get the nearest k vector in relation to k_try
            k_dist = np.absolute(k_try - k_near)    # calculate the distance between the two in each dimension
            
            # it should be a wavevector with at least one valid velocity and within grid tolerance
            in_tol = np.logical_and(np.any(s_out[q_near, :], axis = 1), np.all(k_dist < tol, axis = 1))
            
            active_k[active_k] = in_tol # update the active to only those within tolerance

            # recalculating reflections of the ones that remained active (redundant but safe)
            k_try = k[active_k, :] - 2*n[i_f, :]*np.sum(k[active_k, :]*n[i_f, :], axis = 1, keepdims = True) # (Qa, 3)
            
            out_q = near_k_func(k_try)                            # out qpoints (Qa,)
            in_q  = np.arange(phonon.number_of_qpoints)[active_k] # in  qpoints (Qa,)

            v_try = v[ in_q, :, :] - 2*n[i_f, :]*np.sum(v[in_q, :, :]*n[i_f, :], axis = 2, keepdims = True) # (Qa, J, 3)
            v_try = np.expand_dims(v_try, axis = 0)                                                         # (1, Qa, J, 3)
            v_out = np.transpose(np.expand_dims(v[out_q, :, :], axis = 0), axes = (2, 1, 0, 3))             # (J, Qa, 1, 3)
            
            # distance between in and out velocities
            v_dist = np.sum((v_try - v_out)**2, axis = -1)**0.5 # (J, Qa, J)
            same_v = v_dist == 0                                # (J, Qa, J)

            # valid reflections only with the same reflected velocity
            active_k[active_k] = np.any(same_v, axis = (0,2))

            same_v = same_v[:, np.any(same_v, axis = (0, 2)), :]
            
            # recalculating reflections of the ones that remained active (redundant but safe)
            k_try = k[active_k, :] - 2*n[i_f, :]*np.sum(k[active_k, :]*n[i_f, :], axis = 1, keepdims = True) # (Qa, 3)
            out_q = near_k_func(k_try)                            # out qpoints (Qa,)
            in_q  = np.arange(phonon.number_of_qpoints)[active_k] # in  qpoints (Qa,)

            # getting which in and out modes have the same omega
            in_omega_i  = inv_omega[in_q, :]                                                # (Qa, J)
            out_omega_i = np.transpose(np.expand_dims(inv_omega[out_q, :], 0), (2, 1, 0))   # (J, Qa, 1)

            same_omega = in_omega_i == out_omega_i # if any branch of out_k has a frequency equal to each branch of in_k (J_out, Qa, J_in)
            same_omega = np.logical_and(same_omega, same_v)

            # same_omega = np.logical_and(same_omega, s_in[in_q, :])                                               # incoming modes should be valid (this is probably redundant)
            # same_omega = np.logical_and(same_omega, np.transpose(np.expand_dims(s_out[out_q, :], 0), (2, 1, 0))) # outgoing modes should be valid (this is probably not)

            same_omega_indexes = np.nonzero(np.any(same_omega, axis = 0)) # indexes where any same_omega == True for the incoming modes
            same_omega_q_in = in_q[same_omega_indexes[0]] # get incoming qpoint with available same omega
            same_omega_j_in = same_omega_indexes[1]       # get incoming branch with available same omega

            same_omega_q_out = out_q[same_omega_indexes[0]] # correspondent ougoing out

            # valid_v = np.where(same_omega, v_dist, np.inf)
            # near_v  = valid_v == 0 # valid_v.min(axis = 0)

            # same_omega_j_out = np.argmax(near_v, axis = 0)[same_omega_indexes[0], same_omega_j_in] # get qpoint with same omega
            same_omega_j_out = np.argmax(same_omega, axis = 0)[same_omega_indexes[0], same_omega_j_in] # get qpoint with same omega

            for i_m, m in enumerate(np.vstack((same_omega_q_in, same_omega_j_in)).T):
                true_spec[i_f, m[0], m[1]] = True # the only true specular modes

            n_ts = same_omega_q_in.shape[0]

            correspondent_modes = np.vstack((correspondent_modes, np.vstack((np.ones(n_ts)*f, same_omega_q_in, same_omega_j_in, same_omega_q_out, same_omega_j_out)).T))

        self.correspondent_modes = correspondent_modes.astype(int)
        self.specular_function   = NearestNDInterpolator(self.correspondent_modes[:, :3], self.correspondent_modes[:, 3:])
        self.true_specular       = copy.copy(true_spec)
        # self.specularity         = np.where(self.true_specular, self.specularity, 0)
        self.specularity         = self.true_specular.astype(int)*self.specularity

        np.savetxt(self.results_folder_name + 'specular_correspondences.txt', self.correspondent_modes, fmt = '%d')

    def get_specular_correspondece(self, a):
        '''a = array containing col_fac, in_q, in_j, in this order'''
        i = np.nonzero(np.all(self.correspondent_modes[:, :3] == a, axis = 1))[0][0]
        return self.correspondent_modes[i, 3:]

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
        normals = -geometry.mesh.facets_normal[connected_facets, :]                                       # get the normals of the new facets
        alpha = 1+2*self.offset/np.sum(L*normals, axis = 1, keepdims = True)                              # correct translation considering offsets
        new_positions = collision_positions + L*alpha                                                     # translate particles

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
        
        ############## SAVING SCATTERING DATA ##################
        # with open(self.results_folder_name+'scattering_spec.txt', 'a') as f:
        #     for i in range(new_modes.shape[0]):
        #         if indexes_spec[i]:
        #             n_g = -geometry.mesh.facets_normal[int(collision_facets[i]), :]
        #             f.write('{:d}, {:d}, {:d}, {:d}, {:.1f}, {:.1f}, {:.1f} \n'.format(incident_modes[i, 0], incident_modes[i, 1],
        #                                                                             new_modes[i, 0]     , new_modes[i, 1]     ,
        #                                                                             n_g[0], n_g[1], n_g[2]))
        ###############################
        
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

        # col_track_n = np.zeros(0)
        # col_track_fac = np.zeros(0)
        # col_track_pos = np.zeros((0, 3))
        # col_track_mode = np.zeros((0, 2))

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
                        energies = self.energies[indexes_del][indexes_res]
                        self.res_energy_balance[i] -= energies.sum()

                        # adding heat flux
                        hflux = energies.reshape(-1, 1)*self.group_vel[indexes_del, :][indexes_res, :]
                        self.res_heat_flux[i, :] += hflux.sum(axis = 0)

                        # adding momentum
                        momentum = self.momentum[indexes_del, :][indexes_res, :]
                        self.res_momentum_balance[i, :] += momentum.sum(axis = 0)

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
            indexes_ref = np.logical_and(indexes_ref               ,
                                         (1-calculated_ts) > new_n_timesteps)
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
        
        if self.n_of_reservoirs > 0:
            
            self.res_energy_balance   = phonon.normalise_to_density(self.res_energy_balance)
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
        
        if self.current_timestep == 0:
            print('Simulating...')

        if (self.current_timestep % 100) == 0:
            self.write_final_state(geometry)
            self.view.postprocess(verbose = False)
            self.update_residue(geometry)

            info ='Timestep {:>5d} - max residue: {:>9.3e} ({:<9s}) ['.format(int(self.current_timestep), self.max_residue, self.max_residue_qt)
            for sv in range(self.n_of_subvols):
                info += ' {:>7.3f}'.format(self.subvol_temperature[sv])
            info += ' ]'
            print(info)

        self.drift()                                    # drift particles

        if self.n_of_reservoirs > 0:
            if self.res_gen == 'fixed_rate':
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
            self.calculate_momentum(geometry, phonon)
            self.write_convergence(geometry)      # write data on file

        if (len(self.rt_plot) > 0) and (self.current_timestep % self.n_dt_to_plot == 0):
            self.plot_real_time()
        
        gc.collect() # garbage collector

    def initialise_residue(self, geo):
        if geo.subvol_type == 'slice':
            self.old_mean_large = np.ones(5*self.n_of_subvols+self.n_of_reservoirs)
            self.old_std_large  = np.ones(5*self.n_of_subvols+self.n_of_reservoirs)
        else:
            self.old_mean_large = np.ones(4*self.n_of_subvols+self.n_of_reservoirs+geo.n_of_subvol_con)
            self.old_std_large  = np.ones(4*self.n_of_subvols+self.n_of_reservoirs+geo.n_of_subvol_con)

        self.residue_all = np.ones(8)
        self.conv_count = 0
        self.finish_sim = False
        self.max_residue    = 1
        self.max_residue_qt = 'none'

        self.residue_qts   = ['T_{:d}'.format(i) for i in range(self.n_of_subvols)] + \
                ['phi_{:s}_{:d}'.format(i, j) for j in range(self.n_of_subvols) for i in ['x', 'y', 'z']] + \
                ['en_res_{:d}'.format(i) for i in range(self.n_of_reservoirs)]
        if geo.subvol_type == 'slice':
            self.residue_qts += ['k_{:d}'.format(i) for i in range(self.n_of_subvols)]
        else:
            self.residue_qts += ['k_{:d}'.format(i) for i in range(geo.n_of_subvol_con)]

    def update_residue(self, geo):
        
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

            if geo.subvol_type == 'slice':
                new_mean_large = np.concatenate((self.view.mean_T, self.view.mean_sv_phi, self.view.mean_en_res, self.view.mean_sv_k))
                new_std_large  = np.concatenate((self.view.std_T , self.view.std_sv_phi , self.view.std_en_res , self.view.std_sv_k ))
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

        with open(self.results_folder_name + 'residue.txt', 'a+') as f:
            f.writelines(s)

    def init_plot_real_time(self, geometry, phonon):
        '''Initialises the real time plot to be updated for each timestep.'''

        n_dt_to_plot = np.floor( np.log10( self.args.iterations[0] ) ) - 1    # number of timesteps to save new animation frame
        n_dt_to_plot = int(10**n_dt_to_plot)
        n_dt_to_plot = max([1, n_dt_to_plot])

        self.n_dt_to_plot = 100 # n_dt_to_plot

        if self.rt_plot[0] in ['T', 'temperature']:
            colors = self.temperatures
            if self.n_of_reservoirs > 0:
                T = np.concatenate((self.res_bound_values[self.res_bound_cond == 'T'], self.subvol_temperature))
            else:
                T = self.subvol_temperature
            vmin = np.floor(T.min()-1)
            vmax = np.ceil(T.max()+1)
            label = 'Temperature [K]'
            format = '{:.1f}'
        elif self.rt_plot[0] in ['e', 'energy']:
            colors = self.energies
            if self.n_of_reservoirs > 0:
                T = np.concatenate((self.res_bound_values[self.res_bound_cond == 'T'], self.subvol_temperature))
            else:
                T = self.subvol_temperature
            vmin = phonon.calculate_energy(T.min()-1, phonon.omega, reference = True).min()
            vmax = phonon.calculate_energy(T.max()+1, phonon.omega, reference = True).max()
            label = r'Energy density deviation $\hbar \omega \delta n$ [eV/angstrom$^3$]'
            format = '{:.2e}'
        elif self.rt_plot[0] in ['omega', 'angular_frequency']:
            colors = self.omega
            vmin = phonon.omega[phonon.omega>0].min()
            vmax = phonon.omega.max()
            label = r'Angular frequency $\omega$ [THz$\cdot$rad]'
            format = '{:.2e}'
        elif self.rt_plot[0] in ['n', 'occupation']:
            colors = self.occupation
            order = [np.floor( np.log10( self.occupation.min()) ), np.floor( np.log10( self.occupation.max()) )]
            vmin = (10**order[0])*np.ceil(self.occupation.min()/(10**order[0]))
            vmax = (10**order[1])*np.ceil(self.occupation.max()/(10**order[1]))
            label = r'Occupation number deviation $\delta n$ [phonons/angstrom$^3$]'
            format = '{:.2e}'
        elif self.rt_plot[0] in ['qpoint']:
            colors = self.modes[:, 0]
            vmin = 0
            vmax = phonon.number_of_qpoints
            label = 'Q-point index [-]'
            format = '{:d}'
        elif self.rt_plot[0] in ['branch']:
            colors = self.modes[:, 1]
            vmin = 0
            vmax = phonon.number_of_branches
            label = 'Branch index [-]'
            format = '{:d}'
        elif self.rt_plot[0] in ['ts_to_boundary']:
            colors = self.n_timesteps
            vel  = np.sqrt( (phonon.group_vel**2).sum(axis = 2) )
            # min_vel = vel[vel>0].min()
            # max_path = np.sqrt( (geometry.bounds[1, :]**2).sum() ).min()
            vmin = 0
            vmax = 100 # max_path/(min_vel*self.dt)
            label = 'Timesteps to collision [timesteps]'
            format = '{:d}'
        elif self.rt_plot[0] in ['subvol']:
            colors = self.subvol_id
            vmin = 0
            vmax = self.n_of_subvols
            label = 'Subvolume index [-]'
            format = '{:d}'

        # plt.ion()
        
        # box_size = np.ptp(geometry.bounds, axis = 0)
        # figsize = np.array([box_size.max()/2, box_size.min()])
        # figsize = figsize*8/(box_size.max()/2)

        figsize = (15,15)

        # fig = plt.figure(figsize = figsize, dpi = 100)
        # ax = fig.add_subplot(111, projection='3d')

        fig, ax = geometry.plot_facet_boundaries(geometry.mesh, l_color = 'grey')
        
        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
        ax.set_xlim(geometry.bounds[:, 0])
        ax.set_ylim(geometry.bounds[:, 1])
        ax.set_zlim(geometry.bounds[:, 2])

        # for f in range(len(geometry.mesh.facets)):
        #     for e in geometry.mesh.facets_boundary[f]:
        #         p = geometry.mesh.vertices[e, :]
        #         # ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'gray')
        #         # testing in a geometry with straight lines, so this makes sense in this case:
        #         equal_coord = p[0, :] == p[1, :]
        #         if np.any(np.isin(p[0, equal_coord], geometry.bounds[0, :])):
        #             ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'gray')

        graph = ax.scatter(self.positions[:, 0],
                           self.positions[:, 1],
                           self.positions[:, 2],
                           s    = 1            ,
                           vmin = vmin         ,
                           vmax = vmax         ,
                           c    = colors       ,
                           cmap = self.colormap)
        
        # for f in range(len(geometry.mesh.facets)):
        #     for e in geometry.mesh.facets_boundary[f]:
        #         p = geometry.mesh.vertices[e, :]
        #         # ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'gray')
        #         # testing in a geometry with straight lines, so this makes sense in this case:
        #         equal_coord = p[0, :] == p[1, :]
        #         if np.any(np.isin(p[0, equal_coord], geometry.bounds[1, :])):
        #             ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', c = 'gray')
        
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap = self.colormap),
                     ax = ax,
                     location = 'bottom',
                     orientation = 'horizontal',
                     fraction = 0.05,
                     aspect = 30,
                     shrink = 0.8,
                     pad = -0.1,
                     format = format)

        cb.set_label(label = label, size = 'small')
        cb.set_ticklabels([format.format(i) for i in cb.get_ticks()], size = 'small')

        # graph.set_animated(True)

        plt.tight_layout()

        fig.canvas.draw()

        # plt.show(block = False)

        self.plot_images = [np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape(fig.canvas.get_width_height()[::-1]+(3,))]

        plt.figure(fig.number)
        plt.savefig(self.results_folder_name+'last_anim_frame.png')

        return graph, fig

    def plot_real_time(self, property_plot = None, colormap = 'viridis'):
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

    def plot_figures(self, geometry, phonon, property_plot=['temperature'], colormap = 'viridis'):
        
        fig, ax = geometry.plot_facet_boundaries(geometry.mesh, l_color = 'grey')
        n = len(property_plot)

        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
        
        graph = ax.scatter(self.positions[:, 0],
                           self.positions[:, 1],
                           self.positions[:, 2],
                           cmap = colormap     ,
                           c = np.random.rand(self.positions.shape[0]),
                           s = 1               )

        for i in range(n):
            if property_plot[i] in ['T', 'temperature', 'temperatures']:
                figname = 'init_temperature'
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
                figname = 'init_omega'
                colors = self.omega
                vmin = phonon.omega[phonon.omega>0].min()
                vmax = phonon.omega.max()
                label = r'Angular frequency $\omega$ [THz$\cdot$rad]'
                format = '{:.2e}'
            elif property_plot[i] in ['n', 'occupation']:
                figname = 'init_occupation'
                colors = self.occupation
                order = [np.floor( np.log10( self.occupation.min()) ), np.floor( np.log10( self.occupation.max()) )]
                vmin = (10**order[0])*np.ceil(self.occupation.min()/(10**order[0]))
                vmax = (10**order[1])*np.ceil(self.occupation.max()/(10**order[1]))
                label = r'Occupation number deviation $\delta n$ [phonons/angstrom$^3$]'
                format = '{:.2e}'
            elif property_plot[i] in ['e', 'energy', 'energies']:
                figname = 'init_energy'
                colors = self.energies
                if self.n_of_reservoirs > 0:
                    T = np.concatenate((self.res_bound_values[self.res_bound_cond == 'T'], self.subvol_temperature))
                else:
                    T = self.subvol_temperature
                vmin = phonon.calculate_energy(T.min()-1, phonon.omega, reference = True).min()
                vmax = phonon.calculate_energy(T.max()+1, phonon.omega, reference = True).max()
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
            if i > 0:
                cb.remove()

            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap = self.colormap),
                              ax = ax,
                              location = 'bottom',
                              orientation = 'horizontal',
                              fraction = 0.05,
                              aspect = 30,
                              shrink = 0.8,
                              pad = -0.1,
                              format = format)

            cb.set_label(label = label, size = 'small')
            cb.set_ticklabels([format.format(i) for i in cb.get_ticks()], size = 'small')

            plt.tight_layout()
            
            plt.savefig(self.results_folder_name+figname+'.png')

        plt.close(fig)

    def open_convergence(self, geometry):

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
            line += ' Np Sv {:>3d} '.format(i)
        if geometry.subvol_type == 'slice':
            line += ' Kappa total  '
            for i in range(self.n_of_subvols):
                line += ' Kappa Sv {:>2d} '.format(i)
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
            for i in range(self.n_of_reservoirs):
                line += np.array2string(self.res_momentum_balance[i, :], formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '

        line += '{:>10d} '.format( self.N_p )  # number of particles

        line += np.array2string(self.subvol_temperature, formatter = {'float_kind':'{:>9.3f}'.format} ).strip('[]') + ' '
        line += np.array2string(self.subvol_energy     , formatter = {'float_kind':'{:>12.5e}'.format}).strip('[]') + ' '
        
        for i in range(self.n_of_subvols):
            line += np.array2string(self.subvol_heat_flux[i, :], formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '
        for i in range(self.n_of_subvols):
            line += np.array2string(self.subvol_momentum[i, :] , formatter = {'float_kind':'{:>14.6e}'.format}).strip('[]') + ' '

        line += np.array2string(self.subvol_N_p.astype(int), formatter = {'int':'{:>10d}'.format}  ).strip('[]') + ' '

        if geometry.subvol_type == 'slice':

            line += np.array2string(self.subvol_kappa, formatter = {'float_kind':'{:>12.5e}'.format}  ).strip('[]') + ' '

            line += '{:>13.6e} '.format(self.kappa)
        else:
            line += np.array2string(self.svcon_kappa, formatter = {'float_kind':'{:>14.7e}'.format}  ).strip('[]') + ' '

        line += '\n'

        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        self.f.writelines(line)

        self.f.close()

    def write_final_state(self, geometry):
        
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # saving final particle states: modes, positions, subvol id and occupation number.
        # Obs.: phonon properties of particles can be retrieved by mode information and subvol temperature
        
        #### PARTICLE DATA ####
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
        
        #### MEAN AND STDEV QUANTITIES ####
        if self.current_timestep > 0:
            filename = self.results_folder_name + 'subvolumes.txt'

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
                filename = self.results_folder_name + 'subvol_connections.txt'

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

                