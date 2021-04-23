# calculations
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime

# plotting
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

# geometry
import trimesh as tm
from trimesh.ray.ray_pyembree import RayMeshIntersector # CHANGE LATER FROM RAY_TRIANGLE TO RAY_PÝEMBREE (FASTER)

# other
import sys
import copy
from functools import partial

from classes.Constants import Constants

np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=np.nan)

matplotlib.use('Agg')

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

#   Class with information about the particles contained in the domain.

#   TO DO
#   - Add option to apply heat flux as boundary condition
#   
#   QUESTIONS
#   
#   - Do the q-poins even matter? Because they are not used in any calculation, and they
#     can be recovered by using self.modes i.e. q-point and branch references.

class Population(Constants):
    '''Class comprising the particles to be simulated.'''

    def __init__(self, arguments, geometry, phonon):
        super(Population, self).__init__()

        self.args = arguments
        
        self.n_of_slices   = self.args.slices[0]

        self.empty_slices      = self.args.empty_slices
        self.n_of_empty_slices = len(self.empty_slices)
        
        self.particles_pmps = int(float(self.args.particles[0]))
        self.N_p   = int(self.particles_pmps*phonon.number_of_modes*(self.n_of_slices-self.n_of_empty_slices))
         
        self.dt            = float(self.args.timestep[0])   # ps
        
        self.slice_axis    = self.args.slices[1]
        self.slice_length  = geometry.slice_length  # angstrom
        self.slice_volume  = geometry.slice_volume  # angstrom³

        self.bound_cond     = self.args.bound_cond[0]
        self.T_boundary     = np.array(self.args.temperatures)
        self.T_distribution = self.args.temp_dist[0]

        self.colormap = self.args.colormap[0]
        self.fig_plot = self.args.fig_plot
        self.rt_plot  = self.args.rt_plot

        self.current_timestep = 0

        print('Initialising population...')

        self.initialise_all_particles(phonon, geometry) # initialising particles

        print('Initialising reservoirs...')
        self.initialise_reservoirs(geometry, phonon)            # initialise reservoirs

        self.n_timesteps = self.timesteps_to_boundary(self.positions, self.group_vel, geometry) # calculating timesteps to boundary

        self.results_folder_name = self.args.results_folder

        print('Creating convergence file...')
        self.open_convergence(self.results_folder_name)

        if len(self.rt_plot) > 0:
            print('Starting real-time plot...')
            self.rt_graph, self.rt_fig = self.init_plot_real_time(geometry, phonon)
        
        print('Initialisation done!')
        

    def initialise_modes(self, phonon):
        '''Generate first modes.'''

        # creating mode matrix
        self.unique_modes = np.stack(np.meshgrid( np.arange(phonon.number_of_qpoints), np.arange(phonon.number_of_branches) ), axis = -1 ).reshape(-1, 2).astype(int)

        # tiling
        modes = np.tile( self.unique_modes, (int( self.particles_pmps*(self.n_of_slices-self.n_of_empty_slices) ), 1) )

        return modes.astype(int)

    def generate_positions(self, number_of_particles, mesh, key):
        '''Initialise positions of a given number of particles'''
        
        if key == 'random':
            bounds = mesh.bounds
            bounds_range = bounds[1,:]- bounds[0,:]

            positions = np.random.rand(number_of_particles, 3)*bounds_range + bounds[0,:] # Generate random points inside domains bounding box, positions in angstrom
            positions, points_out = self.remove_out_points(positions, mesh)       # Remove points outside mesh

            while positions.shape[0] != number_of_particles:
                # Keep generating points until all of them are inside the mesh
                new_positions = np.random.rand(points_out, 3)*bounds_range + bounds[0,:]    # generate new
                new_positions, points_out = self.remove_out_points(new_positions, mesh)     # check which are inside
                positions = np.append(positions, new_positions, 0)                          # add the good ones

        elif key == 'center':
            
            center = mesh.center_mass
            positions = np.ones( (number_of_particles, 3) )*center # Generate all points at the center of the bounding box, positions in angstrom

        return positions

    def remove_out_points(self, points, mesh):
        '''Remove points outside the geometry mesh.'''
        in_points = mesh.contains(points)     # boolean array
        out_points = ~in_points
        return np.delete(points, out_points, axis=0), out_points.sum()   # new points, number of particles

    def initialise_all_particles(self, phonon, geometry):
        '''Uses Population atributes to generate all particles.'''
        
        self.modes = self.initialise_modes(phonon) # getting modes

        # initialising positions one mode at a time (slower but uses less memory)

        self.positions = np.empty( (0, 3) )
        
        number_of_particles = phonon.number_of_modes

        if self.args.part_dist[0] == 'random_domain':
            for _ in range(self.n_of_slices*self.particles_pmps):
                new_positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'random')
                self.positions = np.vstack( (self.positions, new_positions) )
        
        elif self.args.part_dist[0] == 'center_domain':
            for _ in range(self.n_of_slices*self.particles_pmps):
                new_positions = self.generate_positions(number_of_particles, geometry.mesh, key = 'center')
                self.positions = np.vstack( (self.positions, new_positions) )
        
        elif self.args.part_dist[0] == 'random_slice':
            for i in range(self.n_of_slices):
                if i not in self.empty_slices:
                    for _ in range(self.particles_pmps):
                        new_positions = self.generate_positions(number_of_particles, geometry.slice_meshes[i], key = 'random')
                        self.positions = np.vstack( (self.positions, new_positions) )
        
        elif self.args.part_dist[0] == 'center_slice':
            for i in range(self.n_of_slices):
                if i not in self.empty_slices:
                    for _ in range(self.particles_pmps):
                        new_positions = self.generate_positions(number_of_particles, geometry.slice_meshes[i], key = 'center')
                        self.positions = np.vstack( (self.positions, new_positions) )
        
        self.temperatures = self.assign_temperatures(self.positions, geometry)
        
        # assigning properties from Phonon
        self.omega, self.q_points, self.group_vel = self.assign_properties(self.modes, self.temperatures, phonon)
        
        self.occupation   = phonon.calculate_occupation(self.temperatures, self.omega)
        self.energies     = phonon.calculate_energy(self.temperatures, self.omega)

    def initialise_reservoirs(self, geometry, phonon):
        # coded at first for simple boxes at the extremities of reservoirs.

        self.res1_active_modes = self.unique_modes
        self.res2_active_modes = self.unique_modes

        reservoir_length = self.slice_length

        self.N_p_res = int(self.particles_pmps*phonon.number_of_modes) # equivalent number of particles in that distance

        self.bound_res1 = copy.deepcopy(geometry.bounds)
        self.bound_res1[:, self.slice_axis] = np.array([-reservoir_length, 0])
        
        self.bound_res2 = copy.deepcopy(geometry.bounds)
        self.bound_res2[:, self.slice_axis] = np.array([self.bound_res2[1, self.slice_axis], self.bound_res2[1, self.slice_axis]+reservoir_length])

        # Generate first particles
        self.fill_reservoirs(phonon)

    def generate_reservoir_particles(self, box, modes):

        '''Generates particles and modes for reservoirs given their bounding box and phonon properties'''

        # positions
        positions_res = np.random.rand(self.N_p_res, 3)*np.ptp(box, axis = 0)+box[0,:]

        # modes
        modes_res = np.tile( modes, (int(self.particles_pmps), 1) )

        return positions_res, modes_res
    
    def fill_reservoirs(self, phonon):

        # - positions and modes
        self.positions_res1, self.modes_res1 = self.generate_reservoir_particles(self.bound_res1, self.res1_active_modes) #phonon)
        self.positions_res2, self.modes_res2 = self.generate_reservoir_particles(self.bound_res2, self.res2_active_modes) #phonon)

        # temperatures
        self.temperatures_res1 = np.ones(self.N_p_res)*self.T_boundary[0]
        self.temperatures_res2 = np.ones(self.N_p_res)*self.T_boundary[1]

        # - assign angular frequencies, q points, group velocities and lifetimes
        self.omega_res1, self.q_points_res1, self.group_vel_res1 = self.assign_properties(self.modes_res1, self.temperatures_res1, phonon)
        self.omega_res2, self.q_points_res2, self.group_vel_res2 = self.assign_properties(self.modes_res2, self.temperatures_res2, phonon)

        self.energies_res1 = phonon.calculate_energy(self.temperatures_res1, self.omega_res1)
        self.energies_res2 = phonon.calculate_energy(self.temperatures_res2, self.omega_res2)

        self.occupation_res1 = phonon.calculate_occupation(self.temperatures_res1, self.omega_res1)
        self.occupation_res2 = phonon.calculate_occupation(self.temperatures_res2, self.omega_res2)

        
    def add_reservoir_particles(self, geometry, phonon):
        '''Add the particles that came from the reservoir into the geometry. Similar to the boundary scattering method.'''
        
        positions    = np.vstack((self.positions_res1, self.positions_res2))
        modes        = np.vstack((self.modes_res1    , self.modes_res2    ))
        group_vel    = np.vstack((self.group_vel_res1, self.group_vel_res2))
        q_points     = np.vstack((self.q_points_res1 , self.q_points_res2 ))

        temperatures       = np.concatenate((self.temperatures_res1      , self.temperatures_res2      ))
        omega              = np.concatenate((self.omega_res1             , self.omega_res2             ))
        occupation         = np.concatenate((self.occupation_res1        , self.occupation_res2        ))
        energies           = np.concatenate((self.energies_res1          , self.energies_res2          ))
        

        if (self.bound_cond == 'periodic') and (geometry.shape == 'cuboid'):    # applicable just for cuboids

            check = positions >= geometry.bounds.reshape(2, 1, 3)    # True if less or equal than each bound value

            check = check.sum(axis = 0) # this will result in 0 for numbers out of the lower limit, 1 for points inside the limits, 2 for outside de upper limit

            # deleting particles that stil are in the reservoirs
            indexes_del = (check[:, self.slice_axis] != 1)

            positions    = positions[~indexes_del, :]
            modes        = modes[~indexes_del, :]
            check        = check[~indexes_del, :]

            # applying periodicity
            lower_points = np.where(check == 0, geometry.bounds[1,:] - (np.abs(positions) % np.ptp(geometry.bounds, axis = 0)), 0)
            in_points    = np.where(check == 1, positions                                                           , 0)
            upper_points = np.where(check == 2, geometry.bounds[0,:] + (positions % np.ptp(geometry.bounds, axis = 0)), 0)

            new_positions = lower_points + in_points + upper_points

            temperatures       = temperatures[~indexes_del]
            omega              = omega[~indexes_del]
            group_vel          = group_vel[~indexes_del, :]
            occupation         = occupation[~indexes_del]
            energies           = energies[~indexes_del]
            q_points           = q_points[~indexes_del, :]

            n_timesteps = self.timesteps_to_boundary(new_positions, group_vel, geometry)  # calculates ts until boundary scattering
            
        # add new particles to the population

        self.positions          = np.vstack((self.positions  , new_positions))
        self.modes              = np.vstack((self.modes      , modes      ))
        self.q_points           = np.vstack((self.q_points   , q_points   ))
        self.group_vel          = np.vstack((self.group_vel  , group_vel  ))
        self.n_timesteps        = np.concatenate((self.n_timesteps, n_timesteps))
        self.temperatures       = np.concatenate((self.temperatures      , temperatures      ))
        self.omega              = np.concatenate((self.omega             , omega             ))
        self.energies           = np.concatenate((self.energies          , energies          ))
        self.occupation         = np.concatenate((self.occupation        , occupation        ))

    def assign_properties(self, modes, temperatures, phonon):
        '''Get properties from the indexes.'''

        omega              = phonon.omega[ modes[:,0], modes[:,1] ]        # THz * rad
        q_points           = phonon.q_points[ modes[:,0], : ]              # reduced reciprocal coordinates
        group_vel          = phonon.group_vel[ modes[:,0], modes[:,1], : ] # THz * angstrom

        return omega, q_points, group_vel

    def assign_temperatures(self, positions, geometry):
        '''Atribute initial temperatures imposing fixed temperatures on first and last slice. Randomly within delta T unless specified otherwise.'''

        number_of_particles = positions.shape[0]
        key = self.T_distribution

        temperatures = np.zeros(number_of_particles)    # initialise temperature array

        if key == 'linear':
            # calculates T for each slice
            step = (self.T_boundary[1]-self.T_boundary[0])/(self.n_of_slices-1)
            T_array = np.arange(self.T_boundary[0], self.T_boundary[1] + step, step)

            for i in range(self.n_of_slices):   # assign temperatures for particles in each slice
                indexes = (positions[:, self.slice_axis] >= i*self.slice_length) & (positions[:, self.slice_axis] < (i+1)*self.slice_length)
                temperatures[indexes] = T_array[i]
                    
        else:
            if key == 'random':
                temperatures = np.random.rand(number_of_particles)*(self.T_boundary.ptp() ) + self.T_boundary.min()
                
            elif key == 'constant_hot':
                temperatures = np.ones(number_of_particles)*self.T_boundary.max()
            elif key == 'constant_cold':
                temperatures = np.ones(number_of_particles)*self.T_boundary.min()
            elif key == 'constant_mean':
                temperatures = np.ones(number_of_particles)*self.T_boundary.mean()
            
            # imposing boundary conditions

            indexes = positions[:, self.slice_axis] < self.slice_length
            temperatures[indexes] = self.T_boundary[0]

            indexes = positions[:, self.slice_axis] >= geometry.bounds[1, self.slice_axis]-self.slice_length
            temperatures[indexes] = self.T_boundary[1]

        return temperatures
    
    def refresh_temperatures(self, geometry, phonon):
        '''Refresh energies and temperatures while enforcing boundary conditions as given by geometry.'''
        self.slice_temperature = np.empty(self.n_of_slices)
        self.slice_energy = np.ones(self.n_of_slices)*phonon.zero_point # eV/a³
        self.slice_heat_flux = np.empty(self.n_of_slices)
        self.slice_N_p = np.empty(self.n_of_slices)

        self.slice_id = -np.ones((self.positions.shape[0]))

        self.energies = self.occupation*self.omega*self.hbar    # eV

        for i in range(self.n_of_slices):   # assign temperatures for particles in each slice
            indexes = (self.positions[:, self.slice_axis] >= i*self.slice_length) & (self.positions[:, self.slice_axis] < (i+1)*self.slice_length)

            self.slice_id[indexes] = i

            self.slice_N_p[i] = int(indexes.sum().astype(int))

            self.slice_energy[i] += phonon.normalise_to_density(self.energies[indexes].sum()/self.particles_pmps) # eV/a³

            if self.slice_energy[i] > phonon.energy_array.max():
                self.slice_temperature[i] = 1000
            elif self.slice_energy[i] > phonon.energy_array.min():
                self.slice_temperature[i] = phonon.temperature_function( self.slice_energy[i] )
            else:
                self.slice_temperature[i] = 0

            self.temperatures[indexes] = copy.deepcopy(self.slice_temperature[i])
            self.slice_heat_flux[i] = self.calculate_heat_flux(i, indexes, geometry, phonon)

    def calculate_heat_flux(self, slice_id, indexes, geometry, phonon):

        group_vel = self.group_vel[indexes, self.slice_axis]   # angstrom * THz = angs / ps

        omega      = self.omega[indexes]        # THz rad
        occupation = self.occupation[indexes]   # []

        energies  = self.hbar*omega*(occupation+0.5)  # eV  # should we include zero-point here??
        
        heat_flux = (group_vel*energies).sum()                  # eV angs / ps
        heat_flux = heat_flux/self.particles_pmps               # eV angs / ps
        heat_flux = heat_flux/geometry.slice_volume[slice_id]   # eV / ps angs^2
        heat_flux = heat_flux*self.eVpsa2_in_Wm2                # W/m²

        return heat_flux

    def drift(self, geometry):
        '''Drift operation.'''

        self.positions += self.group_vel*self.dt

        self.positions_res1 += self.group_vel_res1*self.dt
        self.positions_res2 += self.group_vel_res2*self.dt

    def find_boundary(self, positions, velocities, geometry):
        '''Finds which mesh triangle will be hit by the particle, given an initial position and velocity
        direction. It works with individual particles as well with a group of particles.
        Returns: array of faces indexes for each particle'''

        boundary_pos, index_ray, _ = geometry.mesh.ray.intersects_location(ray_origins      = positions ,
                                                                            ray_directions   = velocities,
                                                                            multiple_hits    = False     )

        no_ray = ~np.in1d(np.arange(positions.shape[0]), index_ray) # find which particles do not have rays
        stationary = (np.linalg.norm(velocities, axis = 1) == 0) # find which particles have zero velocity

        on_boundary =  ~stationary & no_ray # non-stationary particles without rays are on the boundary

        all_boundary_pos                 = np.zeros( (positions.shape[0], 3) )
        all_boundary_pos[stationary,  :] = np.inf                       # those stationary have infinite collision position
        all_boundary_pos[on_boundary, :] = positions[on_boundary, :]    # those on the boundary are colliding
        all_boundary_pos[~no_ray,     :] = boundary_pos                 # those with ray have correspondent collision position

        return all_boundary_pos

    def timesteps_to_boundary(self, positions, velocities, geometry):
        '''Calculate how many timesteps to boundary scattering.'''
        
        boundary_pos = self.find_boundary(positions, velocities, geometry)

        # calculate distances for given particles
        boundary_dist = np.linalg.norm( positions - boundary_pos, axis = 1 )

        ts_to_boundary = boundary_dist/( np.linalg.norm(velocities, axis = 1) * self.dt )

        ts_to_boundary = np.floor(ts_to_boundary)    # such that particle hits the boundary when no_timesteps == 0 (crosses boundary)

        return ts_to_boundary
    
    def delete_particles(self, indexes):
        '''Delete all information about particles according to the given indexes'''

        self.positions          = np.delete(self.positions         , indexes, axis = 0)
        self.q_points           = np.delete(self.q_points          , indexes, axis = 0)
        self.group_vel          = np.delete(self.group_vel         , indexes, axis = 0)
        self.omega              = np.delete(self.omega             , indexes, axis = 0)
        self.energies           = np.delete(self.energies          , indexes, axis = 0)
        self.occupation         = np.delete(self.occupation        , indexes, axis = 0)
        self.temperatures       = np.delete(self.temperatures      , indexes, axis = 0)
        self.n_timesteps        = np.delete(self.n_timesteps       , indexes, axis = 0)
        self.modes              = np.delete(self.modes             , indexes, axis = 0)
    
    def boundary_scattering(self, geometry):
        '''Applies boundary scattering or other conditions to the particles where it happened, given their indexes.'''

        indexes = self.n_timesteps <= 0   # find all boundary scattering particles

        positions = self.positions[indexes, :]
        modes     = self.modes[indexes, :]
        reference = np.where(indexes)[0]
        group_vel = self.group_vel[indexes, :]

        if (self.bound_cond == 'periodic') and (geometry.shape == 'cuboid'):    # applicable just for cuboids

            check = positions >= geometry.bounds.reshape(2, 1, 3)    # True if plus or equal than each bound value

            check = check.sum(axis = 0).round().astype(int) # this will result in 0 for numbers out of the lower limit, 1 for points inside the limits, 2 for outside de upper limit

            # deleting particles that are in the reservoirs
            indexes_del = (check[:, self.slice_axis] != 1)

            positions = positions[~indexes_del, :]
            modes     = modes[~indexes_del, :]
            check     = check[~indexes_del, :]
            group_vel = group_vel[~indexes_del, :]

            # applying periodicity

            lower_points = np.where(check == 0, geometry.bounds[1,:] - (np.abs(positions) % np.ptp(geometry.bounds, axis = 0)), 0)
            in_points    = np.where(check == 1, positions                                                           , 0)
            upper_points = np.where(check == 2, geometry.bounds[0,:] + (positions % np.ptp(geometry.bounds, axis = 0)), 0)

            new_positions = lower_points + in_points + upper_points

            new_ts_to_boundary = self.timesteps_to_boundary(new_positions, group_vel, geometry)  # calculates new ts until boundary scattering
        
            indexes = np.delete(indexes, reference[indexes_del])
            self.delete_particles(reference[indexes_del])
            
            self.positions[indexes, :] = new_positions
            self.n_timesteps[indexes] = new_ts_to_boundary

    def lifetime_scattering(self, geometry, phonon):
        '''Performs lifetime scattering.'''

        # N_as = N_ad + dt/tau (N_BE(T*) - N_ad)

        for slc in range(self.n_of_slices):
        
            T = self.slice_temperature[slc]     # slice temperature
            
            indexes = (self.slice_id == slc)    # identifying particles
            
            if indexes.sum()>0: # if slice is not empty

                modes   = self.modes[indexes, :]    # getting their modes
                omega   = self.omega[indexes]

                func = partial(self.get_lifetime, T = T, phonon = phonon)   # defining partial function

                tau = np.array(list(map(func, modes)))  # mapping lifetime
                
                occupation_ad = copy.deepcopy(self.occupation[indexes]) # getting current occupation
                occupation_BE = phonon.calculate_occupation(T, omega)   # calculating Bose-Einstein occcupation

                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    occupation_as = np.where(tau>0, occupation_ad + (self.dt/tau) *(occupation_BE - occupation_ad), occupation_ad)
                    # occupation_as = np.where(occupation_as < 0, 0, occupation_as)
                    self.occupation[indexes] = occupation_as

    def get_lifetime(self, mode, T, phonon):
        function = phonon.lifetime_function[mode[0]][mode[1]]
        return function(T)          

    def run_timestep(self, geometry, phonon):

        self.drift(geometry)                            # drift particles

        self.add_reservoir_particles(geometry, phonon)  # add reservoir particles that come in the domain

        self.boundary_scattering(geometry)              # perform boundary scattering/periodicity and particle deletion

        self.refresh_temperatures(geometry, phonon)     # refresh cell temperatures

        self.lifetime_scattering(geometry, phonon)      # perform lifetime scattering

        self.fill_reservoirs(phonon)                    # refill reservoirs

        self.n_timesteps -= 1                           # -1 timestep in the counter to boundary scattering

        self.current_timestep += 1                      # +1 timestep index

        self.t = self.current_timestep*self.dt          # 1 dt passed

        if  ( self.current_timestep % 100) == 0:
            print('Timestep {:>5d}'.format( int(self.current_timestep) ) )
            # self.write_modes_data()
        if ( self.current_timestep % 1) == 0:
            self.write_convergence()      # write data on file

        if len(self.rt_plot) > 0:
            self.plot_real_time(geometry)
                
    def init_plot_real_time(self, geometry, phonon):
        '''Initialises the real time plot to be updated for each timestep.'''

        if self.rt_plot[0] in ['T', 'temperature']:
            colors = self.temperatures
            vmin = 0# np.floor(self.T_boundary.min()-5)
            vmax = np.ceil(self.T_boundary.max()+5)
        elif self.rt_plot[0] in ['e', 'energy']:
            colors = self.energies
            vmin = self.energies[self.energies>0].min()
            vmax = self.hbar*self.omega.max()*(0.5+1/(np.exp(self.hbar*self.omega.max()/(self.T_boundary.max()*self.kb)) - 1) )
        elif self.rt_plot[0] in ['omega', 'angular_frequency']:
            colors = self.omega
            vmin = self.omega[self.omega>0].min()
            vmax = self.omega.max()
        elif self.rt_plot[0] in ['n', 'occupation']:
            colors = self.occupation
            vmin = 0
            order = np.floor( np.log10( self.occupation.max()) )
            vmax = (10**order)*np.ceil(self.occupation.max()/(10**order))
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
            vmax = 1000 # max_path/(min_vel*self.dt)

        # plt.ion()

        # Think about a way to make the animation "pretty" with a proper aspect ratio depending on dimensions of the domain

        # box_size = np.ptp(geometry.bounds, axis = 0)
        # figsize = np.array([box_size.max()/2, box_size.min()])
        # figsize = figsize*8/(box_size.max()/2)

        figsize = (8,8)

        fig = plt.figure(figsize = figsize, dpi = 150)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect( np.ptp(geometry.bounds, axis = 0) )
        ax.set_xlim(geometry.bounds[:, 0])
        ax.set_ylim(geometry.bounds[:, 1])
        ax.set_zlim(geometry.bounds[:, 2])

        graph = ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], s = 1, vmin = vmin, vmax = vmax, c = colors, cmap = self.colormap)
        fig.colorbar(graph)

        plt.tight_layout()

        fig.canvas.draw()

        self.plot_images = [np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape(fig.canvas.get_width_height()[::-1]+(3,))]

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
        
        self.rt_graph._offsets3d = [self.positions[:,0], self.positions[:,1], self.positions[:,2]]

        self.rt_graph.set_array(colors)

        self.rt_fig.canvas.draw()
        self.rt_fig.canvas.flush_events()

        self.plot_images += [np.fromstring(self.rt_fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '').reshape( self.rt_fig.canvas.get_width_height()[::-1]+(3,) ) ]

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

    def open_convergence(self,
                         timestep   = True,
                         sim_time   = True,
                         avg_energy = True,
                         N_p        = True,
                         slice_T    = True,
                         slice_e    = True,
                         slice_phi  = True,
                         slice_N_p  = True,
                         real_time  = True):

        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        line = ''

        if real_time:
            line += 'Real Time                  '
        if timestep:
            line += 'Timest. '
        if sim_time:
            line += 'Simula. Time '
        if avg_energy:
            line += 'Average Ene. '
        if N_p:
            line += 'No. Part. '
        if slice_T:
            for i in range(self.n_of_slices):
                line += 'T Slc {:>3d} '.format(i)
        if slice_e:
            for i in range(self.n_of_slices):
                line += 'Energ Sl {:>2d} '.format(i) # temperature per slice
        if slice_phi:
            for i in range(self.n_of_slices):
                line += 'Hflux Sl {:>2d} '.format(i)
        if slice_N_p:
            for i in range(self.n_of_slices):
                line += 'Np Sl  {:>2d} '.format(i)
        
        line += '\n'

        self.f.write(line)
        self.f.close()

    def write_convergence(self,
                          timestep   = True,
                          sim_time   = True,
                          avg_energy = True,
                          N_p        = True,
                          slice_T    = True,
                          slice_e    = True,
                          slice_phi  = True,
                          slice_N_p  = True,
                          real_time  = True):
        line = ''

        if real_time:
            line += datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f ')  # real time in ISO format
        if timestep:
            line += '{:>8d} '.format( int(self.current_timestep) )   # timestep
        if sim_time:
            line += '{:>11.5e} '.format( self.t )   # time
        if avg_energy:
            line += '{:>11.5e} '.format( self.energies.mean() )  # average energy
        if N_p:
            line += '{:>10d}'.format( self.energies.shape[0] )  # number of particles
        if slice_T:
            for i in range(self.n_of_slices):
                line += '{:>9.3f} '.format( self.slice_temperature[i] ) # temperature per slice
        if slice_e:
            for i in range(self.n_of_slices):
                line += '{:>11.5e} '.format( self.slice_energy[i] ) # temperature per slice
        if slice_phi:
            for i in range(self.n_of_slices):
                line += '{:>12.5e} '.format( self.slice_heat_flux[i] ) # heat flux per slice
        if slice_N_p:
            for i in range(self.n_of_slices):
                line += '{:>10d}'.format( int(self.slice_N_p[i]) )
        
        line += '\n'

        filename = self.results_folder_name+'convergence.txt'

        self.f = open(filename, 'a+')

        self.f.writelines(line)

        self.f.close()
    
    def plot_modes_histograms(self, phonon):

        columns = 5
        rows = int(np.ceil(self.n_of_slices/columns))

        self.hist_fig = plt.figure(figsize = (columns*3, rows*3) )
        
        ax = []

        for i in range(self.n_of_slices):
            ax += [self.hist_fig.add_subplot(rows, columns, i+1)]

            data = []

            for j in range(phonon.number_of_branches):

                indexes = (self.modes[:,1] == j) & (self.slice_id == i)

                data += [self.modes[indexes, 0]]

            ax[i].hist(data, stacked = True, histtype = 'bar', density = True, bins = int(phonon.number_of_qpoints/20) )
    
        plt.tight_layout()
        plt.savefig(self.results_folder_name+'time_{:>.3e}'.format(self.t)+'.png')
        # plt.show()

        plt.clf()

    def write_modes_data(self):
        '''Write on file the mode distribution in a given timestep (qpoint, branch, slice).'''

        filename = self.results_folder_name + 'time_{:>.3e}'.format(self.t)+'.txt'

        data = np.hstack((self.modes, self.slice_id.reshape(-1, 1)))

        np.savetxt(filename, data, '%4d %2d %2d')
    
    def write_final_state(self):

        print('Saving end of run particle data...')
        
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        
        # saving final particle states: modes, positions, slice id and occupation number.
        # Obs.: phonon properties of particles can be retrieved by mode information and slice temperature

        filename = self.results_folder_name + 'particle_data.txt'

        header ='Particles final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'q-point, branch, pos x [angs], pos y [angs], pos z [angs], slice, occupation'

        data = np.hstack( (self.modes,
                           self.positions,
                           self.slice_id.reshape(-1, 1),
                           self.occupation.reshape(-1, 1)) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %d, %.3f, %.3f, %.3f, %d, %.6e', delimiter = ',', header = header)

        # saving final slice states: temperatures and heat flux

        filename = self.results_folder_name + 'slice_data.txt'

        header ='Slices final state data \n' + \
                'Date and time: {}\n'.format(time) + \
                'hdf file = {}, POSCAR file = {}\n'.format(self.args.hdf_file, self.args.poscar_file) + \
                'slice id, temperature [K], heat flux [W/m^2]'

        data = np.hstack( (np.arange(self.n_of_slices).reshape(-1, 1),
                           self.slice_temperature.reshape(-1, 1),
                           self.slice_heat_flux.reshape(-1, 1),
                           self.slice_volume.reshape(-1, 1)) )
        
        # comma separated
        np.savetxt(filename, data, '%d, %.3f, %.3e, %.3e', delimiter = ',', header = header)

        

        
                
