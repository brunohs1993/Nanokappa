import numpy as np
import numba
from numba import jit, njit

@njit
def q_to_k_jit(q, rl):
    '''Convert q-points to wave vectors in the first brillouin zone.
        q  (N, 3): nparray of qpoints;
        rl (3, 3): reciprocal lattice basis matrix.'''
    k = np.dot(q, rl.T)

    return k

@njit
def k_to_q_jit(k, rl, data_mesh):
        '''Convert wave vectors to q-points in the first brillouin zone, with margin for interpolation.
        k  (N, 3): nparray of wavevectors;
        rl (3, 3): reciprocal lattice basis matrix;
        data_mesh (3,): number of discretisation points of FBZ in each direction.'''
        
        # convert wave vectors to q-points in the first brillouin zone
        a = np.linalg.inv(rl)  # transformation vector
        
        q = np.dot(k, a.T)

        # bring all points to the first brillouin zone
        q = q % 1

        # adjust for nearest interpolation
        q = np.where(q >=  1-1/(2*data_mesh), q-1, q)
        q = np.where(q <    -1/(2*data_mesh), q+1, q)

        return q

@njit
def calculate_specularity_jit(n, k, eta, v):
    '''Calculate specularity for one particles:
    normals (N,3): nparray of inward normals;
    q       (N,3): nparray of qpoints;
    rl      (3,3): reciprocal lattice basis matrix;
    eta     (N, ): nparray of roughness for each particle;
    v       (N,3): nparray of group velocities;
    return_cos (bool), optional: If needs to return a list of cosine of the angle of incidence.
    
    Returns:
    specularity (N,): nparray of specularity for each particle;
    cos_theta   (N,): nparray of cos theta for each particle.   
    '''
    k_norm = np.sum(k**2)**0.5

    v_norm = np.sum(v**2)**0.5

    dot = np.sum(v*n)
    cos_theta = dot/v_norm

    p = np.exp(-(2*eta*k_norm*cos_theta)**2)

    return p

@njit
def nearest_interpolator_jit(x, y, p):

    d = np.sum((x-p)**2, axis = 1)**0.5

    i = np.where(d == d.min())[0][0]
    
    return y[i]

@jit
def calculate_diffuse_probs_jit(n, eta, q_all, v_all, rl, data_mesh):
    '''Calculate picking probabilities for a given facet'''

    n_q = v_all.shape[0]
    n_b = v_all.shape[1]

    prob = np.zeros((n_q, n_b), dtype = np.double)

    for iq in range(n_q):
        q = q_all[iq, :]
        k = k_to_q_jit(q, rl, data_mesh)
        for ib in range(n_b):
            v = v_all[iq, ib, :]
            if (v*n).sum()>0:
                p = calculate_specularity_jit(n, k, eta, v)
                
                prob[iq, ib] = (1-p)*np.sum(v*n)
    
    prob = prob/np.sum(prob)

    return prob

@njit(numba.types.Tuple((numba.int32[:, :], numba.boolean[:])) # output
      (numba.float64[:, :],                                    # k_in
       numba.float64[:, :],                                    # n
       numba.float64[:],                                       # eta
       numba.float64[:, :],                                    # v_in
       numba.float64[:],                                       # omega_in
       numba.float64[:, :],                                    # conc_unique
       numba.int64[:],                                         # inv_i
       numba.float64[:, :],                                    # q_all
       numba.float64[:, :],                                    # omega_all
       numba.float64[:, :, :],                                 # v_all
       numba.float64[:, :],                                    # rl
       numba.int32[:]),                                        # data_mesh
       parallel = True)
def select_reflected_modes_jit(k_in, n, eta, v_in, omega_in, # particle inputs
                               conc_unique, inv_i,           # and geometry inputs
                               q_all, omega_all, v_all, rl, data_mesh # inputs from phonon class
                               ):
    '''Calculates reflection for several particles.'''

    n_p = k_in.shape[0]

    # first, calculate for every unique combination of normal and roughness
    n_f = conc_unique.shape[0]
    n_q = omega_all.shape[0]
    n_b = omega_all.shape[1]

    prob = np.zeros((n_f, n_q, n_b))

    for iif in range(n_f):
        prob[iif, :, :] = calculate_diffuse_probs_jit(conc_unique[iif, 0:3], conc_unique[iif, 3], q_all, v_all, rl, data_mesh)

    new_modes    = np.zeros((n_p, 2)).astype(np.int_)
    indexes_spec = np.zeros(n_p).astype(np.bool_)

    # print('Entered loop')

    for ip in range(n_p):
        # print(ip, n_p)

        p = calculate_specularity_jit(n[ip, :], k_in[ip, :], eta[ip], v_in[ip, :])

        # print('Calculated specularity')
        r = np.random.rand()

        if r <= p: # specular
            indexes_spec[ip] = True

            # print('Marked spec index')

            k_try = k_in[ip, :] - 2*n[ip, :]*(n[ip, :]*k_in[ip, :]).sum()

            # print('Calc k_try')

            q_try = k_to_q_jit(k_try, rl, data_mesh)

            # print('Converted to q_try')

            new_qpoint = nearest_interpolator_jit(q_all[1:, :], np.arange(1, n_q), q_try)

            # print('got new qpoint')

            omega_diff = np.absolute(omega_all[new_qpoint, :] - omega_in[ip])

            new_branch = np.where( omega_diff == omega_diff.min())[0][0]

            # print('got new branch')

        else: # diffuse

            i_unique = inv_i[ip]

            # print('calc diff prob')

            # nq = omega_all.shape[0]
            nb = omega_all.shape[1]
            
            roulette = np.cumsum(prob[i_unique, :, :])

            # print('calc roulette')

            rd = np.random.rand()

            arg = rd <= roulette
            flat_i = 0
            while not arg[flat_i]:
                flat_i = flat_i+1

            # print('got flat ind')
            
            new_qpoint = int(np.floor(flat_i/nb))
            new_branch = int(flat_i - new_qpoint*nb)

            # print('new qpoint and branch')

        new_modes[ip, 0] = new_qpoint
        new_modes[ip, 1] = new_branch

        # print('registered on new_modes array')

    # print('out of loop')
    
    return new_modes, indexes_spec

