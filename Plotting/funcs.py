# Standard library imports
import time
from functools import partial

# External package imports
import numpy as np
import kwant
from pfapack.ctypes import pfaffian as cpf
from scipy.ndimage import shift
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.stats import multivariate_normal
from scipy import optimize, signal
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import pickle

# Internal imports
import systems
import tools

default_params = dict(
    Smag_imp=0,
    exp=np.exp,
    re=np.real,
    im=np.imag,
    phi_0=1.0,
    hbar=1
)

Bi2Se3 = dict(
        A_perp=4.1,
        A_z=2.2,
        M_0=0.28,
        M_perp=56.6,
        M_z=10,
        C_0=-0.0068,
        C_perp=-19.6,
        C_z=-1.3,
        m_z=0,
        S_imp=0,
        mu_ti=0, 
)

#second variant of parameters
# NOT IN USE
Bi2Se3_2 = dict(
        A_perp=3.33,
        A_z=2.26,
        M_0=0.28,
        M_perp=44.5,
        M_z=6.86,
        C_0=-0.0083,
        C_perp=-30.4,
        C_z=-5.74,
        m_z=0,    
        S_imp=0,
        mu_ti=0,
)
# NOT IN USE
Bi2Te3 = dict(
        A_perp=2.87,
        A_z=0.30,
        M_0=0.30,
        M_perp=57.38,
        M_z=2.79,
        C_0=-0.18,
        C_perp=-49.68,
        C_z=-6.55, 
        m_z=0,    
        S_imp=0,
        mu_ti=0,
)
# NOT IN USE
Sb2Te3 = dict(
        A_perp=3.40,
        A_z=0.84,
        M_0=0.22,
        M_perp=48.51,
        M_z=19.64,
        C_0=0.001,
        C_perp=10.78,
        C_z=12.39,
        m_z=0,    
        S_imp=0,
        mu_ti=0,
)

Bi2Se3_8band = dict(
        P_1 = 3.33,
        Q_1 = 2.26,
        P_2 = 2.84,
        Q_2 = 2.84,
        P_3 = -2.62,
        Q_3 = 2.48,
        F_1 = 3.73,
        K_1 = 6.52,
        F_3 = -1.12,
        K_3 = -14,
        F_5 = 1.5,
        K_5 = -3.11,
        F_7 = 2.71,
        K_7 = -5.08,
        U_35= -2.31-7.45*1j,
        V_35= -1.05-5.98*1j,
        F_37= 2.47,
        K_37= -8.52,
        U_47= -7.86,
        V_47= -8.95*1j,
        U_58= -2.31-2.57*1j,
        V_58= -0.64-4.29*1j,
        E_1 = -0.29,
        E_3 = 0.28,
        E_5 = -0.57,
        E_7 = -0.98,
        m_z=0,    
#        S_imp=0,
        mu_ti=0,    
        conjugate=np.conj
)

Bi2Te3_8band = dict(
        P_1 = 2.87,
        Q_1 = 0.30,
        P_2 = 2.68,
        Q_2 = 2.68,
        P_3 = -1.94,
        Q_3 = 1.23,
        F_1 = 7.16,
        K_1 = 3.72,
        F_3 = 3.76,
        K_3 = -7.70,
        F_5 = -0.62,
        K_5 = -7.17,
        F_7 = 3.77,
        K_7 = 22.27,
        U_35= -2.21-9.85*1j,
        V_35= -2.43-3.53*1j,
        F_37= 4.39,
        K_37= -6.50,
        U_47= -4.29,
        V_47= -0.83*1j,
        U_58= -0.24-3.69*1j,
        V_58= -0.85-6.64*1j,
        E_1 = -0.48,
        E_3 = 0.12,
        E_5 = -0.63,
        E_7 = -1.18,
        m_z=0,    
#        S_imp=0,
        mu_ti=0,    
        conjugate=np.conj
)

Sb2Te3_8band = dict(
        P_1 = 3.40,
        Q_1 = 0.84,
        P_2 = 3.19,
        Q_2 = 3.19,
        P_3 = -2.46,
        Q_3 = 2.11,
        F_1 = 3.82,
        K_1 = 2.49,
        F_3 = -32.02,
        K_3 = -59.28,
        F_5 = -2.26,
        K_5 = -13.00,
        F_7 = 5.04,
        K_7 = 2.40,
        U_35= -11.31-46.00*1j,
        V_35= -4.50-22.80*1j,
        F_37= 16.96,
        K_37= -24.17,
        U_47= -45.46,
        V_47= -17.64*1j,
        U_58= -2.01-3.98*1j,
        V_58= 1.28-9.02*1j,
        E_1 = -0.22,
        E_3 = 0.22,
        E_5 = -0.88,
        E_7 = -1.51,
        m_z=0,    
#        S_imp=0,
        mu_ti=0,    
        conjugate=np.conj
)

Bi2Se3_4band = dict(
        A_0=3.33,
        B_0=2.26,
        M_0=-0.28,
        M_2=44.5,
        M_1=6.86,
        C_0=-0.0083,
        C_2=30.4,
        C_1=5.74,
        R_1=50.6,
        R_2=-113.3,
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj    
)

Bi2Te3_4band = dict(
        A_0=2.87,
        B_0=0.30,
        M_0=-0.30,
        M_2=57.38,
        M_1=2.79,
        C_0=-0.18,
        C_2=49.68,
        C_1=6.55,
        R_1=45.02,
        R_2=-89.37,      
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj    
)

Sb2Te3_4band = dict(
        A_0=3.40,
        B_0=0.84,
        M_0=-0.22,
        M_2=48.51,
        M_1=19.64,
        C_0=0.001,
        C_2=-10.78,
        C_1=-12.39,
        R_1=-244.67,
        R_2=-14.45,    
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj    
)

toy_A = dict(
        A_0=3,
        B_0=3,
        M_0=-0.3,
        M_1=15,
        M_2=15,
        C_0=0,
        C_1=-5,
        C_2=0,
        R_1=0,
        R_2=0,    
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj  
)

Bi2Se3_nechaev = dict(
        A_0=2.51,
        B_0=1.83,
        M_0=-0.17,
        M_1=3.35,
        M_2=29.35,
        C_0=0.048,
        C_1=1.41,
        C_2=13.9,
        R_1=0,
        R_2=0,    
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj  
)

Bi2Te3_nechaev = dict(
        A_0=4,
        B_0=0.9,
        M_0=-0.3,
        M_1=9.25,
        M_2=177.23,
        C_0=-0.12,
        C_1=2.67,
        C_2=154.35,
        R_1=0,
        R_2=0,
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj  
)

Sb2Te3_nechaev = dict(
        A_0=3.7,
        B_0=1.17,
        M_0=-0.18,
        M_1=22.12,
        M_2=51.28,
        C_0=0.02,
        C_1=-14.2,
        C_2=-6.97,
        R_1=0,
        R_2=0,    
        m_z=0,    
        S_imp=0,
        mu_ti=0,
        conjugate=np.conj  
)

param_list = [Bi2Se3, Bi2Se3_2, Bi2Te3, Sb2Te3, Bi2Se3_4band, Bi2Te3_4band, Sb2Te3_4band, Bi2Se3_8band, Bi2Te3_8band, Sb2Te3_8band, toy_A]

fnames = [
           'Bi2Se3_max_k0.3_points_x30_points_z95_Cheng',
           'Bi2Te3_max_k0.3_points_x15_points_z40_Cheng',
           'Sb2Te3_max_k0.3_points_x30_points_z40_Cheng',
           'Bi2Se3_max_k0.3_points_x20_points_z90_Philipp',
           'Bi2Te3_max_k0.3_points_x15_points_z75_Philipp',
           'Sb2Te3_max_k0.3_points_x20_points_z40_Philipp',
]
#indexes 11,12,13

for fname in fnames:

    path = "fit_"+fname+".pkl"

    with open(path, 'rb') as f:
        params = pickle.load(f)
        params.update(C_0=0, R_1=0, R_2=0, m_z=0, S_imp=0, mu_ti=0, conjugate=np.conj)
        param_list.append(params)
        
param_list = param_list+[Bi2Se3_nechaev, Bi2Te3_nechaev, Sb2Te3_nechaev]

#create BST: (Bi0.8Sb1.2)2Te3
path_bite = "fit_"+fnames[1]+".pkl"
path_sbte = "fit_"+fnames[2]+".pkl"
#index 14
with open(path_bite, 'rb') as f:
    params_bite = pickle.load(f)
    
with open(path_sbte, 'rb') as f:
    params_sbte = pickle.load(f)

def sum_dict(d1, d2, weigth):
    for key, value in d1.items():
        d1[key] = value*weigth + d2.get(key, 0)*(1-weigth)
    return d1

params_bst = sum_dict(params_bite, params_sbte, 0.8/2)
params_bst.update(C_0=0, R_1=0, R_2=0, m_z=0, S_imp=0, mu_ti=0, conjugate=np.conj)
    
#param_list.append(params_bst)        

#Get correct lattice constants
with open('a_optimal_90.pkl', 'rb') as f:
    a_optimal_90 = pickle.load(f)
with open('a_optimal_80.pkl', 'rb') as f:
    a_optimal_80 = pickle.load(f)
with open('H_eff_parameters.pkl', 'rb') as f:
    H_eff_parameters = pickle.load(f)    
sigma_0 = np.identity(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def get_correct_lattice_constant(params):
    try:
        pos = np.where( [d['h_mode'] == params['h_mode'][0] and d['p_mode'] == params['p_mode'][0] 
                         for d in a_optimal_80] )[0][0]
        params['a'] = [params['W'][0] / np.ceil(params['W'][0]/a_optimal_80[pos]['a'])]
    except:
        pos = np.where( [d['h_mode'] == params['h_mode'] and d['p_mode'] == params['p_mode'] 
                         for d in a_optimal_80] )[0][0]
        params['a'] = params['W'] / np.ceil(params['W']/a_optimal_80[pos]['a'])
    return params

def optimize_parameters(a, a_z, W, T, mu, m_z, u_B, params, sym, h_mode, p_mode):
    
    if h_mode == 1:
        param_pos = p_mode*6+(T-1)
        H_params = H_eff_parameters[param_pos]
        params['m0']   = H_params['Delta']/2
        params['m1']   = H_params['B']
        params['D']    = H_params['D']
        params['v_F']  = H_params['v_F']
    elif 0 <= p_mode <= 3:
        h_mode = 0
    elif 10 <= p_mode <= 13:
        h_mode = 3    
    
    a_dict = a_optimal_80
    if a == -80:
        a_dict = a_optimal_80
    elif a == -90:
        a_dict = a_optimal_90

    if a < 0 and (h_mode == 0 or h_mode == 3):
        pos = np.where( [d['h_mode'] == h_mode and d['p_mode'] == p_mode for d in a_dict] )[0][0]
        if (h_mode == 0 and p_mode < 4) or (h_mode == 3 and p_mode > 9):        
            a = W / np.ceil(W/a_dict[pos]['a'])
        else:
            a = W / np.ceil(W/10)
            
    a_dict = a_optimal_80            
    if a_z == -80:
        a_dict = a_optimal_80
    elif a_z == -90:
        a_dict = a_optimal_90            

    if (a_z < 0 and a_z > -100) and (h_mode == 0 or h_mode == 3):
        points_z = abs(a_z)
        pos = np.where( [d['h_mode'] == h_mode and d['p_mode'] == p_mode for d in a_dict] )[0][0]
        if (h_mode == 0 and p_mode < 4) or (h_mode == 3 and p_mode > 9):
            max_a_z = T / np.ceil(T/a_dict[pos]['a_z'])
        else:
            max_a_z = T / np.ceil(T/5)
        if T/points_z > max_a_z or a_z == -80 or a_z == -90:
            a_z = max_a_z
        else: 
            a_z = T / points_z
    elif a_z <= -100: 
        points_z = abs(a_z)
        a_z = T / points_z
    else:
        a_z = T / np.ceil(T/a_z)
        
    if sym and h_mode == 0:
        params['C_0'] = 0
        params['C_z'] = 0
        params['C_perp'] = 0
    if sym and h_mode == 3:
        params['C_0'] = 0
        params['C_1'] = 0
        params['C_2'] = 0
    
    if h_mode == 1 and sym:
        params['D'] = 0
        
    if h_mode == 0:
        C_0 = params['C_0']
        M_0 = params['M_0']
        C_1 = params['C_z']
        M_1 = params['M_z']  
        C_2 = params['C_perp']
        M_2 = params['M_perp']        
    elif h_mode == 3:
        C_0 = params['C_0']
        M_0 = params['M_0']
        C_1 = params['C_1']
        M_1 = params['M_1']
        C_2 = params['C_2']
        M_2 = params['M_2']        
        
    if mu == -100 and (h_mode == 0 and p_mode < 4) or (h_mode == 3 and p_mode > 9):
        mu = C_0 - M_0*(C_1/M_1)
    elif mu == -200 and (h_mode == 0 and p_mode < 4) or (h_mode == 3 and p_mode > 9):
        mu = C_0 + ( - M_0*(C_1/M_1) - M_0*(C_2/M_2) )/2
    
    if mu == -100 and h_mode == 1:
        m0 = params['m0']
        D  = params['D']
        m1 = params['m1']
        if abs(m_z) > abs(m0) or m0*m1 > 0: #When the system is in the QAH, or QSH phase
            mu = -(m0-m_z)*D/m1# - m_z/2*D*m1/abs(D*m1) + 1e-3
        else:
            mu = 0
        
    if h_mode != 1:
        T=T*(1+1e-8)
        
    return a, a_z, W, T, mu, m_z, params, h_mode, p_mode

def delta_profile_funtion(T):
    def d(delta, x, y, z):
        if z > T - 10:
            return delta
        else:
            return 0.0
    return d

def u_profile_funtion(T):
    def u(u_B, x, y, z):
        if z < 10:
            return u_B
        else:
            return 0.0
    return u


def delta_profile_funtion_fitite_ribbon(T, L, L_nodelta):
    def d(delta, x, y, z):
        if (z == T and x >= L_nodelta and x <= L-L_nodelta):
            return delta
        else:
            return 0.0
    return d


def delta_profile_funtion_all():
    def d(delta, x, y, z):
        return delta
    return d


def convolve_gaussian_random_field(
        seed, size, correlation_length, normalize=True):
    d = len(size)
    pos = np.mgrid[tuple(slice(-3*correlation_length, 3*correlation_length+1)
                         for i in range(d))]
    cov = np.diag([correlation_length**2 for i in range(d)])
    filter_kernel = multivariate_normal.pdf(pos.T, mean=None, cov=cov)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size)
    gfield = signal.fftconvolve(noise, filter_kernel, mode='same')

    if normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)

    return gfield


def syst_disorder_box(seed, syst, a, correlation_length, normalize=True):
    syst_index = np.array([s.tag for s in syst.sites])
    syst_index_box = syst_index - np.amin(syst_index, 0)
    syst_box_shape = np.amax(syst_index, 0) - np.amin(syst_index, 0) + 1
    disorder_box = convolve_gaussian_random_field(
        seed, syst_box_shape, correlation_length//a,
        normalize=False)
    not_syst_box = np.full(syst_box_shape, True)
    not_syst_box[tuple(syst_index_box.T)] = False
    disorder_box[not_syst_box] = np.nan

    if normalize:
        disorder_box = disorder_box - np.nanmean(disorder_box)
        disorder_box = disorder_box/np.nanstd(disorder_box)

    return disorder_box


def get_S_imp(seed, a, syst, correlation_length):
    disorder_box = syst_disorder_box(seed, syst, a, correlation_length)
    syst_index = np.array([s.tag for s in syst.sites])

    def S_imp(site, Smag_imp):
        ind_x, ind_y, ind_z = site.tag - np.amin(syst_index, 0)
        return Smag_imp * disorder_box[ind_x, ind_y, ind_z]

    return S_imp


def cell_mats(lead, params, bias=0, sparse=False):
    h = lead.cell_hamiltonian(params=params, sparse=sparse)
    if sparse:
        h -= bias * sp.identity(h.shape[0])
    else:
        h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params, sparse=sparse)
    return h, t


def get_h_k(lead, params, bias=0, sparse=False):
    h, t = cell_mats(lead, params, bias, sparse)

    def h_k(k):
        return h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)

    return h_k


def sort_spectrum(energies, wfs):
    index = np.argsort(energies)
    return energies[index], wfs[:, index]


def fix_shift(energies, wfs):
    k = energies.shape[0]
    shift_intex = k//2 - energies[energies < 0].shape[0]
    energies = shift(energies, shift_intex, cval=np.NaN)
    wfs = shift(wfs, (0, shift_intex), cval=np.NaN)
    return energies, wfs


def bands(k, lead, params, num_bands=20, sigma=0, sort=True):
    h_k = get_h_k(lead, params, bias=0, sparse=True)
    ham = h_k(k) 

    if np.isnan(ham.data).any():
        raise Exception(f'{params}')

    if num_bands is None:
        energies, wfs = np.linalg.eig(ham.todense())
    else:
        energies, wfs = sla.eigs(ham, k=num_bands, sigma=sigma)

    if sort:
        energies, wfs = sort_spectrum(energies, wfs)
        energies, wfs = fix_shift(energies, wfs)

    return energies, wfs


def translation_ev(h, t, tol=1e6):
    """Compute the eigenvalues of the translation operator of a lead.
    Adapted from kwant.physics.leads.modes.
    Parameters
    ----------
    h : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    t : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.
    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=r*exp(i*k),
        for |r|=1 they are propagating modes.
    """
    a, b = kwant.physics.leads.setup_linsys(h, t, tol, None).eigenproblem
    ev = kwant.physics.leads.unified_eigenproblem(a, b, tol=tol)[0]
    return ev


def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.
    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.
    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))


def get_rhos(syst, sum=False):
    p = np.kron(np.kron((np.eye(2) + sigma_z)//2, np.eye(2)), np.eye(2))
    h = np.kron(np.kron((np.eye(2) - sigma_z)//2, np.eye(2)), np.eye(2))
    sx = np.kron(np.kron(np.eye(2), sigma_x), np.eye(2))
    sy = np.kron(np.kron(np.eye(2), sigma_y), np.eye(2))
    sz = np.kron(np.kron(np.eye(2), sigma_z), np.eye(2))

    rohs = dict(
        all=kwant.operator.Density(syst, np.eye(8), sum=sum),
        p=kwant.operator.Density(syst, p, sum=sum),
        h=kwant.operator.Density(syst, h, sum=sum),

        sx=kwant.operator.Density(syst, sx, sum=sum),
        sy=kwant.operator.Density(syst, sy, sum=sum),
        sz=kwant.operator.Density(syst, sz, sum=sum),
    )

    return rohs

def get_gap(k,lead, params):
    Bands = bands(k=k, lead=lead, params=params, sort=False, num_bands=10)
    Bands = np.real(np.nan_to_num(Bands[0], nan = -1))
    
    bott_cond = min(Bands[Bands > 1e-12])
    top_val   = max(Bands[Bands <-1e-12])
    
    return (bott_cond - top_val)/2

def get_continuum_spectrum(h_mode, p_mode, ph_symmetry, params, k_range=(0,0.5), k_type="k_x", N=1000, **kwargs):

    params = param_list[p_mode].copy()
    
    for key in kwargs.keys():
        params[key] = kwargs[key]
    
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 
    
    if h_mode > 2:
        del params["conjugate"]
    
    f = systems.get_continuum_ham(ham_type, ph_symmetry)
    E = []

    if k_type=="k":
        E.append(eigvals(f(*k_range, **params)).real)
    else:
        ks = np.linspace(*k_range,N)
        for k in ks:
            if k_type=="k_x":
                h = f(k,0,0, **params)  
            if k_type=="k_y":
                h = f(0,k,0, **params)  
            if k_type=="k_z":
                h = f(0,0,k, **params) 
            E.append(eigvals(h).real)
    
#    ax.plot(ks,E,".", color=color, markersize=2, markeredgewidth=0);
    
    return np.array(E)

def get_discretized_spectrum(a, a_z, L, W, T, delta, m_z, mu, u_B, h_mode, p_mode, ph_symmetry, sym, num_bands, k, k_type="k_x", directions="x", **kwargs):
    
    params = param_list[p_mode].copy()
            
    params.update(default_params)
    
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 
    
    flux = 0
    
    for key in kwargs.keys():
        params[key] = kwargs[key]
    
    syst = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=ph_symmetry,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )
    
    syst = kwant.wraparound.wraparound(syst).finalized()
    
    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['a'] = a
    params['a_z'] = a_z       
    params['Delta_lead'] = delta_profile_funtion(T)
    params['U_B_lead'] = u_profile_funtion(T)    
    params['mu_ti'] = mu
    params['m_z'] = m_z
    params['delta'] = delta
    params['m0'] = 0
    params['m1'] = 0
    params['D'] = 0    
    params['S_imp'] = 0
    
    def h_k(**k_args):
        return syst.hamiltonian_submatrix(params=dict(**k_args, **params))
    
    which_position = np.where([d == k_type[2:] for d in directions])[0][0]
    keys = ["k_x", "k_y", "k_z"][:len(directions)]
    
    ks = dict()
    for key, val in zip(keys,np.zeros(len(directions))):
        ks[key] = val
    ks[keys[which_position]] = k
   
    Es = sla.eigs(h_k(**ks), k = num_bands, sigma = 0, return_eigenvectors=False)
    
    data = dict(
        Es=Es.real,
        mn=0
    )
    return data

def phase_diagram(a, a_z, T, W, delta, flux, m_z, mu, m0, m1, D, h_mode, ph_symmetry, p_mode):
    
    params = param_list[p_mode].copy()
    params.update(default_params)
    
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 
    
    flux = 0
    
    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=ph_symmetry,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )

    lead = lead.finalized()

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['a'] = a
    params['a_z'] = a_z       
    params['Delta_lead'] = delta_profile_funtion(T)
    params['U_B_lead'] = u_profile_funtion(T)    
    params['mu_ti'] = mu
    params['m_z'] = m_z
    params['delta'] = delta
    params['m0'] = m0
    params['m1'] = m1
    params['D'] = D      

    prop_modes, stab_modes = lead.modes(energy=delta/10, params=params)
    momenta = prop_modes.momenta
    wave_functions = prop_modes.wave_functions
    
    start = time.perf_counter()
        
    try:
        modes = prop_modes.block_nmodes[0]+prop_modes.block_nmodes[1]
    except:
        modes = prop_modes.block_nmodes[0]
        
    if ph_symmetry:
        rho         = kwant.operator.Density(lead,np.kron( sigma_0,np.kron( sigma_0, sigma_0)))
        rho_top_2D  = kwant.operator.Density(lead,np.kron( sigma_0,np.kron( sigma_0, (sigma_0+sigma_z)/2)))
        rho_sigma_z = kwant.operator.Density(lead,np.kron( sigma_0,np.kron( sigma_z, sigma_0)))
    else:
        rho         = kwant.operator.Density(lead,np.kron( sigma_0, sigma_0))
        rho_top_2D  = kwant.operator.Density(lead,np.kron( sigma_0, (sigma_0+sigma_z)/2))    
        rho_sigma_z = kwant.operator.Density(lead,np.kron( sigma_z, sigma_0))
    
    density_1 = -2
    density_2 = -2       

    if wave_functions.shape[1] == 2:
        pos1 = np.where(momenta==max(momenta))[0][0]
        density_1 = np.sum(rho_sigma_z(wave_functions[:,pos1]))/np.sum(rho(wave_functions[:,pos1]))
    if wave_functions.shape[1] == 4:
        pos2 = np.where(momenta==max(momenta[momenta<max(momenta)]))[0][0]
        density_2 = np.sum(rho_sigma_z(wave_functions[:,pos2]))/np.sum(rho(wave_functions[:,pos2]))

    density_e_left  = -2
    density_e_right = -2
    density_e_top   = -2
    density_e_bot   = -2
    density_h_left  = -2
    density_h_right = -2
    density_h_top   = -2
    density_h_bot   = -2
    
    if wave_functions.shape[1] == 2:

        wf_e = wave_functions[:,np.where(momenta==max(momenta))[0][0]]
        wf_h = wave_functions[:,np.where(momenta==min(momenta))[0][0]]

        density_e = rho(wf_e)
        density_h = rho(wf_h)

    if wave_functions.shape[1] == 2 and ham_type == '3D':

        density_e = density_e.reshape(int(W//a)+1,int(np.ceil(T/a_z))+1)
        density_h = density_h.reshape(int(W//a)+1,int(np.ceil(T/a_z))+1)

        density_e_left  = np.sum(density_e[0:3,:])/np.sum(density_e)
        density_e_right = np.sum(density_e[int(W//a)-2:int(W//a)+1,:])/np.sum(density_e)
        density_e_top   = np.sum(density_e[3:int(W//a)-2,int(T//(2*a_z)):])/np.sum(density_e)

        density_h_left  = np.sum(density_h[0:3,:])/np.sum(density_h)
        density_h_right = np.sum(density_h[int(W//a)-2:int(W//a)+1,:])/np.sum(density_h)
        density_h_top   = np.sum(density_h[3:int(W//a)-2,int(T//(2*a_z)):])/np.sum(density_h)
        
        density_e_bot= 1-density_e_left-density_e_right-density_e_top
        density_h_bot= 1-density_h_left-density_h_right-density_h_top
        
    if wave_functions.shape[1] == 2 and ham_type == '2D':

        density_e_top = rho_top_2D(wf_e)
        density_h_top = rho_top_2D(wf_h)

        density_e_left  = np.sum(density_e[0:3])/np.sum(density_e)
        density_e_right = np.sum(density_e[int(W//a)-2:int(W//a)+1])/np.sum(density_e)
        density_e_top   = np.sum(density_e_top[3:int(W//a)-2])/np.sum(density_e)

        density_h_left  = np.sum(density_h[0:3])/np.sum(density_h)
        density_h_right = np.sum(density_h[int(W//a)-2:int(W//a)+1])/np.sum(density_h)
        density_h_top   = np.sum(density_h_top[3:int(W//a)-2])/np.sum(density_h)
        
        density_e_bot= 1-density_e_left-density_e_right-density_e_top
        density_h_bot= 1-density_h_left-density_h_right-density_h_top
    
#    overlap_mat = np.kron(np.diag(np.ones(len(wave_functions[:,index_e])//4)), np.kron(np.kron(sigma_y, sigma_y), sigma_0))
#    overlap = np.dot(np.conj(wave_functions[:,index_e]),np.dot(overlap_mat,wave_functions[:,index_h]))

    data = dict(
        modes = modes,
        sigma_z_1=density_1,
        sigma_z_2=density_2,
        density_e_left =density_e_left,
        density_e_right=density_e_right,
        density_e_top  =density_e_top,
        density_e_bot  =density_e_bot,
        density_h_left =density_h_left,
        density_h_right=density_h_right,
        density_h_top  =density_h_top,
        density_h_bot  =density_h_bot,        
        time=time.perf_counter()-start
    )
    return data

def get_modes(a, a_z, T, W, mu, m_z, delta, h_mode, params):
    
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 
    
    flux = 0
    
    W = a*(W//a)
    
    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=False,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )

    lead = lead.finalized()
        
    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['mu_ti'] = mu        
    params['m_z'] = m_z
    params['delta'] = 0
        
    prop_modes, stab_modes = lead.modes(energy=1e-8, params=params)

    data = dict(
        modes=prop_modes.block_nmodes[0]
    )
    print(data)
    return data

def get_bands_topological(a, a_z, T, W, delta, flux, m_z, mu, m0, m1, D, h_mode, get_delta=False, p_mode=0):
    
    data_top = gap_zero_k(a, a_z, T, W, delta, flux, m_z, mu, m0, m1, D, h_mode, get_delta, p_mode)
    
    data_bands = calculate_lead_spectrum(a, a_z, T, W, k, 0, flux, m_z, mu, u_B, m0, m1, D, 0, 120, 0, h_mode)

def get_fig_of_merit(a, a_z, T, W, delta, m_z, mu, h_mode, u_B=0, get_delta=False, p_mode=0, k_points=1000, num_bands=50, sym=False, **kwargs):
    
    params = dict()
    if h_mode != 1:
        params = param_list[p_mode].copy()
       
    params.update(default_params)
        
    a, a_z, W, T, mu, m_z, params, h_mode, p_mode = optimize_parameters(a=a, a_z=a_z, W=W, T=T, mu=mu, m_z=m_z, u_B=u_B, params=params, sym=sym, h_mode=h_mode, p_mode=p_mode)
        
    flux = 0
    
    for key in kwargs.keys():
        params[key] = kwargs[key]    

    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode]            
        
    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=False,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )
    
    lead = lead.finalized()

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['a'] = a
    params['a_z'] = a_z       
    params['Delta_lead'] = delta_profile_funtion(T)
    params['U_B_lead']   = u_profile_funtion(T)    
    params['mu_ti'] = mu
    params['m_z'] = m_z
    params['u_B'] = u_B
    params['delta'] = delta
    params['S_imp'] = 0  
    mu = mu - u_B/2

    start = time.perf_counter()
    
    es_0  =[0]
    es_u , es_l  = [0], [0]
    es3_u, es3_l = [0], [0]
    
    E_gap = -1e-12 # so the learner traces the rest of the phase space with better resolution
    E_u,  E_l  = -1e-12, -1e-12
    E3_u, E3_l = -1e-12, -1e-12   
    
    mu_gap= -100    
    mu_u,  mu_l  = -100, -100
    mu3_u, mu3_l = -100, -100
    
    gap_u,  gap_l  = -1e-12, -1e-12
    gap3_u, gap3_l = -1e-12, -1e-12
    
    mn_u,  mn_l  = 0, 0
    mn3_u, mn3_l = 0, 0
     
    if h_mode == 1:
        k_max = 0.30 #here a_max = 10.5 \AA
    else:
        k_max = 0.075 #here a_max = 40 \AA

    if h_mode == 1:
        delta_E = 1 * k_max / k_points
    else:
        delta_E = 3 * k_max / k_points
    ks = np.linspace(0,k_max*a,k_points)
    Es = []

    for k in ks:
        data = calculate_lead_spectrum(
            a=a, a_z=a_z, T=T, W=W, k=k, delta=delta, flux=flux, m_z=m_z, mu=mu, u_B=u_B,
            ph_symmetry=False, num_bands=num_bands, m_eff=1, h_mode=h_mode, p_mode=p_mode, get_wf=False, sym = sym,
            **kwargs
        )
        
        Es.append(data['Es'])
        
#    return ks, Es

    Es = np.nan_to_num(Es,nan=-100)
    
    if h_mode != 1 and u_B == 0 :
        
        params_cont = params.copy()
        E_cont = get_continuum_spectrum(h_mode=h_mode, p_mode=p_mode, params=params_cont, ph_symmetry=False, k_type='k_x', k_range=[0,0.1], N=100, m_z=m_z)

        bottom_cond = np.min(np.max(E_cont,axis=1))
        top_valence = np.max(np.min(E_cont,axis=1))
    else:
        bottom_cond = 0.15
        top_valence = - 0.15

    max_E = np.min(  [bottom_cond, np.max(Es[0])]) 
    min_E = np.max(  [top_valence, np.min(Es[0][Es[0] > -100])] )
    
#    return min_E, max_E, top_valence, bottom_cond

    # Remove energies outside of the bulk gap
    E_stripped = []
    for i in np.arange(0,len(Es)):
        E_tmp = Es[i]
        E_tmp  = E_tmp [ (E_tmp > min_E) & (E_tmp < max_E) ]
        if len(E_tmp) > 0:
            E_stripped.append(E_tmp)
    
    # Determine required shift in order to have ordered bands
    deviations = [0]
    
    for i in np.arange(1,len(E_stripped)):
        deviation = []
        E_left  = E_stripped[i-1]
        E_right = E_stripped[i]

        min_length = min(len(E_left), len(E_right))
        position_shift = np.arange(-len(E_right) + 1,len(E_left))
        for shift in position_shift:
            if( shift < min(0,len(E_left)-len(E_right)) ):
                left_start = 0
                left_end   = min(len(E_right)+shift, min_length)

                right_start = max(-shift , 0)
                right_end   = len(E_right)
            elif( shift > max(0,len(E_left)-len(E_right))):
                left_start = max(shift , 0)
                left_end   = len(E_left)

                right_start = 0 
                right_end   = min(len(E_left)-shift, min_length)
            else:
                left_start = min(abs(shift),len(E_left)-min_length)
                left_end   = min(abs(shift)+min_length,len(E_left))

                right_start = min(abs(shift),len(E_right)-min_length) 
                right_end   = min(abs(shift)+min_length,len(E_right))

            deviation.append(np.sum(np.abs( E_left[left_start:left_end] - E_right[right_start:right_end] ))/(right_end-right_start) )
        deviations.append(deviations[-1] + position_shift[np.where(deviation == min(deviation))[0][0]] )
        
    max_len_E_diff = np.max([len(E) for E in E_stripped])-len(E_stripped[0])

    max_neg_deviation = np.max([-np.min(deviations), max_len_E_diff])
    max_pos_deviation = np.max([ np.max(deviations), max_len_E_diff])

    bands    = [[0] for i in np.arange(0,max_neg_deviation)] + [[E] for E in E_stripped[0]] + [[0] for i in np.arange(0,max_pos_deviation)]
    ks_bands = [[0] for i in np.arange(0,max_neg_deviation+max_pos_deviation+len(E_stripped[0]))]
    
    for i in np.arange(1,len(E_stripped)):
        [ bands [max_neg_deviation + deviations[i]+j].append(E_stripped[i][j]) for j in np.arange(len(E_stripped[i])) ]
        [ ks_bands[max_neg_deviation + deviations[i]+j].append(ks[i]) for j in np.arange(len(E_stripped[i])) ]
        
    # Determine first order derivative
    curvature = []
    for i in np.arange(len(bands)):
        if len(bands[i]) > 1:
            curvature.append( [(bands[i][j]-bands[i][j-2])/(ks_bands[i][j]-ks_bands[i][j-2]) 
                               for j in np.append(1,np.arange(1,len(bands[i]))) ] )
        else:
            curvature.append([0])
            
    dirac_points = []
    for i in np.arange(1,len(curvature)):
        if curvature[i-1][0] < 0 and curvature[i][0] > 0 and abs(bands[i-1][0] - bands[i][0]) < delta_E:
            dirac_points.append( (bands[i-1][0] + bands[i][0])/2 )
            
#    return dirac_points
            
    # extracts section with curvatures with different signs
    begin_sec = []
    end_sec = []
    for i in np.arange(len(bands)):
        band_sections = np.array_split(bands[i], np.where(np.array(curvature[i]) < 0)[0]) + np.array_split(bands[i]   ,np.where(np.array(curvature[i]) >= 0)[0])

        for sec in band_sections:
            if(len(sec) > 1) and np.min(sec) != 0 and np.max(sec) != 0:
                begin_sec.append(np.min(sec))
                end_sec.append(np.max(sec))                
                
    
    nu_energies = int((max_E-min_E)/1e-4)
    energies = np.linspace(min_E,max_E,nu_energies)
    
    nu_prop_modes = []
    for i in np.arange(nu_energies):
        nu_prop_modes.append( [begin_sec[j] < energies[i] and end_sec[j] > energies[i] for j in np.arange(len(begin_sec))].count(True) )
        
    nu_prop_modes = np.array(nu_prop_modes)

    # Find the maximum of prop states in the valence and conduction band

    split_pos = int(nu_energies/2)
    lower_prop_modes = nu_prop_modes[:split_pos]
    upper_prop_modes = nu_prop_modes[split_pos:]
    pos_max_valence  = np.min( np.where( lower_prop_modes == np.max(lower_prop_modes) )[0] )
    pos_max_conduc   = split_pos + np.max( np.where(upper_prop_modes == np.max(upper_prop_modes) )[0] )
    
    # Only consider the states between these two values
    energies      = energies[pos_max_valence:pos_max_conduc]
    nu_prop_modes = nu_prop_modes[pos_max_valence:pos_max_conduc]
    
    # Find the position of the dirac points
    dirac_where = []
    if len(dirac_points) > 0:
        dirac_where = np.where([ dp > energies[i-1] and dp < energies[i] for dp in dirac_points for i in np.arange(len(energies))])[0]%len(energies)
    
    #########################################
    # Find energies with no propagating modes    
    where_to_split_zero = np.where(nu_prop_modes != 0)[0]
    
    if len(dirac_where) > 0:
        where_to_split_zero = np.unique(np.append(where_to_split_zero, dirac_where))
    
    mode_energies   = np.array_split(energies     , where_to_split_zero)
    mode_bands = np.array_split(nu_prop_modes, where_to_split_zero)
        
    zero_mode_energies = []
    for i in np.arange(len(mode_bands)):
        if len(mode_bands[i][1:]) > 1:
            zero_mode_energies.append(mode_energies[i][1:])
    
    sizes_zero = [len(o) for o in zero_mode_energies]        
    
    #########################################
    # Find energies with one propagating modes
    where_to_split_one = np.where(nu_prop_modes != 1)[0]
    
    #If we have dirac point split the prop modes accordingly
    if len(dirac_where) > 0:
        where_to_split_one = np.unique(np.append(where_to_split_one, dirac_where))
    
    mode_energies  = np.array_split(energies     , where_to_split_one)
    mode_bands = np.array_split(nu_prop_modes, where_to_split_one)
    
    one_mode_energies = []
    for i in np.arange(len(mode_bands)):
        if len(mode_bands[i][1:]) > 1:
            one_mode_energies.append(mode_energies[i][1:])
    
    sizes_one = [len(o) for o in one_mode_energies]
    
    sorted_indexes_one = np.flip([x for _, x in sorted(zip(sizes_one, range(len(sizes_one))))])
            
    #########################################
    # Find energies with three propagating modes
    where_to_split_three = np.where(nu_prop_modes != 3)[0]
    
    #If we have dirac point split the prop modes accordingly
    if len(dirac_where) > 0:
        where_to_split_three = np.unique(np.append(where_to_split_three, dirac_where))
    
    mode_energies    = np.array_split(energies     , where_to_split_three)
    mode_bands = np.array_split(nu_prop_modes, where_to_split_three)
    
    three_mode_energies = []
    for i in np.arange(len(mode_bands)):
        if len(mode_bands[i][1:]) > 1:
            three_mode_energies.append(mode_energies[i][1:])
    
    sizes_three = [len(o) for o in three_mode_energies]
    
    sorted_indexes_three = np.flip([x for _, x in sorted(zip(sizes_three, range(len(sizes_three))))])
            
    #########################################
    
    if len(sizes_zero) > 0:
        es_0 = zero_mode_energies[ np.where([len(o)==np.sort(sizes_zero)[-1] for o in zero_mode_energies])[0][0] ]    

    if len(sizes_one) > 0:
        es1 = one_mode_energies[ sorted_indexes_one[0] ]
        if sym:
            es_u = es1
        else:
            es_l = es1

    if len(sizes_one) > 1:
        es2 = one_mode_energies[ sorted_indexes_one[1] ]
        if np.min(es1) < np.min(es2):
            es_l = es1
            es_u = es2
        else:
            es_l = es2
            es_u = es1
            
    if len(sizes_three) > 0:
        es31 = three_mode_energies[ sorted_indexes_three[0] ]
        if sym:
            es3_u = es31
        else:
            es3_l = es31

    if len(sizes_three) > 1:
        es32 = three_mode_energies[ sorted_indexes_three[1] ]
        if np.min(es31) < np.min(es32):
            es3_l = es31
            es3_u = es32
        else:
            es3_l = es32
            es3_u = es31            
            
    #########################################
    
    if len(es_0) > 1 and np.max(es_0) - np.min(es_0) > delta_E and np.max(es_0) < max_E-1e-8 and np.min(es_0) > min_E+1e-8:
        E_gap  =  np.max(es_0) - np.min(es_0)  
        mu_gap = (np.max(es_0) + np.min(es_0))/2

    if len(es_u) > 1 and np.max(es_u) - np.min(es_u) > delta_E:
        E_u  =  np.max(es_u) - np.min(es_u)
        mu_u = (np.max(es_u) + np.min(es_u))/2
        params_u = params.copy()
        params_u.update(mu_ti = mu_u + mu)        
        if get_delta:
            gap_u = gap_search_k(a=a, a_z=a_z, T=T, W=W, mu=params_u['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params_u)["gap"]
        if delta > 0:
            mn_u = calculate_majorana_num(a=a, a_z=a_z, T=T, W=W, mu=params_u['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params_u)["mn"]

    if len(es_l) > 1 and np.max(es_l) - np.min(es_l) > delta_E:
        E_l  =  np.max(es_l) - np.min(es_l)
        mu_l = (np.max(es_l) + np.min(es_l))/2
        params_l = params.copy()
        params_l.update(mu_ti = mu_l + mu)
        if get_delta:
            gap_l = gap_search_k(a=a, a_z=a_z, T=T, W=W, mu=params_l['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params_l)["gap"]
        if delta > 0:
            mn_l = calculate_majorana_num(a=a, a_z=a_z, T=T, W=W, mu=params_l['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params_l)["mn"]
            
    if len(es3_u) > 1 and np.max(es3_u) - np.min(es3_u) > delta_E and ( p_mode == 15 or h_mode == 1):
        E3_u  =  np.max(es3_u) - np.min(es3_u)
        mu3_u = (np.max(es3_u) + np.min(es3_u))/2
        params3_u = params.copy()
        params3_u.update(mu_ti = mu3_u + mu)        
        if get_delta:
            gap3_u = gap_search_k(a=a, a_z=a_z, T=T, W=W, mu=params3_u['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params3_u)["gap"]
        if delta > 0:
            mn3_u = calculate_majorana_num(a=a, a_z=a_z, T=T, W=W, mu=params3_u['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params3_u)["mn"]

    if len(es3_l) > 1 and np.max(es3_l) - np.min(es3_l) > delta_E and ( p_mode == 15 or h_mode == 1):
        E3_l  =  np.max(es3_l) - np.min(es3_l)
        mu3_l = (np.max(es3_l) + np.min(es3_l))/2
        params3_l = params.copy()
        params3_l.update(mu_ti = mu3_l + mu)
        if get_delta:
            gap3_l = gap_search_k(a=a, a_z=a_z, T=T, W=W, mu=params3_l['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params3_l)["gap"]
        if delta > 0:
            mn3_l = calculate_majorana_num(a=a, a_z=a_z, T=T, W=W, mu=params3_l['mu_ti'], m_z=m_z, delta=delta, h_mode=h_mode, params=params3_l)["mn"]            

    data = dict(
        ks = ks,
        Es = Es,
        ks_bands = ks_bands,
        bands    = bands,
        energies_prop_modes = energies,
        nu_prop_modes       = nu_prop_modes,
        E_gap = E_gap,
        E_u   = E_u,
        E_l   = E_l,
        E3_u  = E3_u,
        E3_l  = E3_l,
        mu_gap= mu_gap,        
        mu_u  = mu_u,
        mu_l  = mu_l,
        mu3_u = mu3_u,
        mu3_l = mu3_l,
        gap_u = gap_u,
        gap_l = gap_l,
        gap3_u= gap3_u,
        gap3_l= gap3_l,
        mn_u  = mn_u,
        mn_l  = mn_l, 
        mn3_u = mn3_u,
        mn3_l = mn3_l,
        time=time.perf_counter()-start
    )
    return data

def gap_search_k_fast(a, a_z, T, W, mu, m_z, delta, m0, m1, D, h_mode,p_mode):

    params = param_list[p_mode].copy()
    params.update(default_params)
    
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 
    
    flux = 0
    
    lead = systems.c_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=False,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    ) 
    
    lead = lead.finalized()

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['a'] = a
    params['a_z'] = a_z       
    params['Delta_lead'] = delta_profile_funtion(T)
    params['mu_ti'] = mu
    params['m_z'] = m_z
    params['delta'] = delta
    params['m0'] = m0
    params['m1'] = m1
    params['D'] = D      

    prop_modes, stab_modes = lead.modes(energy=0, params=params)
    momenta = prop_modes.momenta
    
    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=ph_symmetry,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )

    lead = lead.finalized()    

    start = time.perf_counter()

    gaps = []
    
    try:
        gaps.append( get_gap(0, lead, params) )
        for k in momenta[momenta > 0]:
            gaps.append( get_gap(k, lead, params) )
    except:
        print("Run error")
        gaps = [0]
    data = dict(
        gap=min(gaps),
        gaps=gaps,
        momenta=momenta,
        time=time.perf_counter()-start
    )
    return data

def gap_search_k(a, a_z, T, W, mu, m_z, delta, h_mode, params):
    
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 

    flux = 0

    W = a*(W//a)

    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=True,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )

    lead = lead.finalized()

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['mu_ti'] = mu        
    params['m_z'] = m_z
    params['delta'] = 0

    prop_modes, stab_modes = lead.modes(energy=0, params=params)
    momenta = prop_modes.momenta 

    params['delta'] = delta

    def E_k(k):
        Es, wfs = bands(k, lead, num_bands=4, sort=False, params=params)
        return np.min(np.abs(Es.real))

    def f(energy):
        if energy < 0:
            return 1
        else:
            return gap_minimizer(lead, params, energy) - 1e-13

    start = time.perf_counter()

    num_modes = momenta.shape[0]

    gaps = np.full(100, np.nan)
    res = list()
    momenta_cros = np.full(100, np.nan)
    momenta_gap = np.full(100, np.nan)

    E_k0, wfs_k0 = bands(
        k=0, lead=lead, num_bands=4, sort=False, params=params
    )
    rhos = get_rhos(lead, sum=True)

#    return E_k0, wfs_k0, rhos

    E_k0 = E_k0.real
    wfs_k0 = wfs_k0[:, E_k0 > 0]
    E_k0 = E_k0[E_k0 > 0]
    rhos_p_k0 = np.array([rhos['p'](wf) for wf in wfs_k0.T])
    rhos_h_k0 = np.array([rhos['h'](wf) for wf in wfs_k0.T])
    densitys_ph = rhos_p_k0 - rhos_h_k0
    fist_h_band_index = np.argmax(densitys_ph > 0)
    fist_h_band_E = E_k0[fist_h_band_index]

    
    momenta_gap[0] = 0
    gaps[0] = fist_h_band_E
    #gaps[0] = np.min(E_k0)

    if num_modes == 0:
        sol = optimize.root_scalar(
            f, x0=E_k0, bracket=[-1e-13, 0.3], method='bisect'
        )

        data = dict(
            gap=abs(sol.root),
            gaps=gaps,
            momenta_cros=momenta_cros,
            momenta_gap=momenta_gap,
            res=res,
            sol=sol,
            time=time.perf_counter()-start
        )
    else:
        for i, j in enumerate(np.arange(num_modes//4 + (num_modes//2) % 2)):
            k = momenta[j]
            delta_k = 1e-3
            #if -0.05 < k < 0:
            #    bounds = (k - delta_k, -0.005)
            #else:
            bounds = (k - delta_k, k + delta_k)
            _res = optimize.minimize_scalar(
                E_k, bounds=bounds, method='bounded',
            )
            momenta_cros[i] = k
            momenta_gap[i+1] = _res.x
            gaps[i+1] = _res.fun
            res.append(_res)

        data = dict(
            gap=min(gaps),
            gaps=gaps,
            momenta_cros=momenta_cros,
            momenta_gap=momenta_gap,
            res=res,
            time=time.perf_counter()-start
        )
    print(data)
    return data


def is_antisymmetric(H):
    return np.allclose(-H, H.T)


def make_skew_symmetric(ham):
    W = ham.shape[0] // 8
    I = np.eye(4, dtype=complex)
    U = np.bmat([[I, I], [-1j * I, 1j * I]])
    U = np.kron(np.eye(W, dtype=complex), U)

    skew_ham = U @ ham @ U.H

    assert is_antisymmetric(skew_ham)

    return skew_ham


def majorana_num(lead, params):
    h_k = get_h_k(lead, params)

    skew_h0 = make_skew_symmetric(h_k(0))
    skew_h_pi = make_skew_symmetric(h_k(np.pi))

    pf_0 = np.sign(cpf(1j * skew_h0, avoid_overflow=True).real)
    pf_pi = np.sign(cpf(1j * skew_h_pi, avoid_overflow=True).real)
    pfaf = pf_0 * pf_pi

    return pfaf

def calculate_majorana_num(a, a_z, T, W, mu, m_z, delta, h_mode, params):

    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 

    flux = 0

    W = a*(W//a)    

    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=True,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )
    
    lead = lead.finalized()

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['mu_ti'] = mu
    params['m_z'] = m_z
    params['delta'] = delta

    return dict(mn=majorana_num(lead, params))

def calculate_bulk_spectrum(
        a, a_z, T, W, k, delta, flux, m_z, mu, m0, m1, D,
        ph_symmetry, num_bands, m_eff, h_mode, p_mode, kwargs
):
    flux = 0
    params = param_list[p_mode].copy()
    params.update(default_params)

    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode] 

    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=ph_symmetry,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)"},
        ham_type=ham_type
    )
def calculate_lead_spectrum(
        a, a_z, T, W, k, delta, flux, m_z, mu, u_B,
        ph_symmetry, num_bands, m_eff, h_mode, p_mode, get_wf, sym, **kwargs
):
    flux = 0
    
    W = a*(W//a)    
    
    params = dict()
    if h_mode != 1:
        params = param_list[p_mode].copy()
    
#    a, a_z, W, T, mu, m_z, params, h_mode, p_mode = optimize_parameters(a, a_z, W, T, mu, m_z, u_B, params, sym, h_mode, p_mode)

    params.update(default_params)
    
    for key in kwargs.keys():
        params[key] = kwargs[key]
        
    ham_type = ['3D', '2D', 'metal','4band','8band'][h_mode]        
        
    lead = systems.make_lead(
        a=a, a_z=a_z, W=W, T=T,
        ph_symmetry=ph_symmetry,
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        subst={'Delta': "Delta_lead(delta, x, y, z)",
               'U_B':   "U_B_lead(u_B, x, y, z)"},
        ham_type=ham_type        
    )

    lead = lead.finalized()

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    params['a'] = a
    params['a_z'] = a_z   
    params['delta'] = delta
    params['Delta_lead'] = delta_profile_funtion(T)
    params['U_B_lead'] = u_profile_funtion(T)    
    params['mu_ti'] = mu
    params['m_z'] = m_z
    params['u_B'] = u_B     
    params["V"] = 0
    params['mu_m'] = mu
    params['C_m'] = 3.81/m_eff
#    params['m0'] = m0
#    params['m1'] = m1
#    params['D'] = D

    print(params, a, a_z, W, T)
    
    Es, wfs = bands(k, lead, params, num_bands)
    Es = Es.real

    if ph_symmetry:
        rhos = get_rhos(lead, sum=True)
        rhos_p = np.array([rhos['p'](wf) for wf in wfs.T]),
        rhos_h = np.array([rhos['h'](wf) for wf in wfs.T])
    else:
        rhos_p = None
        rhos_h = None
    
    mn = 0
    if ph_symmetry:
        mn = majorana_num(lead, params)
    try:
        gap = np.min([0.1, np.min(Es[Es > 0]) - np.max(Es[Es < 0]) ])
    except:
        gap = 0
    
    data = dict(
            Es=Es.real,
            mn=mn,
            a =a,
            a_z=a_z,
            gap = gap
            )
    if get_wf:
        data.update(wfs=wfs, rhos_p=rhos_p, rhos_h=rhos_h)
        
    return data



def tunnel_conductance_finite_ribbon(
        a, T, W, L, L_nodelta,  W_lead, T_lead, y_pos, z_pos, flux, E,
        m_eff, mu_m, mu_ti, Smag_imp, correlation_length, delta, m_z, n_av):
    params = default_params.copy()

    fsyst = systems.make_simple_tunnel_finite_junction(
        a, T, W, L, W_lead, T_lead, y_pos, z_pos)

    params['a'] = a
    params['Delta_metal'] = lambda delta, x, y, z: 0
    params['Delta_sc'] = delta_profile_funtion_fitite_ribbon(T, L, L_nodelta)
    params['delta'] = delta
    params['mu_ti'] = mu_ti
    params['mu_m'] = mu_m
    params['V'] = 0
    params['Smag_imp'] = Smag_imp
    params['C_m'] = 3.81/m_eff
    params['m_z'] = m_z

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    def _conductance(smatrix, lead_in, lead_out):
        if len(smatrix.data) == 0:
            c = np.nan
            N_e = np.nan
            r_ee = np.nan
            r_he = np.nan
        else:
            N_e = smatrix.submatrix((lead_in, 0), (lead_in, 0)).shape[0]
            r_ee = smatrix.transmission((lead_out, 0), (lead_in, 0))
            r_he = smatrix.transmission((lead_out, 1), (lead_in, 0))

            if lead_in == lead_out:
                c = N_e - r_ee + r_he
            else:
                c = r_ee - r_he

        return [c, N_e, r_ee, r_he]

    cs = []
    for i in range(n_av):
        if Smag_imp == 0:
            params["S_imp"] = lambda site, Smag_imp: 0
        else:
            params["S_imp"] = get_S_imp(i, a, fsyst, correlation_length)

        smatrix = kwant.smatrix(fsyst, energy=E, params=params)
        cs.append(_conductance(smatrix, 0, 0))

    cs = np.stack(cs)
    cs_mean = np.mean(cs, 0)
    return dict(c=cs_mean[0], cs_mean=cs_mean, cs_std=np.std(cs, 0), cs=cs)


def spectrum(syst, params, sigma=0, k=20, sort=True, solver='mumps'):

    if solver == 'numpy':
        ham = syst.hamiltonian_submatrix(params=params)
    else:
        ham = syst.hamiltonian_submatrix(params=params, sparse=True)

    solvers = {
        'mumps': partial(tools.mumps_eigsh, k=k, sigma=sigma),
        'scipy': partial(sla.eigsh, k=k, sigma=sigma),
        'numpy': np.linalg.eigh
    }

    energies, wfs = solvers[solver](ham)

    if sort:
        energies, wfs = sort_spectrum(energies, wfs)
        return fix_shift(energies, wfs)
    else:
        return energies, wfs


def finite_spectrum(
        a, T, W, L, L_nodelta,
        flux, delta, mu_ti, Smag_imp, m_z,
        correlation_length, n_av, k, sum=False):
    params = default_params.copy()

    fsyst = systems.make_ti_ribbon(a, L, W, T)

    params['a'] = a
    params['Delta'] = delta_profile_funtion_fitite_ribbon(T, L, L_nodelta)
    params['delta'] = delta
    params['mu_ti'] = mu_ti
    params['V'] = 0
    params['Smag_imp'] = Smag_imp
    params['m_z'] = m_z

    if T == 0 and flux == 0:
        params.update(dict(B_x=0, B_y=0, B_z=0))
    else:
        params.update(dict(B_x=flux/(W*T), B_y=0, B_z=0))

    rhos = get_rhos(fsyst, sum=sum)

    Es = []
    solver_time = []
    rhos_p_list = []
    rhos_h_list = []
    rhos_all_list = []

    for i in range(n_av):
        if Smag_imp == 0:
            params["S_imp"] = lambda site, Smag_imp: 0
        else:
            params["S_imp"] = get_S_imp(i, a, fsyst, correlation_length)
        solver_start = time.perf_counter()
        energies, wfs = spectrum(
            syst=fsyst, params=params, k=k, solver='mumps'
        )
        solver_time.append(time.perf_counter() - solver_start)
        rhos_p_list.append([rhos['p'](wf) for wf in wfs.T])
        rhos_h_list.append([rhos['h'](wf) for wf in wfs.T])
        rhos_all_list.append([rhos['all'](wf) for wf in wfs.T])
        Es.append(energies)

    data = dict(
        Es_mean=np.mean(Es, 0),
        rhos_p_mean=np.mean(rhos_p_list, 0),
        rhos_h_mean=np.mean(rhos_h_list, 0),
        rhos_all_mean=np.mean(rhos_all_list, 0),
        solver_time_mean=np.mean(solver_time)
    )
    return data