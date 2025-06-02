# Standard library imports

# External package imports
import numpy as np
from numpy import diag, ones
from sympy import conjugate
from scipy.linalg import block_diag
import sympy
import kwant
from kwant.continuum.discretizer import discretize, momentum_operators

# Internal imports
import tools
import peierls

sigma_0 = np.identity(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

conservation_law = -np.kron(sigma_z, np.kron(sigma_0, sigma_0))

def get_sympy_hamiltonian(ham_type='3D', ph_symmetry=True, subst=dict()):
    if ham_type == '3D':
        if ph_symmetry:
            ham = (
                "-mu_ti  * kron(sigma_z, sigma_0, sigma_0) + "
                "-U      * kron(sigma_z, sigma_0, sigma_0) + "
                "epsilon * kron(sigma_z, sigma_0, sigma_0) + "
                "M * kron(sigma_z, sigma_0, sigma_z) - "
                "A_perp * k_x * kron(sigma_z, sigma_y, sigma_x) + "
                "A_perp * k_y * kron(sigma_0, sigma_x, sigma_x) + "
                "A_z * k_z * kron(sigma_z, sigma_0, sigma_y) + "
                "m_z * kron(sigma_z, sigma_z, sigma_0) +"
                "S_imp * kron(sigma_z, sigma_0, sigma_0)"
                "-re(Delta) * kron(sigma_y, sigma_y, sigma_0)"
                "-im(Delta) * kron(sigma_x, sigma_y, sigma_0)"
            )
        else:
            ham = (
                "-mu_ti  * kron(sigma_0, sigma_0) + "
                "-U      * kron(sigma_0, sigma_0) + "
                "epsilon * kron(sigma_0, sigma_0) + "
                "M * kron(sigma_0, sigma_z) - "
                "A_perp * k_x * kron(sigma_y, sigma_x) + "
                "A_perp * k_y * kron(sigma_x, sigma_x) + "
                "A_z * k_z * kron(sigma_0, sigma_y) + "
                "m_z * kron(sigma_z, sigma_0) +"
                "S_imp * kron(sigma_0, sigma_0)"
            )

        subst = dict(
            epsilon="(C_0 - C_perp * (k_x**2 + k_y**2) - C_z * k_z**2)",
            M="(M_0 - M_perp * (k_x**2 + k_y**2) - M_z * k_z**2)",
            **subst
        )
        ham = kwant.continuum.sympify(ham, locals=subst)
    elif ham_type == '2D':
        if ph_symmetry:
            ham = (
                "-mu_ti * kron(sigma_z, sigma_0, sigma_0) + "
                " S_imp * kron(sigma_z, sigma_0, sigma_0) + "
                "-u_B/2 * kron(sigma_z, sigma_0, sigma_0) + "
                " u_B/2 * kron(sigma_z, sigma_0, sigma_z) + "
                "-u_T/2 * kron(sigma_z, sigma_0, sigma_0) - "
                " u_T/2 * kron(sigma_z, sigma_0, sigma_z) + "
                "hbar * v_F * k_y * kron(sigma_0, sigma_x, sigma_z) - "
                "hbar * v_F * k_x * kron(sigma_z, sigma_y, sigma_z) + "
                "m * kron(sigma_z, sigma_0, sigma_x) + "
                "m_z * kron(sigma_z, sigma_z, sigma_0) - "
                "D * ( k_x**2 + k_y**2 ) * kron(sigma_z, sigma_0, sigma_0) + "               
                "delta/2 * kron(sigma_y, sigma_y, sigma_0) + "
                "delta/2 * kron(sigma_y, sigma_y, sigma_z) "
            )
        else:
            ham = (
                "-mu_ti * kron( sigma_0, sigma_0) + "
                " S_imp * kron( sigma_0, sigma_0) + "
                "-u_B/2 * kron( sigma_0, sigma_0) + "
                " u_B/2 * kron( sigma_0, sigma_z) + "
                "-u_T/2 * kron( sigma_0, sigma_0) - "
                " u_T/2 * kron( sigma_0, sigma_z) + "
                "hbar * v_F * k_y * kron( sigma_x, sigma_z) - "
                "hbar * v_F * k_x * kron( sigma_y, sigma_z) + "
                "m * kron( sigma_0, sigma_x) + "
                "m_z * kron( sigma_z, sigma_0) - "
                "D * ( k_x**2 + k_y**2 ) * kron( sigma_0, sigma_0)"
            )
            
        subst = dict(
             m = "(m0 - m1 * (k_x**2 + k_y**2))",
            **subst
        )
        ham = kwant.continuum.sympify(ham, locals=subst)
        
    elif ham_type == '2D_sides':
        if ph_symmetry:
            ham = (
                "- mu_ti * kron(sigma_z, sigma_0, sigma_0) + "
                "hbar * v_F * k_y * kron(sigma_0, sigma_x, sigma_z) - "
                "hbar * v_F * k_x * kron(sigma_z, sigma_y, sigma_z) + "
                "m_z * kron(sigma_z, sigma_z, sigma_0) - "
                "D * ( k_x**2 + k_y**2 ) * kron(sigma_z, sigma_0, sigma_0) + "               
                "delta/2 * kron(sigma_y, sigma_y, sigma_0) + "
                "delta/2 * kron(sigma_y, sigma_y, sigma_z) "
#                "-delta/2 * kron(sigma_y, sigma_y, sigma_0) + "
#                "delta/2 * kron(sigma_y, sigma_y, sigma_z)"
            )
        else:
            ham = (
                "- mu_ti * kron( sigma_0, sigma_0) + "
                "hbar * v_F * k_y * kron( sigma_x, sigma_z) - "
                "hbar * v_F * k_x * kron( sigma_y, sigma_z) + "
                "m_z * kron( sigma_z, sigma_0) - "
                "D * ( k_x**2 + k_y**2 ) * kron( sigma_0, sigma_0)"
            )
            
        ham = kwant.continuum.sympify(ham, locals=subst)        

    elif ham_type == 'metal':
        if ph_symmetry:
            ham = (
                "(-mu_m + V) * kron(sigma_z, sigma_0, sigma_0) + "
                "C_m * (k_x**2 + k_y**2 + k_z**2) * kron(sigma_z, sigma_0, sigma_0)"
                "-re(Delta) * kron(sigma_y, sigma_y, sigma_0)"
                "-im(Delta) * kron(sigma_x, sigma_y, sigma_0)"
            )
        else:
            ham = (
                "(-mu_m + V) * sigma_0 + "
                "C_m * (k_x**2 + k_y**2 + k_z**2) * sigma_0 +"
                "S_imp * sigma_0"
            )
        ham = kwant.continuum.sympify(ham, locals=subst)

    elif ham_type == '8band':
        k_x, k_y, k_z = sympy.symbols('k_x k_y k_z', real = True)
        P_1, P_2, P_3, Q_1, Q_2, Q_3, F_1, F_3, F_5, F_7, K_1, K_3, K_5, K_7, E_1, E_3, E_5, E_7, U_35, U_36, V_35, V_36, F_37, K_37, U_47, V_47, U_58, U_68, V_58, V_68, m_z, mu_ti = sympy.symbols('P_1 P_2 P_3 Q_1 Q_2 Q_3 F_1 F_3 F_5 F_7 K_1 K_3 K_5 K_7 E_1 E_3 E_5 E_7 U_35 U_36 V_35 V_36 F_37 K_37 U_47 V_47 U_58 U_68 V_58 V_68 m_z mu_ti')
        k_m = k_x - 1j*k_y
        k_p = k_x + 1j*k_y
        g_35 = U_35*k_z*k_p+V_35*k_m**2
        g_36 = conjugate(U_35)*k_z*k_p-conjugate(V_35)*k_m**2
        g_47 = U_47*k_z*k_p+V_47*k_m**2
        g_58 = U_58*k_z*k_p+V_58*k_m**2
        g_68 =-conjugate(U_58)*k_z*k_p+conjugate(V_58)*k_m**2
        f_37= F_37*k_z**2+K_37*(k_x**2+k_y**2)
        f_1 = F_1 *k_z**2+K_1 *(k_x**2+k_y**2)+E_1
        f_3 = F_3 *k_z**2+K_3 *(k_x**2+k_y**2)+E_3
        f_5 = F_5 *k_z**2+K_5 *(k_x**2+k_y**2)+E_5
        f_7 = F_7 *k_z**2+K_7 *(k_x**2+k_y**2)+E_7

        ham_upper = sympy.Matrix([
          [0, 0, k_z*Q_1,            k_m*P_1,            k_p*Q_2,           k_p*P_2,           k_z*Q_3,            k_m*P_3           ],
          [0, 0, k_p*conjugate(P_1),-k_z*conjugate(Q_1),-k_m*conjugate(P_2),k_m*conjugate(Q_2),k_p*conjugate(P_3),-k_z*conjugate(Q_3)], 
          [0, 0, 0,                  0,                  g_35,              g_36,              f_37,              -conjugate(g_47)   ], 
          [0, 0, 0,                  0,                  conjugate(g_36),  -conjugate(g_35),   g_47,               conjugate(f_37)   ], 
          [0, 0, 0,                  0,                  0,                 0,                -conjugate(g_68),    g_58              ], 
          [0, 0, 0,                  0,                  0,                 0,                 conjugate(g_58),    g_68              ], 
          [0, 0, 0,                  0,                  0,                 0,                 0,                  0                 ], 
          [0, 0, 0,                  0,                  0,                 0,                 0,                  0                 ]])
        h_diag = sympy.Matrix(block_diag(f_1*diag(ones(2)), f_3*diag(ones(2)), f_5*diag(ones(2)), f_7*diag(ones(2))))
        U = sympy.Matrix(block_diag(diag(ones(4)), np.array([[1, 1], [1, -1]])/np.sqrt(2), diag(ones(2))))
        
        ham = - mu_ti *diag(ones(8)) + U*(ham_upper + conjugate(ham_upper.T) + h_diag)*U + m_z *diag([1,-1,1,-1,3,-3,1,-1])

    elif ham_type == '4band':
        if ph_symmetry:
            ham = (
                "-mu_ti * kron(sigma_z, sigma_0, sigma_0) + "
                "-U   * kron(sigma_z, sigma_0, sigma_0) + "                
                "epsilon * kron(sigma_z, sigma_0, sigma_0) + "
                "M * kron(sigma_z, sigma_0, sigma_z) - "
                "A_0 * k_x * kron(sigma_z, sigma_y, sigma_x) + "
                "A_0 * k_y * kron(sigma_0, sigma_x, sigma_x) + "
                "B_0 * k_z * kron(sigma_z, sigma_0, sigma_y) + "
                "m_z * kron(sigma_z, sigma_z, sigma_0) +"
                "S_imp * kron(sigma_z, sigma_0, sigma_0)"
                "-re(Delta) * kron(sigma_y, sigma_y, sigma_0)"
                "-im(Delta) * kron(sigma_x, sigma_y, sigma_0)"
            )
        else:
            ham = (
                "-mu_ti * kron(sigma_0, sigma_0) + "
                "-U   * kron(sigma_0, sigma_0) + "                
                "epsilon * kron(sigma_0, sigma_0) + "
                "M * kron(sigma_0, sigma_z) - "
                "A_0 * k_x * kron(sigma_y, sigma_x) + "
                "A_0 * k_y * kron(sigma_x, sigma_x) + "
                "B_0 * k_z * kron(sigma_0, sigma_y) + "
                "m_z * kron(sigma_z, sigma_0) +"
                "S_imp * kron(sigma_0, sigma_0)"
            )
        subst = dict(
            epsilon="(C_0 + C_2 * (k_x**2 + k_y**2) + C_1 * k_z**2)",
            M="(M_0 + M_2 * (k_x**2 + k_y**2) + M_1 * k_z**2)",
            **subst
        )
        ham = kwant.continuum.sympify(ham, locals=subst)
        
    else:
        raise ValueError('ham_type \'{}\' not available.'.format(ham_type))
    return ham

def get_continuum_ham(ham_type='3D', ph_symmetry=False):
    
    H = get_sympy_hamiltonian(ham_type, ph_symmetry)
    k_x, k_y, k_z = sympy.symbols('k_x k_y k_z', real = True)
    
    if ham_type == '8band':
        P_1, P_2, P_3, Q_1, Q_2, Q_3, F_1, F_3, F_5, F_7, K_1, K_3, K_5, K_7, E_1, E_3, E_5, E_7, U_35, V_35, F_37, K_37, U_47, V_47, U_58, V_58, m_z, mu_ti = sympy.symbols('P_1 P_2 P_3 Q_1 Q_2 Q_3 F_1 F_3 F_5 F_7 K_1 K_3 K_5 K_7 E_1 E_3 E_5 E_7 U_35 V_35 F_37 K_37 U_47 V_47 U_58 V_58 m_z mu_ti')
        f = sympy.lambdify( (k_x, k_y, k_z, P_1, P_2, P_3, Q_1, Q_2, Q_3, F_1, F_3, F_5, F_7, K_1, K_3, K_5, K_7, E_1, E_3, E_5, E_7, U_35, V_35, F_37, K_37, U_47, V_47, U_58, V_58, m_z, mu_ti), H )
        
    elif ham_type == '4band':  
        A_0, B_0, C_0, C_1, C_2, M_0, M_1, M_2, R_1, R_2, m_z, S_imp, mu_ti, U = sympy.symbols('A_0 B_0 C_0 C_1 C_2 M_0 M_1 M_2 R_1 R_2 m_z S_imp mu_ti U', real = True)
        if ph_symmetry == True:
            re, im, Delta = sympy.symbols('re im Delta')
            f = sympy.lambdify( (k_x, k_y, k_z, A_0, B_0, C_0, C_1, C_2, M_0, M_1, M_2, R_1, R_2, m_z, S_imp, mu_ti, U, re, im, Delta), H )
        else:
            f = sympy.lambdify( (k_x, k_y, k_z, A_0, B_0, C_0, C_1, C_2, M_0, M_1, M_2, R_1, R_2, m_z, S_imp, mu_ti, U), H )        
        
    elif ham_type == '3D': 
        A_perp, A_z, M_0, M_perp, M_z, C_0, C_perp, C_z, m_z, S_imp, mu_ti, U = sympy.symbols('A_perp A_z M_0 M_perp M_z C_0 C_perp C_z m_z S_imp mu_ti U', real = True)
        if ph_symmetry == True:
            re, im, Delta = sympy.symbols('re im Delta')
            f = sympy.lambdify( (k_x, k_y, k_z, A_perp, A_z, M_0, M_perp, M_z, C_0, C_perp, C_z, m_z, S_imp, mu_ti, U, re, im, Delta), H )
        else:
            f = sympy.lambdify( (k_x, k_y, k_z, A_perp, A_z, M_0, M_perp, M_z, C_0, C_perp, C_z, m_z, S_imp, mu_ti, U), H )
    return f

@tools.memoize
def get_shape(L_start, W_start, L, W, T, v):
    L_start, W_start, T_start = (0, 0, 0)
    L_stop, W_stop, T_stop = (L, W, T)

    def _shape(site):
        (x, y, z) = site.pos - v
        is_in_shape = (
            (L_start <= x <= L_stop) and            
            (W_start <= y <= W_stop) and
            (T_start <= z <= T_stop)
        )
        return is_in_shape

    return _shape, v + np.array([L_start, W_start, T_start])


def get_shape_2D(L_start, W_start, L, W, v):
    L_start, W_start = (L_start, W_start)
    L_stop, W_stop = (L, W)

    def _shape(site):
        (x, y) = site.pos - v
        is_in_shape = (
            (W_start <= y <= W_stop) and
            (L_start <= x <= L_stop)
        )
        return is_in_shape

    return _shape, v + np.array([L_stop, W_stop])


def get_lead_shape_2D(W, T, y_pos, z_pos):
    v = np.array([0, y_pos, z_pos])
    W_start, T_start = (0, 0)
    W_stop, T_stop = (W, T)

    def _shape(site):
        (x, y, z) = site.pos - v
        is_in_shape = (
            (W_start <= y <= W_stop) and
            (T_start <= z <= T_stop)
        )
        return is_in_shape

    return _shape, v + np.array([0, W_stop, T_stop])


@tools.memoize
def get_template(a, a_z, vector_potential=None, ph_symmetry=True, **kwargs):
    kwargs['ph_symmetry'] = ph_symmetry
    ham = get_sympy_hamiltonian(**kwargs)

    norbs = ham.shape[0]
    if kwargs['ham_type'][:2] == '2D':
        grid = kwant.lattice.general(prim_vecs=[(a, 0),(0, a)],norbs=norbs)
    else:
        grid = kwant.lattice.general(prim_vecs=[(a, 0, 0),(0, a, 0),(0, 0, a_z)],norbs=norbs)
    
    tb_ham, coords = kwant.continuum.discretize_symbolic(ham)
    
    template = kwant.continuum.build_discretized(
            tb_ham, grid=grid, coords=coords)

    #template = discretize(ham, grid=grid)
    
    return template
#    return template, tb_ham


@tools.memoize
def make_lead(
        a, a_z, W, T, L=0, v=None, vector_potential=None,
        conservation_law=None, directions='x', **kwargs):
    
    if L == 0:
        L = a
    
    vectors_3D = dict(x=(L, 0, 0),y=(0, W, 0),z=(0, 0, T))
#    vectors_4band = dict(x=(3*L, 0, 0),y=(0, 3*W, 0),z=(0, 0, 3*T))
    vectors_2D = dict(x=(L, 0),y=(0, W))    
    
    if kwargs['ham_type'] == '2D':
        if not v:
            v = np.zeros(2)        
        shape_lead = get_shape_2D(L_start=0, W_start=0, L=L, W=W, v=v)
        vectors = vectors_2D
    elif kwargs['ham_type'] == '2D_sides':
        if not v:
            v = np.zeros(2)        
        shape_lead = get_shape_2D(L_start=0, W_start=0, L=L, W=W+(T-a), v=v)
        vectors = vectors_2D
    else:
        if not v:
            v = np.zeros(3)        
        shape_lead = get_shape(L_start=0, W_start=0, L=L, W=W, T=T, v=v)
        vectors = vectors_3D
        
    if len(directions) == 1:
        symmetry = kwant.TranslationalSymmetry(vectors[directions[0]])
    elif len(directions) == 2:
        symmetry = kwant.TranslationalSymmetry(vectors[directions[0]], vectors[directions[1]])
    elif len(directions) == 3:
        symmetry = kwant.TranslationalSymmetry(vectors[directions[0]], vectors[directions[1]], vectors[directions[2]])
    
    lead = kwant.Builder(symmetry, conservation_law=conservation_law)            
    template = get_template(a, a_z, vector_potential, **kwargs)
    if kwargs['ham_type'] == '2D_sides':
        lead.fill(template, *get_shape_2D(L_start=0, W_start=0, L=L, W=(T-a)/2-a, v=v))
        lead.fill(template, *get_shape_2D(L_start=0, W_start=W+(T-a)/2+a, L=L, W=W+(T-a), v=v))
        kwargs['ham_type'] = '2D'
        template = get_template(a, a_z, vector_potential, **kwargs)
        lead.fill(template, *get_shape_2D(L_start=0, W_start=(T-a)/2, L=L, W=W+(T-a)/2, v=v))
    else:
        lead.fill(template, *shape_lead)
        
    return lead

@tools.memoize
def make_ti_ribbon(a, a_z, L, W, T):
    template = get_template(
        a=a,
        a_z=a_z,
        subst={'S_imp': 'S_imp(site, Smag_imp)'},
        ham_type='2D',
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
        ph_symmetry=True
    )

    syst = kwant.Builder()
    shape = get_shape_2D(L_start=0, W_start=0, L=L, W=W, v=np.zeros(2))
    syst.fill(template, *shape)

    return syst.finalized()


@tools.memoize
def make_simple_tunnel_finite_junction(
        a, a_z, T, W, L, W_lead, T_lead, y_pos, z_pos
):
    syst = kwant.Builder()

    template_metal = get_template(
        a=a,
        a_z=a_z,
        subst={'S_imp': 0,
               'Delta': "Delta_metal(delta, x, y, z)"},
        ham_type='metal',
        vector_potential=None
    )

    template_sc = get_template(
        a=a,
        a_z=a_z,
        subst={'S_imp': 'S_imp(site, Smag_imp)',
               'Delta': "Delta_sc(delta, x, y, z)"},
        vector_potential="[0, - B_x * (z - {}), 0]".format(T),
    )

    shape = get_shape(L, W, T, (0, 0, 0))
    lead_shape = get_lead_shape(W_lead, T_lead, y_pos, z_pos)
    syst.fill(template_sc, *shape)

    symmetry0 = kwant.TranslationalSymmetry((a, 0, 0))
    lead0 = kwant.Builder(symmetry0, conservation_law=conservation_law)
    lead0.fill(template_metal, *lead_shape)
    syst.attach_lead(lead0)

    return syst.finalized()
