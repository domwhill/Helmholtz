'''
    Helmholtz solver -trial
    
    KSP algorithm GMres
    + ILU preconditioner
import numpy as np
import scipy.sparse.linalg as spla

A = np.array([[ 0.4445,  0.4444, -0.2222],
              [ 0.4444,  0.4445, -0.2222],
              [-0.2222, -0.2222,  0.1112]])

b = np.array([[ 0.6667], 
              [ 0.6667], 
              [-0.3332]])

M2 = spla.spilu(A)
M_x = lambda x: M2.solve(x)
M = spla.LinearOperator((3,3), M_x)

x = spla.gmres(A,b,M=M)

    Authour: Dominic Hill
      Email: dominicwhill@gmail.com  
'''
import sys
import os
import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import re
import matplotlib.ticker as ticker
import matplotlib as mpl
from scipy import interpolate as SI
from scipy import sparse as spa
import scipy.sparse.linalg as spla
import time
#-----------------------------------------------------------------------
q_e = 1.602e-19
m_e = 9.11e-31
m_p = 1.67e-27
k_b = 1.38e-23
epsilon0 = 8.854e-12
c = 3e8 # m/s
#---------------
#================
# chfoil_default5
v_te = 1.93E+07 # ms^-1
tau_ei = 7.44E-14 # s
nu_ei = 1.0/tau_ei # s^-1
lambda_mfp = 1.43675E-06 # m
omega_pe_over_nu_ei = 228.2351
atomic_Z = 6.51
atomic_A = 13.0
m_i = atomic_A*m_p

c_over_vte = 15.5177
c_norm = c_over_vte

no_lambda_L_per_dx = 6.0#2.39824871856
#================
n_name = 'ne.txt'
T_name = 'Te.txt'
Cx_name = 'Cx.txt'
grid_name = 'R.txt'
ref_info = np.loadtxt('chfoil_default5_norm.txt')
Te_ref,ne_ref,logLambda = ref_info[0],ref_info[1],ref_info[2]
lambda_mu = 0.5 # micrometers
lambda_norm = (lambda_mu*1e-6)/lambda_mfp
omega_0_norm = 2.0*np.pi*c_norm/(lambda_norm)
romega = omega_0_norm**-1
#---------------------
n_crit = 1.11e21/(lambda_mu**2) # cm^-3
n_crit_norm = n_crit/ne_ref
I0_TW_per_cm2 = 300 #vacuum intensity?
I0_W_per_m2 = I0_TW_per_cm2*(1e12)*(100.0**2)
#----------------------
# Flags ----
abs_on = True
dens_on = True
gauss_laser_on = True
y_factor = 1.0 # 1.0 = on, 0.0 = off
x_l,x_u = 2,-1
max_input = 100
#----------------------
#---------------------

n_data = np.loadtxt(n_name)
T_data = np.loadtxt(T_name)
C_data = np.loadtxt(Cx_name)
x_grid_cbg = np.loadtxt(grid_name)



n_data = n_data[::-1]
T_data = T_data[::-1]
C_data = C_data[::-1]

if max_input != -1:
    n_data = n_data[:max_input]
    T_data = T_data[:max_input]
    C_data = C_data[:max_input]
    x_grid_cbg = x_grid_cbg[:max_input+1]

#---------------------

xmin,xmax = x_grid_cbg[0],x_grid_cbg[-1] # lambda_mfp
ymin,ymax = 0.0,10.0 # lambda_mfp
Z = 6.51

# Compute Helmholtz grid
#
dx_h = ((lambda_mu*1e-6)/lambda_mfp)/no_lambda_L_per_dx # lambda_mfp
dt_h = 20.0*(dx_h/c_norm)# Helmholtz time step (collision times)
rdt = dt_h**-1
nx_h = int((xmax-xmin)/dx_h)
ny_h = int((ymax-ymin)/dx_h)

dx_h = (xmax-xmin)/float(nx_h)
dy_h =  (ymax-ymin)/float(ny_h)
tmax = dt_h*1000

print '\n-------------------------------------- time = ',time.clock()
print ' xmin = ', xmin, 'xmax = ', xmax
print ' ymin = ', ymin, 'ymax = ',ymax
print ' nx_h = ', nx_h
print ' ny_h = ', ny_h
print ' dx_h = ',dx_h,' [lambda_mfp]'
print ' dy_h = ',dy_h,' [lambda_mfp]'
print ' dt_h = ',dt_h,' [ t_col    ]'
print ' omega = ', omega_0_norm, '  romega = ', romega
print ' (c_norm/omega0)^2 = ',((c_norm*romega)**2)*(-2.0/(dx_h**2)) 
#--------------------------------------------

n_data2d,Y = np.meshgrid(n_data,np.ones(ny_h))
C_data2d,Y = np.meshgrid(C_data,np.ones(ny_h))
T_data2d,Y = np.meshgrid(T_data,np.ones(ny_h))

nx = len(n_data)

if dens_on:
    n_refractive_norm =  (1.0 - (n_data[0]/n_crit_norm))**0.5 # this is used to calculate E-field amplitude at incoming boundary
else:
    n_refractive_norm = 1.0
print ' n_refractive_in = ', n_refractive_norm
print ' n_data[0] = %3.4f, n_data[-1] = %3.4f n_crit_norm = %3.4f, 1.0 - n/ncrit = %3.4f' % (n_data[0],n_data[-1],n_crit_norm, 1.0 - (n_data[0]/n_crit_norm))
econst = (q_e/(m_e*lambda_mfp*(tau_ei**-2)))*((2.0/(c*n_refractive_norm*epsilon0))**0.5)
E_norm0 = econst*(I0_W_per_m2**0.5)
print ' E_norm0 = ', E_norm0
#----------------------
# Set boundary conditions:
# Currently only bcs available are: 
#   'laser' -  Laser source 
#   'refl' - reflectie bcs
x_bc_lh = 'laser'
x_bc_rh = 'refl'


#---------------------
x_grid_ccg = 0.5*(x_grid_cbg[1:] + x_grid_cbg[:-1])
y_grid_ccg = np.arange(ymin,ymax,dy_h) # this data is uniform in y direction

x_grid_h = np.arange(xmin,xmax,dx_h)
y_grid_h = np.arange(ymin,ymax,dy_h)
plot_scale = 1.0/(c_norm*dt_h)
lims = [x_grid_h[x_l]*plot_scale,x_grid_h[x_u]*plot_scale,y_grid_h[0]*plot_scale,y_grid_h[-1]*plot_scale]
print ' ymin = ', ymin, 'ymax = ', ymax
print ' x_grid_h[x_l] = ', x_grid_h[x_l],'x_grid_h[x_u] = ', x_grid_h[x_u]
print ' ny_h = ', ny_h, 'length of y_grid = ', len(y_grid_h)
print '\n--------------------------------------\n\n'

# Linearly interpolate IMPACT n_e  + T_e -> helmholtz grid
def interp_data(x_in,y_in,x_data_smooth):
    '''
        SI.PchipInterpolater
        y_data_smooth = interp_data(x_in,y_in,x_out)
    '''
    f = SI.PchipInterpolator(x_in,y_in,extrapolate=True)
    y_data_smooth = f(x_data_smooth) #linear

    return y_data_smooth
#-----------------------------------------------------------------------
def interp_data2D_abs(x_in,y_in,z_in,x_data_smooth,y_data_smooth):
    '''
        SI.PchipInterpolater
        z_out = interp_data2D_abs(x_in,y_in,z_in,x_data_smooth,y_data_smooth)
    '''
    X,Y = np.meshgrid(x_in,y_in)
    print ' np.shape(z_in) = ', np.shape(z_in),np.shape(X),np.shape(Y)
    f = SI.interp2d(X,Y,z_in,kind='linear')
    z_out = f(x_data_smooth,y_data_smooth)

    return z_out

# generate matrix

def get_indices_xy(i1d):
    '''
        ### takes the index of the vector x and tells you the x,y values
        iy,ix = get_indices_xy(i1d)
        converts matrix index to index
        ij = x,y
        00,10,20,...nx0,01,11,....nx1,....,nxny
        
    '''
    ix = (i1d % nx_h) # % = mod
    iy= (i1d/nx_h) % ny_h
    
    return iy,ix
#-----------------------------------------------------------------------

def get_indices_xy_vector(ix,iy):
    '''
        col,row = get_indices_xy_shift(ix,iy,ir,ip)
        ix + ir, iy + ip returns matrix index
        converts matrix index to index
        ij = x,y
        00,10,20,...nx0,01,11,....nx1,....,nxny
        i1d  = get_indices_xy_vector(ix,iy)
    '''
    x_index = (ix % nx_h)
    y_index = (iy % ny_h)
    i1d = (y_index*nx_h)+x_index
    # periodic bcs
    return i1d
#-----------------------------------------------------------------------

def get_indices_xy_shift(ix,iy,ir,ip):
    '''
        col,row = get_indices_xy_shift(ix,iy,ir,ip)
        ix + ir, iy + ip returns matrix index
        converts matrix index to index
        ij = x,y
        00,10,20,...nx0,01,11,....nx1,....,nxny
        
    '''
    if ix+ir>nx_h:
        x_index = nx_h-1
    else:
        x_index = ((ix+ir) % nx_h)
    
    y_index = (iy + ip) % ny_h
    row = (iy*nx_h)+ix
    # periodic bcs

    
    col = x_index + (y_index)*nx_h
    
    
    return col,row
#-----------------------------------------------------------------------

def get_nu_ei(ne,Te):
    '''
        nu_ei_norm = get_nu_ei(n_data,T_data)
    '''
    nu_ei_norm= ((2.0*Te)**1.5)/ne
    # probably should add a facotr of (3(sqrt(pi)/4))**-1 here to bring into line with Epperlein and Haines
    return nu_ei_norm
#-----------------------------------------------------------------------

def get_nu_norm(ne,Te):    
    '''
    nu_norm = get_nu_norm(n_data,T_data)
    '''
    nu_ei_norm = get_nu_ei(ne,Te)
    nu_ei_norm = nu_ei_norm*(ne)*(1.0/n_crit_norm)
    
    return nu_ei_norm
#-----------------------------------------------------------------------

def generate_laser():
    '''
        Generates a gaussian laser beam profile
        envelope= generate_laser()
    '''
    sigma,mu = (ymax-ymin)/5.0, 0.5*(ymax+ymin)
    envelope = np.zeros((len(y_grid_h)))
    if gauss_laser_on:
        envelope =  E_norm0*np.exp(-0.5*((y_grid_h-mu*np.ones(ny_h))/sigma)**2)
    else:
        envelope[ny_h/3:-ny_h/3] = E_norm0
    return envelope
#-----------------------------------------------------------------------
def generate_matrix(ne,Te):
    '''
        A.x =b
        this function generates A where x is E-field
        mat_sparse = 
        x bcs need adding i.e. oscillating laser field
    '''
    if abs_on:
        nu_norm = get_nu_norm(ne,Te)
    else:
        nu_norm = np.zeros((np.shape(ne)))
    #---------
    N_h = ne*(1.0/n_crit_norm)
    print ' np.shape(ne) = ', np.shape(ne),'np.shape Te = ', np.shape(Te)
    
    print '------- packing matrix ------time = ', time.clock()
    #nu_ei_h = get_nuei(n_data,T_data)#assuming nu_ei

    diag_indices = np.arange(ny_h*nx_h)
    diag_row,diag_col = diag_indices,diag_indices
    xp1_row = np.arange((ny_h*nx_h)-1)
    xp1_col = np.arange(1,ny_h*nx_h)
    xm1_row = np.arange(1,(ny_h*nx_h))
    xm1_col = np.arange((ny_h*nx_h)-1)

    ym1_col = np.arange(ny_h*nx_h)
    yp1_col = np.arange(ny_h*nx_h)
    # init
    ym1_row = np.zeros((ny_h*nx_h))
    yp1_row = np.zeros((ny_h*nx_h))
       
    ii = 0
    for iy in range(ny_h):
        for ix in range(nx_h):
            col,ym1_row[ii] = get_indices_xy_shift(ix,iy,0,-1)
            col,yp1_row[ii] = get_indices_xy_shift(ix,iy,0,1)
            ii+=1
            #ym1_row[ii] = diag_indices[ii] - nx_h
            #--- insert periodic bcs here
    
    #row_h,col_h = diag_indices,diag_indices
    
    #--------fill_diag
     
    coeff_laplacian_ij = ((c_norm*romega)**2)*((-2.0/(dx_h**2)) + (-2.0/(dy_h**2)*y_factor))
    coeff_diag = np.zeros((nx_h*ny_h),dtype=complex)
    
    for i1d in range(ny_h*nx_h):
        #for xi in range(nx_h):
        iy,ix = get_indices_xy(i1d)
        #print 'iy = ', iy, 'ix = ', ix, 'diff_y = ', (ny_h-1)-iy, 'diff_x = ix-nx_h = ', (nx_h-1)-ix
        coeff_diag[i1d] = complex(0.0,1.0)*(2.0/omega_0_norm)*(rdt - nu_norm[iy,ix]) + 2.0*(1.0-N_h[iy,ix]) + coeff_laplacian_ij
        if ix==0:
            coeff_diag[i1d] =  ((c_norm*romega)**2)*(-2.0/(dx_h**2))
        #indices.append((xi,yi))
    print '------- calculated diag --------- time = ', time.clock()
    #sys.exit()
    print ' np.shape(N_h = ', N_h[0,0], N_h[0,1]
    print 'coeff_diag 0,1 = ', coeff_diag[0], coeff_diag[1]
    #---------- BCS ----------
    # now over write top left and bottom right corners (first and last values in coeff_diag) with the x bcs
    # add in oscillating laser field? as coefficient in constant vector? < need to add in  i-1,j value of Laplacian somewhere
    # Look at rob's notes
    #-------------------------
    
    
    T = ((c_norm*romega)**2)*(1.0/(dx_h**2))
    J = ((c_norm*romega)**2)*(1.0/(dy_h**2))*y_factor
    
    I_diag = np.ones((ny_h*nx_h),dtype=complex)
    I_off_diag = np.ones(((ny_h*nx_h)-1),dtype=complex)
    coeff_laplacian_ip1j = T*I_off_diag
    
    # ---- bcs -------- remove all xp1 values at ix = nx
    for nnx in range(nx_h-1,nx_h*ny_h-1,nx_h):
        coeff_laplacian_ip1j[nnx] = 0.0
    
    
    coeff_laplacian_im1j = T*I_off_diag
    coeff_laplacian_ijp1 = J*I_diag
    coeff_laplacian_ijm1 = J*I_diag
    # bcs ====
    coeff_laplacian_ijp1[0] = 0.0
    coeff_laplacian_ijm1[0] = 0.0
    
    
    print '---- calculated diag --------time = ', time.clock()
    row = np.concatenate((diag_row,xm1_row,xp1_row,ym1_row,yp1_row))
    col = np.concatenate((diag_col,xm1_col,xp1_col,ym1_col,yp1_col))
    data = np.concatenate((coeff_diag,coeff_laplacian_im1j,coeff_laplacian_ip1j,coeff_laplacian_ijm1,coeff_laplacian_ijp1))
    
    
    print '\n\n\n---- COL SUMS-----------------------------#'
    print ' sum diag = ', np.sum(coeff_diag)
    print ' sum xm1 = ', np.sum(coeff_laplacian_im1j)
    print ' sum xp1 = ', np.sum(coeff_laplacian_ip1j)
    print ' sum ym1 = ', np.sum(coeff_laplacian_ijm1)
    print ' sum yp1 = ', np.sum(coeff_laplacian_ijp1)
    print '\n\n---- concated rows --------time = ', time.clock()
    mat_sparse = spa.csc_matrix((data,(row,col)),shape=(ny_h*nx_h,ny_h*nx_h),dtype=complex)

    print '---- packed matrix --------time = ', time.clock()
    print 'mat_sparse[1,:] = ', mat_sparse[0,:]
    print 'mat_sparse[2,:] = ', mat_sparse[1,:]
    print 'mat_sparse[3,:] = ', mat_sparse[2,:]
    print 'mat_sparse_jm1 = ', coeff_laplacian_ijm1[0], coeff_laplacian_ijp1[0]

    return mat_sparse,nu_norm


#-----------------------------------------------------------------------
def generate_const(psi_old,tstep):
    '''
         const = generate_const(psi_old)
    '''    
    
    print '---- calculating const --------time = ', time.clock()
    
    const = 2.0*complex(0.0,1.0)*romega*rdt*psi_old
    # bcs lh 
    if x_bc_lh == 'laser':
        
        envelope = generate_laser()
        print ' omega t = ', omega_0_norm*tstep
        
        psi_laser = envelope#*np.exp(1j*(-omega_0_norm*tstep*np.ones(ny_h)+Kx*0.0))
        
        bc_A = ((c_norm*romega)**2)*(1.0/(dx_h**2))*envelope
        for iy in range(ny_h):
            i1d  = get_indices_xy_vector(0,iy)
            const[i1d] = -bc_A[iy]
        for iy in range(ny_h):
            i1d = get_indices_xy_vector(0,iy)
            i1d2 = get_indices_xy_vector(nx_h-1,iy)
            

        
    
    return const
#-----------------------------------------------------------------------

def interp_to_helmholtz(data):
    '''
        wrapper to interpolate onto helmholtz grid
        data_out = interp_to_helmholtz(data)
    '''
    data_out = interp_data2D_abs(x_grid_ccg,y_grid_ccg,data,x_grid_h,y_grid_h)
    return data_out
#-----------------------------------------------------------------------

def interp_to_helmholtz_1d(data):
    '''
        wrapper to interpolate onto helmholtz grid
        data_out = interp_to_helmholtz(data)
    '''
    data_1d = interp_data(x_grid_ccg,data,x_grid_h)
    data2d,Y = np.meshgrid(data_1d,np.ones(ny_h))
    return data2d

#-----------------------------------------------------------------------
# init psi vector

tstep = 0.0
psi_old = np.zeros((nx_h*ny_h))


# Sparse CSR matrix:  A = spa.csr_matrix(data,(row,col)))

if dens_on:
    n_data_h = interp_to_helmholtz_1d(n_data)
    T_data_h = interp_to_helmholtz_1d(T_data)
else:
    n_data_h = interp_to_helmholtz_1d(n_data)
    T_data_h = interp_to_helmholtz_1d(T_data)
    n_data_h *=0.0
    T_data_h *= 0.0


print '---- interpolated data --------time = ', time.clock()

fig = plt.figure()
ax = plt.subplot2grid((4,1),(0,0),rowspan=2)
ax2 = plt.subplot2grid((4,1),(2,0))
ax3 = plt.subplot2grid((4,1),(3,0))
ax2.plot(x_grid_h,n_data_h[ny_h/2,:])


plt.ion()
cbar_on = True

while tstep<= tmax:
    print '---------------------'
    print ' time = ', tstep, ' number of timesteps = ', tstep/dt_h
    print '----- generating matrix ----- time = ',time.clock()
    mat,nu_norm = generate_matrix(n_data_h,T_data_h)
    #print ' SUM OF MATRIX = ', mat.sum()
    #print ' TRACE of MATRIX = ', mat.diagonal().sum()
    ##sys.exit()
    const = generate_const(psi_old,tstep)

    #print ' mat = ', mat
    print '----- initiating solve----- time = ', time.clock()


    M2 = spla.spilu(mat)
    M_x = lambda x: M2.solve(x)
    M = spla.LinearOperator((nx_h*ny_h,nx_h*ny_h), M_x)
    # Solve matrix
    # precondition + GMRES solve
    # GENERATE MATRIX NEEDS X-BCS <- oscillating laser field
    psi_new = spla.gmres(mat,const,M=M)
    #psi_new = spla.gmres(mat,const)
    
    print 'psi_new = ',psi_new , np.shape(psi_new[0]),' time = ', time.clock()
    print ' np.shape(psi_new) = ', np.shape(psi_new)
    tstep += dt_h
    diff_array = psi_new[0] - psi_old
    print 'difference = ', psi_new[0] - psi_old
    psi_old = psi_new[0]
    
    if True:#(tstep % dt_h*10)==0: # print out data every time steps
        
        psi_array = psi_old.reshape(ny_h,nx_h)
        plot_array = psi_array[:,x_l:x_u].real
        absorption = (nu_norm[:,x_l:x_u]*np.abs(psi_array[:,x_l:x_u]))**2
        #psi_array = diff_array.reshape(ny_h,nx_h)
        ax.cla()
        if cbar_on:
            vmin,vmax = np.min(plot_array)*0.4,np.max(plot_array)*0.4
            amin,amax = np.min(absorption),np.max(absorption)
        im = ax.imshow(plot_array,aspect='auto',vmin=vmin,vmax=vmax,extent=lims)
        imabs = ax3.imshow(absorption,aspect='auto',vmin=amin,vmax=amax,cmap='hot',extent=lims)
        
        ax.set_ylabel(r' y [ $c \Delta t$ ]')
        ax.set_xlabel(r' x [ $c \Delta t$]')
        if cbar_on:
            cbar = plt.colorbar(im,ax=ax)
            cbarabs = plt.colorbar(imabs,ax=ax3)
            cbar_on = False
        cbar.set_clim(vmin,vmax)
        cbarabs.set_clim(amin,amax)
        
        ax.set_title(' tstep = ' + str(tstep) + ' it_num = ' + str(tstep/dt_h))
        fig.savefig(str(tstep/dt_h) + '.png')
        plt.pause(0.05)
        
