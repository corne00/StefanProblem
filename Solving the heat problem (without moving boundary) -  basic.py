#  Basic model for heat equation (without moving boundary)

### Import necessary modules
import numpy as np 
import time as time
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy.special import erf, erfc
from scipy.optimize import fsolve
from scipy.interpolate import lagrange
from scipy import linalg as linalg

### Plot settings
plot_boundary, plot_temperature, plot_errors = True, True, True

### Set simulation parameters
xmin, xmax = 0.0, 0.5 # set the domain
tmin, tmax = 0.001, 0.1 # set the inital and end time
M_array = np.array([10, 20, 40, 80, 160, 320, 640, 1280], dtype=np.longlong) # array with number of space steps for which you want to run the simulation
N_array = np.array([5000, 10000, 20000]) # array with number oftime steps for which you want to run the simulation
dt_array, dx_array = (tmax-tmin)/N_array, (xmax-xmin)/M_array # set array with size of time step and an array with size of the spatial step
errors = np.zeros((len(M_array), len(N_array))) # initalize 

#Problem parameters
s0 = 0.55
L= 1.0
Tl, Ts, Tm = 0.53, -0.1, 0.0
kl, ks = 1.0, 1.0

### Set functions
def Transcendental(lam):
    """This function is necesessary to find the lambda which is needed to 
    find the analytical solutions to the problem"""
    left = lam
    right = np.sqrt(ks)/(np.sqrt(np.pi)*L)*Ts/erfc(lam/np.sqrt(ks))*np.exp(-lam**2/ks)+np.sqrt(kl)/(np.sqrt(np.pi)*L)*Tl/(2-erfc(lam/np.sqrt(kl)))*np.exp(-lam**2/kl)
    return right-left
def T_analytical(lam,x,t):
    """"This function returns the temperature at a given array of positions
    at a given time.""" 
    s = s0 + 2*lam*np.sqrt(t) # find the position of the moving boundary at time t
    T_ana = np.ones_like(x)*Tm  # create an array with the value of the melting temperature
    T_ana = np.where(x<s,-Tl*erfc(lam/np.sqrt(kl))/(2-erfc(lam/np.sqrt(kl)))+Tl*erfc((x-s0)/(2*np.sqrt(kl*t)))/(2-erfc(lam/np.sqrt(kl))),T_ana) # compute the analytical solution at the left hand side of the boundary
    T_ana = np.where(x>s, Ts - Ts*erfc((x-s0)/(2*np.sqrt(ks*t)))/(erfc(lam/np.sqrt(ks))), T_ana) # compute the analytical solution at the right hand side of the boundary
    return T_ana
def s_analytical(lam, t):
    """This function returns the position of the moving boundary at given time"""
    s = s0 + 2*lam*np.sqrt(t)
    return s
def update_temp(Ti, t, lam, M, dx, dt):
    """This function updates the temperature profile using the values of the signed
    distance function and the previous temperature at each grid point. This function
    used the FVM-CN scheme for points away from the boundary. Close to the boundary,
    Lagrange interpolation polynomials are used to obtain an estimate for the value close
    to the boundaries using a BE scheme."""
    A = np.zeros((M,M)) # set the matrix (FVM-CN, homogeneous Neumann BCs)
    T0n = T_analytical(lam, xmin, t) # temperature at left boundary at time t_n
    T0n1 = T_analytical(lam, xmin, t+dt) # temperature at left boundary at time t_{n+1}
    TMn = T_analytical(lam, xmax, t) # temperature at right boundary at time t_n
    Tmn1 = T_analytical(lam, xmax, t+dt) # temperature at right boundary at time t_{n+1}
    for i in range(M):
        A[i,i] = 2+2*kl*dt/(dx**2)
        if i!=0: A[i,i-1] = -kl*dt/(dx**2)
        if i!=M-1: A[i,i+1] = -ks*dt/(dx**2)
    A[0,0] = 2+3*kl*dt/(dx**2)
    A[M-1,M-1] = 2+3*ks*dt/(dx**2)
    Rn = np.zeros(M)
    a1 = kl*dt/(dx**2)
    for i in range(1,M-1):
        Rn[i] = a1*Ti[i-1] + (2-2*a1)*Ti[i] + a1*Ti[i+1]
    Rn[0] = (2-3*a1)*Ti[0] + a1*Ti[1] +2*a1*T0n +2*a1*T0n1
    Rn[M-1] = a1*Ti[M-2] + (2-3*a1)*Ti[M-1] + 2*a1*TMn + 2*a1*Tmn1 
    Ti = np.linalg.solve(A,Rn) # Solve the matrix equation to find the new value of T
    return Ti

### Run the program for the given numbers of time- and spatial steps
lam = fsolve(Transcendental, 1.0)[0] # compute the lambda necessary to find the analytical solution

for Mcount, M in enumerate(M_array):
    for Ncount, N in enumerate(N_array):
        print("Simulation for M =", M, "and N =", N)
        start_time = time.time()
        dt, dx = dt_array[Ncount], dx_array[Mcount]
        print("dt/dx**2 =", dt/dx**2)
        xi = np.linspace(xmin+1/2*dx, xmax-1/2*dx, M)
        
        # Initialize Ti, t
        Ti = T_analytical(lam, xi, tmin)
        t = tmin
                        
        # Create saving arrays
        tarray = np.array([tmin]) # array with times
        Tnum = np.zeros((N+1, M)) # set array with numerical temperature
        Tnum[0] = Ti
        Tana = np.zeros((N+1, M)) # set array with analytical temperature
        Tana[0] = Ti
        
        for timestep in range(N):
            Ti = update_temp(Ti, t, lam, M, dx, dt)
            t= t+dt
            
            tarray = np.append(tarray, t)
            Tnum[timestep+1] = Ti
            Tana[timestep+1] = T_analytical(lam, xi, t)
            
        error_L2 = np.sqrt(1/len(Tnum[-1])) * linalg.norm(Tnum[-1]-Tana[-1])
        errors[Mcount, Ncount] = error_L2
        print("The error for this simulation is given by", error_L2)
        
        # plot the temperature    
        if plot_temperature is True:
            for timestep in np.array([0, N/16, N/8, N/4, N/2, N-1]).astype(int):
                plt.plot(xi, Tana[timestep], "C0", label="Analytical solution")
                plt.plot(xi, Tnum[timestep], "C1--", label="Numerical solution")
            plt.xlabel("x (m)")
            plt.ylabel("T (°C)")
            plt.legend(["Analytical solutions", "Numerical solutions"])
            plt.title("Temperature profiles for different timesteps (M =" + str(M) + ", dt =" + str(round(dt,4)) + ")")
            plt.show()
        print("This calculation took", round(time.time()-start_time, 2), "seconds \n")
 
if plot_errors is True:
    ### Plot the numerical error as a function of the temporal stepsize
    if len(dt_array)>2:
        for Mcount, M in enumerate(M_array):
            plt.loglog(N_array, errors[Mcount], "o--", label= str(M) + " spatial steps")
        a1 = errors[0][0]/(dt_array[0])
        b1 = errors[0][0]/(dt_array[0]**2)
        c1 = errors[0][0]/(dt_array[0]**3)
        plt.loglog(N_array, a1*dt_array, "-", label="Ideal 1st order")
        plt.loglog(N_array, b1*dt_array**2,"-", label="Ideal 2nd order")
        plt.loglog(N_array, c1*dt_array**3, "-", label="Ideal 3rd order")
        plt.xlabel(r"Number of time steps")
        plt.ylabel("Numerical L2-error")
        plt.legend(prop={'size': 10}, loc="best")
        plt.title("Numerical error of the FVM-CN method as a function of temporal stepsize")
        plt.show()
    
    ### Plot the numerical error as a function of the spatial stepsize
    if len(M_array)>2:
        for dtcount, dt in enumerate(dt_array):
            plt.loglog(M_array, errors[:,dtcount], "o--", label= str(N_array[dtcount]) + " temporal steps")
        a1 = np.transpose(errors)[0][0]/(1/M_array[0])
        a2 = np.transpose(errors)[0][0]/(1/M_array[0]**2)
        a3 = np.transpose(errors)[0][0]/(1/M_array[0]**3)
        plt.loglog(M_array, a1/M_array, "-", label="Ideal 1st order")
        plt.loglog(M_array, a2/M_array**2,"-", label="Ideal 2nd order")
        plt.loglog(M_array, a3/M_array**3, "-", label="Ideal 3rd order")
        plt.xlabel(r"Number of spatial steps")
        plt.ylabel("Numerical L2-error")
        plt.legend(prop={'size': 10}, loc="best")
        plt.title("Numerical error of the FVM-CN method as a function of spatial stepsize")
        plt.show()      




