#  Basic model with Lagrange scheme for advection equation

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
plot_boundary, plot_temperature, plot_errors, plot_vel = True, True, True, True

### Set simulation parameters
xmin, xmax = 0.0, 1.0 # set the domain
tmin, tmax = 0.001, 0.1 # set the inital and end time
M_array = np.array([200]) # array with number of space steps for which you want to run the simulation
N_array = np.array([200, 400, 800]) # array with number oftime steps for which you want to run the simulation
dt_array, dx_array = (tmax-tmin)/N_array, (xmax-xmin)/M_array # set array with size of time step and an array with size of the spatial step
errors = np.zeros((len(M_array), len(N_array))) # initalize 

### Problem parameters
s0 = 0.5   # initial position of the moving boundary
rho = 1.0  # density of liquid and solid
L = 1.0    # latent heat of the material
Tl, Ts, Tm  = 0.53, -0.1, 0.0 # temperarure of the liquid (>0), the solid (<0) and the melting point (0)
kl, ks  =  1.0, 1.0 # thermal conductivity of the liquid and the solid

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
def find_vel(Ti, xi, phi, dx):
    """This function estimates the velocity of the moving boundary by estimating
    the temperature gradients at both sides of the boundary and by using the Stefan condition"""
    Tgradsol = 0 # initial estimate for the gradient at the solid side
    Tgradliq = 0 # inital estimate for the gradient at the liquid side
    j = np.min(np.where(phi<0))-1 # find the index of the grid point in front of the moving boundary
    r1 = -(phi[j]/(phi[j+1]-phi[j])) # find the distance r1 between x_j and the moving interface (s - x_j = r1*dx)
    r2 = (phi[j+1]/(phi[j+1]-phi[j])) # find the distance r1 between x_{j+1} and the moving interface (x_{j+1} - s = r2*dx) 
    xs = xi[j] + r1*dx # find position of the moving boundary
    # Find a second-order Lagrange polynomial and take its derivative to find the gradient jump at the moving boundary.
    pc = 1/3
    if r1 <= pc:
        Tgradliq = np.polyder(lagrange([xi[j-2], xi[j-1], xs], [Ti[j-2], Ti[j-1], Tm]))(xs)
    elif r1 > pc:
        Tgradliq = np.polyder(lagrange([xi[j-1], xi[j], xs], [Ti[j-1], Ti[j], Tm]))(xs)
    if r2 > pc:
        Tgradsol = np.polyder(lagrange([xs, xi[j+1], xi[j+2]], [Tm, Ti[j+1], Ti[j+2]]))(xs)
    elif r2 <=pc:
        Tgradsol = np.polyder(lagrange([xs, xi[j+2], xi[j+3]], [Tm, Ti[j+2], Ti[j+3]]))(xs)  
    return (1/L*(ks*Tgradsol - kl*Tgradliq))
def solve_advection(vels, phi, M, dx, dt):
    """This function solves the advection equation for the given array of 
    constant velocities. It returns the updated advection function."""    
    if len(vels)==1:
        phi_new = phi + dt*vels[-1]
    elif len(vels)==2:
        phi_new = phi + dt/2*(3*vels[-1]-vels[-2])
    else:
        phi_new = phi + dt/2*(vels[-3]-3*vels[-2]+ 4*vels[-1])
    return phi_new 
def update_temp(phi, Ti, t, lam, M, dx, dt):
    """This function updates the temperature profile using the values of the signed
    distance function and the previous temperature at each grid point. This function
    used the FVM-CN scheme for points away from the boundary. Close to the boundary,
    Lagrange interpolation polynomials are used to obtain an estimate for the value close
    to the boundaries using a BE scheme."""
    j = np.min(np.where(phi<0))-1 # find the index of gridpoint in front of the moving boundary
    r1 = -(phi[j]/(phi[j+1]-phi[j])) # find the distance r1 between x_j and the moving interface (s - x_j = r1*dx)
    r2 = (phi[j+1]/(phi[j+1]-phi[j])) # find the distance r1 between x_{j+1} and the moving interface (x_{j+1} - s = r2*dx) 
    A = np.zeros((M,M)) # set the matrix (FVM-CN, homogeneous Neumann BCs)
    
    T0n = T_analytical(lam, xmin, t) # temperature at left boundary at time t_n
    T0n1 = T_analytical(lam, xmin, t+dt) # temperature at left boundary at time t_{n+1}
    TMn = T_analytical(lam, xmax, t) # temperature at right boundary at time t_n
    Tmn1 = T_analytical(lam, xmax, t+dt) # temperature at right boundary at time t_{n+1}

    for i in range(M):
        if i <j:
            A[i,i] = 2+2*kl*dt/(dx**2)
            A[i,i+1] = -kl*dt/(dx**2)
            if i!=0: A[i,i-1] = -kl*dt/(dx**2)
        elif i==j:
            A[i,i] = 1+kl*dt/(dx**2) * 2/r1
            A[i,i-1] = -kl*dt/(dx**2) * 2/(1+r1)
        elif i==j+1:
            A[i,i] = 1+ks*dt/(dx**2)* 2/r2
            A[i,i+1] = -ks*dt/(dx**2) * 2/(1+r2)
        elif i>j+1:
            A[i,i] = 2+2*ks*dt/(dx**2)
            A[i,i-1] = -ks*dt/(dx**2)
            if i!=M-1: A[i,i+1] = -ks*dt/(dx**2)
    A[0,0] = 2+3*kl*dt/(dx**2)
    A[M-1,M-1] = 2+3*ks*dt/(dx**2)
    Rn = np.zeros(M)
    a1 = kl*dt/(dx**2)
    a2 = ks*dt/(dx**2)
    for i in range(1,j):
        Rn[i] = a1*Ti[i-1] + (2-2*a1)*Ti[i] + a1*Ti[i+1]
    for i in range(j+1,M-1):
        Rn[i] = a2*Ti[i-1] + (2-2*a2)*Ti[i] + a2*Ti[i+1]
    Rn[0] = (2-3*a1)*Ti[0] + a1*Ti[1] +2*a1*T0n +2*a1*T0n1
    Rn[M-1] = a2*Ti[M-2] + (2-3*a2)*Ti[M-1] + 2*a2*TMn + 2*a2*Tmn1 
    Rn[j] = Ti[j]
    Rn[j+1] = Ti[j+1]
    Ti = np.linalg.solve(A,Rn) # Solve the matrix equation to find the new value of T
    return Ti
def pos_s(phi, xi):
    """This function returns the position of the moving interface, using the 
    values of the signed distance function"""
    j = np.min(np.where(phi<0))-1 # find the index of the grid point in front of the moving boundary
    r1 = -(phi[j]/(phi[j+1]-phi[j]))
    xs = xi[j] + r1*dx # find position of the moving boundary
    return xs

### Run the program for the given numbers of time- and spatial steps
lam = fsolve(Transcendental, 1.0)[0] # compute the lambda necessary to find the analytical solution

for Mcount, M in enumerate(M_array):
    for Ncount, N in enumerate(N_array):
        print("Simulation for M =", M, "and N =", N)
        start_time = time.time() # find the starting time
        dt, dx = dt_array[Ncount], dx_array[Mcount] # find dt and dx
        xi = np.linspace(xmin+1/2*dx,xmax-1/2*dx,M) # set a grid with xi at the center of each grid cell Ki
        print("dt/dx**2 = ", dt/dx**2)
        # Initalize t, phi, Ti
        s = s_analytical(lam, tmin)
        t = tmin
        phi = s - xi
        Ti = T_analytical(lam, xi, tmin)
        
        # Create arrays in which the solution is stored
        tarray = np.array([tmin]) # array with times
        snum = np.array([s]) # array with numerical boundary position
        sana = np.array([s]) # array with analytical boundary positions
        Tnum = np.zeros((N+1, M)) # set array with numerical temperature
        Tnum[0] = Ti
        Tana = np.zeros((N+1, M)) # set array with analytical temperature
        Tana[0] = Ti
        vels = np.array([])
        r1_array = np.array([])
        
        for timestep in range(N):
            v = find_vel(Ti, xi, phi, dx)               ### Step 1: find the front velocity
            j = np.min(np.where(phi<0))-1 
            r1_array = np.append(r1_array, -(phi[j]/(phi[j+1]-phi[j])))
            vels = np.append(vels, v)
            phi = solve_advection(vels, phi, M, dx, dt)    ### Step 2: solve the advection equation
            Ti = update_temp(phi, Ti, t, lam, M, dx, dt)### Step 3: solve the heat equation
            t = t+dt                                    ### Step 4: update counter
            
            # Check if CFL-condition on advection equation is satisfied
            CFL = v*dt/dx
            if CFL>1/2: print("CFL is too big. It is given by: ", CFL, "at t = ", t)
            
            ### Solve solutions
            xs = pos_s(phi, xi)
            snum = np.append(snum, xs)
            sana = np.append(sana, s_analytical(lam, t))
            tarray = np.append(tarray, t)
            Tnum[timestep+1] = Ti
            Tana[timestep+1] = T_analytical(lam, xi, t)
        
        # Compute the errors
        error_L2 = np.sqrt(1/len(snum))* linalg.norm(snum-sana, ord=2)
        errors[Mcount, Ncount] = error_L2
        print("The error for this simulation is given by", error_L2)
        
        # Plot the solution for the moving boundary
        if plot_boundary is True:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.plot(tarray, sana, label="Analytical solution")
            plt.plot(tarray, snum, label="Numerical solution")
            plt.xlabel("t (s)")
            plt.ylabel("s (m)")
            plt.legend()
            plt.title("Position of the moving boundary (M =" + str(M) + ", N =" + str(N) + ")")
            plt.show() 
        # plot the temperature    
        if plot_temperature is True:
            for timestep in np.array([0, N/16, N/8, N/4, N/2, N-1]).astype(int):
                plt.plot(xi, Tana[timestep], "C0", label="Analytical solution")
                plt.plot(xi, Tnum[timestep], "C1--", label="Numerical solution")
            plt.xlabel("x (m)")
            plt.ylabel("T (°C)")
            plt.legend(["Analytical solutions", "Numerical solutions"])
            plt.title("Temperature profiles for different timesteps (M =" + str(M) + ", N =" + str(N) + ")")
            plt.show()
        print("This calculation took", round(time.time()-start_time, 2), "seconds \n")
        
        if plot_vel is True:
            plt.plot(tarray[0:int(N/20)], lam/np.sqrt(tarray[0:int(N/20)]), "C0", label = "Analytical solution" )
            plt.plot(tarray[0:int(N/20)], vels[0:int(N/20)], "C1--", label="Numerical solution")
            plt.plot(tarray[0:int(N/20)], r1_array[0:int(N/20)]+4, "C2", label = "r₁ + 4")
            plt.ylabel("v(t) (m/s)")
            plt.xlabel("Time (s)")
            plt.legend()
            #plt.title("Numerical versus analytical velocity (M = " + str(M) + ", N = " + str(N) + ")")
            plt.show()
    
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
            plt.loglog(M_array, errors[:,dtcount], "o--", label= str(N_array[dtcount]) + " temporal step")
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

            