import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as linalg
import time



#Model parameters
v = 1
M_array = np.array([200])#16, 32,64,128, 256, 512])
dx1 = min(1/M_array)
dt_array = np.array([dx1/32, dx1/16, dx1/8, dx1/4, dx1/2, dx1])
t0 = 0
t_end = 0.5

UpwindMethod = True

#Set error matrices
errors_L2_UM = np.zeros((len(M_array), len(dt_array)))

#Define the analytical solution to the problem
def analytical_solution(x,v,t):
    return np.exp(-1000*(x-v*t-0.2)**2)    
    return (x-v*t)

# Define the upwind method
def Upwind_method(u,v):
    A = np.zeros((M+1,M+1))
    amax = max(v,0)*dt/dx
    amin = min(v,0)*dt/dx
    for i in range(M+1):
        A[i,i] = 1 + amin - amax
        if i!=0: A[i,i-1] = amax
        if i!=M: A[i,i+1] = -amin
    A[M,0] = -amin
    A[0,M] = amax
    return np.matmul(A,u)

#set counters for M and dt
Mcount = 0
dtcount = 0

for M in M_array:
    for dt in dt_array:
        # Define the grid and the initial solution
        xi = np.linspace(0,1,M+1)
        dx=1/M
        u0 = analytical_solution(xi,v,0)
        # Run the upwind method and calculate errors
        if UpwindMethod is True:
            start_time = time.time()
            u = u0
            t = t0
            
            b = int(t_end/dt)
            plot_array = [0, int(b/16), int(b/8), int(b/4), int(b/2), int(b-1)]
            bcount = 0
            
            while t<t_end:
                u = Upwind_method(u,v)
                
                if bcount in plot_array: 
                    plt.plot(xi, u)
                    plt.plot(xi, analytical_solution(xi,v,t))
                t = t+dt
                bcount = bcount+1
            
                
            plt.show()
            t_calc = t-dt
            
            error_L2 = np.sqrt(dx)*linalg.norm((u-analytical_solution(xi,v,t_calc)),2)
            errors_L2_UM[Mcount, dtcount] = error_L2
            print("The Upwind Method took ", time.time()-start_time, "seconds to run.")
            
            bcount = 0
        dtcount = dtcount+1
    Mcount = Mcount + 1
    dtcount =0
    
plt.plot(1/M_array, errors_L2_UM)
plt.xlabel(r"$\Delta x$")
plt.show()

plt.loglog(1/M_array, errors_L2_UM)
plt.xlabel(r"$\Delta x$")
plt.show()

plt.plot(dt_array, np.transpose(errors_L2_UM))
plt.xlabel(r"$\Delta t$")
plt.show()            
    
plt.loglog(dt_array, np.transpose(errors_L2_UM))
plt.xlabel(r"$\Delta t$")
plt.show()   