import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as linalg
import time



#Model parameters
v = 1
l=10
M_array = np.array([16,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])#16, 32,64,128, 256, 512])
dx1 = min(l/M_array)
dt_array = np.array([1])
t0 = 0
t_end = 5

UpwindMethod = True

#Set error matrices
errors_L2_UM = np.zeros((len(M_array), len(dt_array)))

#Define the analytical solution to the problem
def analytical_solution(x,v,t):
    #return np.exp(-1000*(x-v*t-0.2)**2)    
    return np.exp(-(x-v*t-2)*(x-v*t-2))

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
        xi = np.linspace(0,l,M+1)
        dx=l/M
        dt=dx
        u0 = analytical_solution(xi,v,0)
        # Run the upwind method and calculate errors
        if UpwindMethod is True:
            start_time = time.time()
            u = u0
            t = t0
            
            b = int(t_end/dt)
            plot_array = [0, 1*int(b/6), 2*int(b/6), 3*int(b/6), 4*int(b/6), 5*int(b/6), int(b)-1]
            bcount = 0
            
            while t<t_end:
                u = Upwind_method(u,v)
                
                if bcount in plot_array: 
                    plt.plot(xi, analytical_solution(xi,v,t), color="green")
                    plt.plot(xi, u)
                t = t+dt
                bcount = bcount+1
            
            plt.xlabel("x")
            plt.ylabel("u(x,t)")
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

plt.loglog(M_array, errors_L2_UM, label="1st order Upwind Method")
plt.xlabel(r"$\Delta x$")
a1 = errors_L2_UM[0][0]/(1/M_array[0]**2)
b1 = errors_L2_UM[0][0]/(1/M_array[0])
c1 = errors_L2_UM[0][0]/(1/M_array[0]**3)
plt.loglog(M_array, b1*1/M_array, "-", base=2, label="Ideal 1st order")
plt.loglog(M_array, a1*1/M_array**2, "-", base=2, label="Ideal 2nd order")
plt.loglog(M_array, c1*1/M_array**3, "-", base=2, label="Ideal 3rd order")
plt.legend()
plt.show()

plt.plot(dt_array, np.transpose(errors_L2_UM))
plt.xlabel(r"$\Delta t$")
plt.show()            
    
plt.loglog(dt_array, np.transpose(errors_L2_UM))
plt.xlabel(r"$\Delta t$")
a1 = np.transpose(errors_L2_UM)[-1][-1]/(dt_array[-1]**2)
b1 = np.transpose(errors_L2_UM)[-1][-1]/(dt_array[-1])
c1 = np.transpose(errors_L2_UM)[-1][-1]/(dt_array[-1]**3)
# plt.loglog(t_end/dt_array, b1*dt_array, "-", base=2, label="Ideal 1st order")
# plt.loglog(t_end/dt_array, a1*dt_array**2, "-", base=2, label="Ideal 2nd order")
# plt.loglog(t_end/dt_array, c1*dt_array**3, "-", base=2, label="Ideal 3rd order")
plt.show()   