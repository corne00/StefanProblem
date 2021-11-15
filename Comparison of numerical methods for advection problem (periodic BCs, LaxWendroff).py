import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as linalg
import time



#Model parameters
v = 1
l=10
M_array = np.array([32,64,128,256,512])#10, 20, 40, 50, 100, 200, 400])#16, 32,64,128, 256, 512])
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

# Define the LaxWendroff method
def LaxWendroff(u,v):
    A = np.zeros((M+1,M+1))
    a = v*dt/dx
    for i in range(M+1):
        A[i,i] = 1 - a**2
        if i!=0: A[i,i-1] = a/2*(a+1)
        if i!=M: A[i,i+1] = a/2*(a-1)
    A[M,0] = a/2*(a-1)
    A[0,M] = a/2*(a+1)
    return np.matmul(A,u)

#set counters for M and dt
Mcount = 0
dtcount = 0

for M in M_array:
    for dt in dt_array:
        # Define the grid and the initial solution
        xi = np.linspace(0,l,M+1)
        dx=l/M
        #dt=dx**2.6
        dt=dx/2
        #dt=dx
        #dt=dx/100
        #dt =dx/10
        #dt=dx**2/30
        u0 = analytical_solution(xi,v,0)
        print("M =", M, ", dt= ", dt)
        # Run the upwind method and calculate errors
        if UpwindMethod is True:
            start_time = time.time()
            u = u0
            t = t0
            
            b = int(t_end/dt)
            plot_array = [0, 1*int(b/6), 2*int(b/6), 3*int(b/6), 4*int(b/6), 5*int(b/6), int(b)-1]
            bcount = 0
            
            while t<t_end:
                u = LaxWendroff(u,v)
                t=t+dt
                if bcount in plot_array: 
                    plt.plot(xi, analytical_solution(xi,v,t), color="green")
                    plt.plot(xi, u)
                #t = t+dt
                bcount = bcount+1
            
                
            plt.show()
            t_calc = t-dt
            
            error_L2 = np.sqrt(dx)*linalg.norm((u-analytical_solution(xi,v,t_calc)),2)
            errors_L2_UM[Mcount, dtcount] = error_L2
            print("The Lax Wendroff Method took ", time.time()-start_time, "seconds to run.")
            
            bcount = 0
        dtcount = dtcount+1
    Mcount = Mcount + 1
    dtcount =0
    
plt.plot(1/M_array, errors_L2_UM)
plt.xlabel(r"$\Delta x$")
plt.show()

plt.loglog(M_array, errors_L2_UM, label="Lax-Wendroff")
plt.xlabel(r"Number of elements")
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
    
plt.loglog(t_end/dt_array, np.transpose(errors_L2_UM))
a1 = np.transpose(errors_L2_UM)[0][0]/(dt_array[0]**2)
b1 = np.transpose(errors_L2_UM)[0][0]/(dt_array[0])
c1 = np.transpose(errors_L2_UM)[0][0]/(dt_array[0]**3)
plt.loglog(t_end/dt_array, b1*dt_array, "-", base=2, label="Ideal 1st order")
plt.loglog(t_end/dt_array, a1*dt_array**2, "-", base=2, label="Ideal 2nd order")
plt.loglog(t_end/dt_array, c1*dt_array**3, "-", base=2, label="Ideal 3rd order")
plt.xlabel(r"Number of timesteps")
plt.show()   