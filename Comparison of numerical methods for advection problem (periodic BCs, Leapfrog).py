import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as linalg
import time

plot_errors = True

#Model parameters
v = 1
l=10
t0 = 0
t_end = 5
M_array = np.array([10, 20, 40, 80, 160, 320, 640])#10, 20, 40, 50, 100, 200, 400])#16, 32,64,128, 256, 512])
dx1 = min(l/M_array)
N_array = np.array([2,4,8,16,32,64])
dt_array = (t_end-t0)/N_array


LF = True

#Set error matrices
errors = np.zeros((len(M_array), len(dt_array)))

#Set function array
u_array = np.array([])

def LeapFrog(u_array, bcount):
    A = np.zeros((M,M))
    B = np.zeros((M,M))
    a = v*dt/dx
    if bcount == 0:
        for i in range(M):
            A[i,i] = 1
            if i!=0: A[i,i-1] = a
            if i!=M-1: A[i,i+1] = -a
    if bcount != 0:
        for i in range(M):
            A[i,i] = 0
            if i!=0: A[i,i-1] = a
            if i!=M-1: A[i,i+1] = -a
        for i in range(M):
            B[i,i] = 1
    A[0,M-1] = a
    A[M-1,0] = -a
    if bcount == 0:
        u_array[1] = u_array[0]
    if bcount !=0: 
        u_array[bcount+1] = np.matmul(B,u_array[bcount-1])+ np.matmul(A, u_array[bcount])
    return u_array
            
Mcount=0
dtcount=0
        

#Define the analytical solution to the problem
def analytical_solution(x,v,t):    
    return np.exp(-(x-v*t-2)*(x-v*t-2))


for M in M_array:
    for dtc, dt in enumerate(dt_array):
        print("M = ", M, ", dt = ", dt)
        dx = l/M
        if N_array[dtcount]==2: dt = dx/2
        if N_array[dtcount]==4: dt = dx/4
        if N_array[dtcount]==8:dt = dx/8
        if N_array[dtcount]==16:dt = dx/16
        if N_array[dtcount]==32:dt = dx/32
        if N_array[dtcount]==64:dt = dx/64

        #dt=1/100*dx
        xi = np.linspace(0+1/2*dx, l-1/2*dx, M)
        u0 = analytical_solution(xi, v, 0)
        u_array = np.zeros((int(t_end/dt)+2, M))
        u_array[0] = u0
        
        if LF is True:
            start_time = time.time()
            b = int(t_end/dt)
            plot_array = [0, 1*int(b/6), 2*int(b/6), 3*int(b/6), 4*int(b/6), 5*int(b/6), int(b)-1]
            bcount = 0 
            t = t0
            print("a =", v*dt/dx)
            while t < t_end:
                u_array = LeapFrog(u_array, bcount)
                if bcount in plot_array:
                    plt.plot(xi, analytical_solution(xi,v,t), color="green")
                    plt.plot(xi, u_array[bcount])
                bcount = bcount+1
                t = t+dt
        t = t-dt
        error_L2 = np.sqrt(dx) * linalg.norm(u_array[bcount-1] - analytical_solution(xi,v,t),2)
        errors[Mcount, dtcount] = error_L2
        plt.title("M = " + str(M) + ", dt = " + str(round(dt,4)))
        plt.show()
        print("The calculation took " + str(round(time.time()-start_time,2)) + " seconds")
        print("\n")
        dtcount = dtcount+1
    Mcount = Mcount+1
    dtcount = 0
                
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
        plt.title("Numerical error as a function of temporal stepsize")
        plt.show()
    
    ### Plot the numerical error as a function of the spatial stepsize
    if len(M_array)>2:
        for dtcount, dt in enumerate(dt_array):
            plt.loglog(M_array, errors[:,dtcount], "o--", label= "a = 1/" + str(N_array[dtcount]))
        a1 = np.transpose(errors)[0][0]/(1/M_array[0])
        a2 = np.transpose(errors)[0][0]/(1/M_array[0]**2)
        a3 = np.transpose(errors)[0][0]/(1/M_array[0]**3)
        plt.loglog(M_array, a1/M_array, "-", label="Ideal 1st order")
        plt.loglog(M_array, a2/M_array**2,"-", label="Ideal 2nd order")
        plt.loglog(M_array, a3/M_array**3, "-", label="Ideal 3rd order")
        plt.xlabel(r"Number of spatial steps")
        plt.ylabel("Numerical L2-error")
        plt.legend(prop={'size': 10}, loc="best")
        plt.title("Numerical error as a function of spatial stepsize")
        plt.show()     