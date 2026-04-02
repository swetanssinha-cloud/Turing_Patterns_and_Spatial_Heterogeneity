'''Multiple simulations'''

from GSw_tanhGraph import solver, first_derivative,Du, Dv, F, k, dx, dt, x
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt


N = 200
n_steps = 400 

def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1]
def solve_for_theta(dVx, dVy):
    a = [1,0]

    dV = np.stack((dVx,dVy), axis = 0)

    dV_new = np.stack((dVy, -1 * dVx), axis = 0)

    resultant = dot_product(a, dV_new)


    cos_theta = resultant / (np.sqrt(dVx**2 + dVy**2) * np.sqrt(a[0]**2 + a[1]**2)) 

    # theta = np.arctan(dVy/dVx)

    #theta = np.arccos(cos_theta) * (180/np.pi) #convert to degrees


    '''this is where we transform cosine theta to cosine 2 theta and then to theta, which is the angle between the vector and the x-axis. '''

    cos_theta_squared = cos_theta**2

    cos_2_theta = 2 * cos_theta_squared - 1

    theta = 0.5 * np.arccos(cos_2_theta)

    theta = theta * (180/np.pi) #convert to degrees
    mean_theta_column = np.mean(theta, axis = 0)
    return mean_theta_column



def run_simulation(seed): 

    np.random.seed(seed)

    if seed % 25 == 0: #I want to keep the value at 10 - because I will be running 500 sims eventually
        print(f"Simulation {seed} finished")


    U = np.ones((N,N))
    V = np.zeros((N,N))
    r = 20
    U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.75 #orginally was 0.50
    V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.25
    U += 0.05 * np.random.rand(N,N)
    V += 0.05 * np.random.rand(N,N)


    for i in range(n_steps):
        for _ in range(100):
            U, V = solver(U, V, Du, Dv, F, k, dx, dt)

    dVx = first_derivative(V, 1)
    dVy = first_derivative(V, 0)

    return solve_for_theta(dVx, dVy)


if __name__=='__main__':
    num_simulations = 50 #run 500 simulations and average the results to get a smoother curve.
    seeds = np.arange(num_simulations)

    with Pool(cpu_count()-2) as pool:
        results = pool.map(run_simulation, seeds)

        results = np.array(results)

        mean_theta = np.mean(results, axis=0)
        plt.plot(x, mean_theta)
        plt.xlabel('X')
        plt.ylabel('Mean Theta (degrees)')
        plt.title('Mean Theta vs X')
        plt.show()



