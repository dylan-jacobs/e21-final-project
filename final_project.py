import numpy as np
import sympy as sp
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MC_LENGTH = 5
PP_LENGTH = 4
DP_LENGTH = 2

def kinematics5_simulator_dh(theta_vals):
    # Define the total length
    length = MC_LENGTH + PP_LENGTH + DP_LENGTH
    og_pos = np.array([length, 0, 0])

    theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip = theta_vals
    
    # Define symbolic variables
    theta, alph, d, r = sp.symbols('theta alph d r')
    
    # Define the DH parameter matrix
    DH_Param = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alph), sp.sin(theta) * sp.sin(alph), r * sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta) * sp.cos(alph), -sp.cos(theta) * sp.sin(alph), r * sp.sin(theta)],
        [0, sp.sin(alph), sp.cos(alph), d],
        [0, 0, 0, 1]
    ])
    
    # Substitute DH parameters for each joint
    CMC_Abd_Matrix = DH_Param.subs({theta: theta_cmc_horiz, alph: sp.pi/2, d: 0, r: 0})
    CMC_Flex_Matrix = DH_Param.subs({theta: theta_cmc, alph: -sp.pi/2, d: 0, r: MC_LENGTH})
    MCP_Abd_Matrix = DH_Param.subs({theta: theta_mcp_horiz, alph: sp.pi/2, d: 0, r: 0})
    MCP_Flex_Matrix = DH_Param.subs({theta: theta_mcp, alph: 0, d: 0, r: PP_LENGTH})
    IP_Flex_Matrix = DH_Param.subs({theta: theta_ip, alph: 0, d: 0, r: DP_LENGTH})
    
    # Compute the transformation matrices
    EP_Matrix = sp.simplify(CMC_Abd_Matrix * CMC_Flex_Matrix * MCP_Abd_Matrix * MCP_Flex_Matrix * IP_Flex_Matrix)
    CMC_net = sp.simplify(CMC_Abd_Matrix * CMC_Flex_Matrix)
    MCP_net = sp.simplify(CMC_net * MCP_Abd_Matrix * MCP_Flex_Matrix)
    
    MCP_Pos = np.array(CMC_net[:-1, 3].evalf(), dtype=float).flatten()
    IP_Pos = np.array(MCP_net[:-1, 3].evalf(), dtype=float).flatten()
    EP_Pos = np.array(EP_Matrix[:-1, 3].evalf(), dtype=float).flatten()
    
    results = np.column_stack((MCP_Pos, IP_Pos, EP_Pos))
    EP_Change = EP_Pos - og_pos
    
    return results, EP_Change

# computes the jacobian of the position vector, which 
# consists of x, y, z coordinate functions of theta_vec.
# theta_vec = length-5 symbolic variable vector
def jacobian(theta_vals):
    theta, alph, d, r = sp.symbols('theta alph d r')
    # Define the DH parameter matrix
    DH_Param = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alph), sp.sin(theta) * sp.sin(alph), r * sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta) * sp.cos(alph), -sp.cos(theta) * sp.sin(alph), r * sp.sin(theta)],
        [0, sp.sin(alph), sp.cos(alph), d],
        [0, 0, 0, 1]
    ])
    
    # Substitute DH parameters for each joint
    theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip = theta_vals

    CMC_Abd_Matrix = DH_Param.subs({theta: theta_cmc_horiz, alph: sp.pi/2, d: 0, r: 0})
    CMC_Flex_Matrix = DH_Param.subs({theta: theta_cmc, alph: -sp.pi/2, d: 0, r: MC_LENGTH})
    MCP_Abd_Matrix = DH_Param.subs({theta: theta_mcp_horiz, alph: sp.pi/2, d: 0, r: 0})
    MCP_Flex_Matrix = DH_Param.subs({theta: theta_mcp, alph: 0, d: 0, r: PP_LENGTH})
    IP_Flex_Matrix = DH_Param.subs({theta: theta_ip, alph: 0, d: 0, r: DP_LENGTH})
    
    # Compute the transformation matrices
    EP_Matrix = sp.simplify(CMC_Abd_Matrix * CMC_Flex_Matrix * MCP_Abd_Matrix * MCP_Flex_Matrix * IP_Flex_Matrix)
    position_vector = EP_Matrix[:3, 3]
    jacobian = position_vector.jacobian([theta_vals])
    return jacobian

def solset():
    theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip = sp.symbols('theta_cmc_horiz theta_cmc theta_mcp_horiz theta_mcp theta_ip')
    delta_theta_vals = [theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip]
    eq1 = sp.zeros(3,1) == jacobian(delta_theta_vals) * sp.Matrix(delta_theta_vals)
    solset = sp.solve(eq1, theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip)
    return solset 

def f(theta_vals):
    # define thumb joint lengths
    MC_LENGTH = 5
    PP_LENGTH = 4
    DP_LENGTH = 2

    # We will only pass in a None type for theta_vals if we want to solve for theta values. 
    # Otherwise, we can use this function to reconstruct a position given KNOWN theta values
    if len(theta_vals) == 0: 
        # Define symbolic variables
        theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip = sp.symbols('theta_cmc_horiz theta_cmc theta_mcp_horiz theta_mcp theta_ip')
        theta_vals = [theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip]
    positions_matrix, position_change = kinematics5_simulator_dh(theta_vals, MC_LENGTH, PP_LENGTH, DP_LENGTH)

    return position_change

def get_thumb_constraints():
    minima = [10.2, 31.2, 0, 60, 88] # minima adduction, flexion angles
    maxima = [62.9, 61.2, 10, 8.1, 12] # maxima extension, abduction
    return minima, maxima

def rand_params():

    # create lower and upper bounds for random gaussian function
    # parameters
    lo, hi = get_thumb_constraints()

    # sample uniformly in between lower and upper bounds
    q = np.random.uniform(lo, hi, 5)

    # convert to float32 for quicker reconstruction
    q = q.astype(np.float32)

    # take 2d array and flatten down to 1d array
    return q.flatten()

def constraint_function(theta_vals):
    minima, maxima = get_thumb_constraints()
    g_sum = 0
    for i in range(len(minima)):
        g_sum += max(theta_vals[i] - minima[i], 0)
        g_sum += max(maxima[i] - theta_vals[i], 0)

    return g_sum

def objective(theta_vals, final_position):
    lambda_val = 100

    return sum(f(theta_vals)) + (lambda_val*(constraint_function(theta_vals))**2)

def compute_final_position(final_position):
    q = rand_params()
    res = scipy.optimize.minimize(fun=objective, x0=q, args=final_position, method='Powell', options=dict(maxfev=10000))
    final_theta = res.x
    error = objective(final_theta, None)
    print(f'Final position: {f(final_theta)}')
    print(f'Error: {error}')

# compute_final_position([1, 1, 1])

sol_set = solset()
print(sol_set)


def plot_thumb_3d(theta_vals):

    position_results, _ = kinematics5_simulator_dh(theta_vals)
    origin = np.zeros((3, 1))
    position_results = np.concatenate([origin, position_results], axis=1)

    xs, ys, zs = zip(*position_results.T) # need to transpose to get matrix that can be iterated over its rows of length 3 (required for zip())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs, ys, zs, '-o', label='Thumb segments')

    # Set labels and equal aspect ratio
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Thumb Representation')
    ax.legend()
    plt.show()

theta_vals = [np.pi/6, np.pi/6, -np.pi/4, -np.pi/4, np.pi/6]  # Example joint angles in radians
plot_thumb_3d(theta_vals)

## STEPS TO PROJECT

#1) WRITE CONSTRAINT EQUATIONS FOR EACH JOINT
#2) APPLY CONSTRAINT EQNS TO POS ARRAYS
        #-) POS ARRAYS USED WILL BE DIFFERENCE BETWEEN SYMBOLIC ABOVE AND DESIRED LOCATION
        #-) 9 EQNS, 5 VARS CAN ELIMINATE SOME
#3) USE NUMERICAL METHODS OF CHOICE TO MINIMIZE ALL VARIABLES IN EQNS WITH CONSTRAINTS
#4) CHECK WITH FUSION MODEL