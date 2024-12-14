import numpy as np
import scipy.linalg
import sympy as sp
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MC_LENGTH = 5
PP_LENGTH = 4
DP_LENGTH = 2
MAX_LENGTH = MC_LENGTH + PP_LENGTH + DP_LENGTH

def kinematics5_simulator_dh(theta_vals):
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
    
    return results, EP_Pos

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
    theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip = sp.symbols('theta_cmc_horiz theta_cmc theta_mcp_horiz theta_mcp theta_ip')
    symbolic_theta = [theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip]

    CMC_Abd_Matrix = DH_Param.subs({theta: theta_cmc_horiz, alph: sp.pi/2, d: 0, r: 0})
    CMC_Flex_Matrix = DH_Param.subs({theta: theta_cmc, alph: -sp.pi/2, d: 0, r: MC_LENGTH})
    MCP_Abd_Matrix = DH_Param.subs({theta: theta_mcp_horiz, alph: sp.pi/2, d: 0, r: 0})
    MCP_Flex_Matrix = DH_Param.subs({theta: theta_mcp, alph: 0, d: 0, r: PP_LENGTH})
    IP_Flex_Matrix = DH_Param.subs({theta: theta_ip, alph: 0, d: 0, r: DP_LENGTH})
    
    # Compute the transformation matrices
    EP_Matrix = sp.simplify(CMC_Abd_Matrix * CMC_Flex_Matrix * MCP_Abd_Matrix * MCP_Flex_Matrix * IP_Flex_Matrix)
    position_vector = EP_Matrix[:3, 3]
    jacobian = position_vector.jacobian([symbolic_theta])
    subs = {theta_cmc_horiz: theta_vals[0], theta_cmc: theta_vals[1], theta_mcp_horiz: theta_vals[2], theta_mcp: theta_vals[3], theta_ip: theta_vals[4]}
    jacobian_evaluated = np.array(jacobian.subs(subs))
    return jacobian_evaluated

# ----------------------
# create numerical jacobian
# q1 is a 3x1 matrix representing the 
# equations depending on theta (R5) that
# represent x, y, z in R3
# ----------------------
def numerical_jacobian(theta_vals, order=4):
    delta = 0.001 # used to artificially create other theta vectors

    rows = 3
    cols = len(theta_vals)
    jacobian = np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            delta_vals = np.zeros((1, cols))
            delta_vals[0, j] = delta
            
            if (order == 2): # order 2 derivative approximation
                theta2 = theta_vals + delta_vals
                theta1 = theta_vals - delta_vals

                _, f2 = kinematics5_simulator_dh(theta2.flatten().tolist())
                _, f1 = kinematics5_simulator_dh(theta1.flatten().tolist())
                jacobian[i, j] = (f2[i]-f1[i])/(2*delta)
            else: # assume fourth order
                theta4 = theta_vals + (2*delta_vals)
                theta3 = theta_vals + delta_vals
                theta2 = theta_vals - delta_vals
                theta1 = theta_vals - (2*delta_vals)

                _, f4 = kinematics5_simulator_dh(theta4.flatten().tolist())
                _, f3 = kinematics5_simulator_dh(theta3.flatten().tolist())
                _, f2 = kinematics5_simulator_dh(theta2.flatten().tolist())
                _, f1 = kinematics5_simulator_dh(theta1.flatten().tolist())
                jacobian[i, j] = ((8*f3[i]) - (8*f2[i]) + f1[i] - f4[i])/(12*delta)
    return jacobian

def solset(theta_vals,theta_past):
    J = numerical_jacobian(theta_vals)
    null = scipy.linalg.null_space(J)
    print(J)
    print(null)

    #Using span of vectors in nullspace retroactively determine a set of solution thets that would minimize magnitude delta theta move

    #MINI OPTIMIZATION PROBLEM
    #MINIMIZE THE FUNCTION OF 2 VARIABLES WHICH IS THE SPAN OF THE NULLSPACE MINUS THE PAST THETA VALS -> BEST
    #N1 and N2 are numeric not symbolic
    # mag_delta_theta_move = (theta_vals-theta_past).norm()
  
    # Objective function to minimize
    def objective(c):
        c1, c2 = c  # Coefficients
        delta_theta = (c1 * null[:, 0]) + (c2 * null[:, 1])
        updated_theta = theta_vals + delta_theta

        # sanity check - ensure position isn't changing much at all
        _, old_pos = kinematics5_simulator_dh(theta_vals)
        _, new_pos = kinematics5_simulator_dh(updated_theta)
        if not (np.allclose(old_pos, new_pos, atol=1e-2)):

            print('Error: ', np.linalg.norm(old_pos - new_pos))

            print('Old pos', old_pos)
            print('New pos', new_pos)

        return np.dot(updated_theta - theta_past, updated_theta - theta_past)

    # Initial guess for coefficients
    initial_guess = [0, 0]

    # Minimize the objective function
    result = scipy.optimize.minimize(objective, initial_guess, method='L-BFGS-B', jac='2-point', options=dict(maxfun=10000))
    s_opt, t_opt = result.x
    theta_opt = theta_vals + (s_opt * null[:, 0]) + (t_opt * null[:, 1])
    #Find linear combination of N1 and N2 which minimizes 
    

    return theta_opt

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
# analytical_jacobian = jacobian(theta_vals)
num_jacobian = numerical_jacobian(theta_vals)
# print(analytical_jacobian)
# print(num_jacobian)
# plot_thumb_3d(theta_vals)

def iterative_ik(theta_vals,qf):
    if (np.linalg.norm(qf) > MAX_LENGTH): 
        print("Input distance outside of the thumb's range!")
        return

    # Intialize guess for new theta
    tol = .1
    delta_q = np.inf
    theta_g = theta_vals

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while np.linalg.norm(delta_q) > tol:

        lam = 10
        results, current_pos = kinematics5_simulator_dh(theta_g)

        origin = np.zeros((3, 1))
        position_results = np.column_stack([origin, results])
        xs, ys, zs = zip(*position_results.T) # need to transpose to get matrix that can be iterated over its rows of length 3 (required for zip())

        ax.clear()
        ax.plot(xs, ys, zs, '-o', label='Thumb segments')
        ax.plot(qf[0], qf[1], qf[2], '-x', label='Desired final position')
        plt.pause(0.01)

        delta_q = qf - current_pos # difference between desired final position and current pos

        # Minimize l(delta_theta ) = ||qf-f(theta_g)+J*(theta_g)delta_theta||^2 + lambda||delta_theta||^2

        # Implement damped jacobian pseudoinverse
        J = numerical_jacobian(theta_g)
        JT = np.transpose(J)
        delta_theta = np.linalg.inv((JT@J)+(lam*np.identity(n=5)))@(JT@delta_q)
        # theta_g += delta_theta
        
        theta_g = solset(theta_g + delta_theta, theta_g)
        
    plt.show()

initial_theta_vals = np.array([0.1, 0.2, 0.1, 0.3, 0.4])
_, initial_end_point = kinematics5_simulator_dh(initial_theta_vals)
qf = np.array(initial_end_point) + np.array([-4, -2, -2])
iterative_ik(initial_theta_vals, qf)

## STEPS TO PROJECT

#1) WRITE CONSTRAINT EQUATIONS FOR EACH JOINT
#2) APPLY CONSTRAINT EQNS TO POS ARRAYS
        #-) POS ARRAYS USED WILL BE DIFFERENCE BETWEEN SYMBOLIC ABOVE AND DESIRED LOCATION
        #-) 9 EQNS, 5 VARS CAN ELIMINATE SOME
#3) USE NUMERICAL METHODS OF CHOICE TO MINIMIZE ALL VARIABLES IN EQNS WITH CONSTRAINTS
#4) CHECK WITH FUSION MODEL