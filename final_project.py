import numpy as np
import scipy.linalg
import sympy as sp
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MC_LENGTH = 5
PP_LENGTH = 4
DP_LENGTH = 2
MAX_LENGTH = MC_LENGTH + PP_LENGTH + DP_LENGTH

# computes the final position of the thumb-tip
# using forward kinematics and given theta values
# for the thumb's 5 degrees of freedom via Denavhit-
# Hartenburg matrix multiplication
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

# computes the analytical jacobian of the position vector, which 
# consists of x, y, z coordinate functions of theta_vec.
# theta_vec = length-5 symbolic variable vector
def analytical_jacobian(theta_vals):
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

# create numerical jacobian
# q1 is a 3x1 matrix representing the 
# equations depending on theta (R5) that
# represent x, y, z in R3
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

# find the best theta that achieves the same 
# final position
def optimize_theta(theta_vals,theta_past):
    J = numerical_jacobian(theta_vals)
    null = scipy.linalg.null_space(J)

    # Using span of vectors in nullspace retroactively determine a set of solution thets that would minimize magnitude delta theta move
    # MINI OPTIMIZATION PROBLEM
    # MINIMIZE THE FUNCTION OF 2 VARIABLES WHICH IS THE SPAN OF THE NULLSPACE MINUS THE PAST THETA VALS -> BEST
    # N1 and N2 are numeric not symbolic
    # mag_delta_theta_move = (theta_vals-theta_past).norm()
  
    # Objective function to minimize
    def objective(c):
        c1, c2 = c  # Coefficients
        delta_theta = (c1 * null[:, 0]) + (c2 * null[:, 1])
        updated_theta = theta_vals + delta_theta

        # Core objective: minimize the magnitude of delta_theta
        core_objective = np.dot(updated_theta - theta_past, updated_theta - theta_past)

        # Add penalty terms for constraints
        constraint_penalty = 100  # Scaling factor for penalties
        penalties = 0 
        # is this just adding a numerical value to the objective function or is it minimizing with the constraint penalties as functions of theta?

        minima, maxima = get_thumb_constraints()
        
        for i in range(len(updated_theta)):  # Iterate over theta components
            if updated_theta[i] > maxima[i]:
                penalties += constraint_penalty * (updated_theta[i] - maxima[i]) ** 2
            elif updated_theta[i] < minima[i]:
                penalties += constraint_penalty * (minima[i] - updated_theta[i]) ** 2

        # Total objective: core objective + penalties
        return core_objective + penalties

    # Initial guess for coefficients
    initial_guess = [0, 0]

    # Minimize the objective function
    result = scipy.optimize.minimize(objective, initial_guess, method='L-BFGS-B', jac='2-point', options=dict(maxfun=10000))
    s_opt, t_opt = result.x

    # linear combination of nullspace vectors that minimizes delta_theta
    theta_opt = theta_vals + (s_opt * null[:, 0]) + (t_opt * null[:, 1])

    return theta_opt

# Return average biological minimum and 
# maximum angles for each degree of 
# freedom in thumb (radians)
def get_thumb_constraints():
    minima = np.deg2rad([-10.2, -31.2, 0, -60, -88]) # minima adduction, flexion angles
    maxima = np.deg2rad([62.9, 61.2, 10, 8.1, 12]) # maxima extension, abduction
    return minima, maxima

# Get random angle parameters 
def rand_params():
    # upper/lower angle bounds for thumb's degrees of freedom
    lo, hi = get_thumb_constraints()

    # randomly sample between upper/lower bounds
    q = np.random.uniform(lo, hi, 5)

    # convert to float32
    q = q.astype(np.float32)

    # convert to 1D array
    return q.flatten()


# Graphing methods
def plot_thumb_3d(ax, position_matrix, qf):

    origin = np.zeros((3, 1))
    position_results = np.column_stack([origin, position_matrix])
    xs, ys, zs = zip(*position_results.T) # need to transpose to get matrix that can be iterated over its rows of length 3 (required for zip())

    ax.clear()
    ax.plot(xs, ys, zs, '-o', label='Thumb segments')
    ax.plot(qf[0], qf[1], qf[2], '-x', label='Desired final position')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Thumb Representation')
    ax.legend()

    plt.pause(0.01)
    plt.autoscale(False)
    plt.savefig('final_thumb_position.png')

def plot_error(step_list, error_list):
    plt.figure()
    plt.plot(step_list, error_list, '-o', label='Error vs Step')
    plt.xlabel('Step')
    plt.ylabel('Error (||delta_q||)')
    plt.title('Error vs. Step for Iterative IK')
    plt.grid(True)
    plt.legend()
    plt.savefig('error_plot.png')

def create_animation(position_tensor, qf):
    frames = position_tensor.shape[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the line object
    origin = np.zeros((3, 1))
    initial_position_matrix = np.column_stack([origin, position_tensor[:, :, 0]])
    xs, ys, zs = zip(*initial_position_matrix.T)
    line, = ax.plot(xs, ys, zs, '-o', label='Thumb segments')
    ax.plot(qf[0], qf[1], qf[2], '-x', label='Desired final position')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Thumb Representation')
    ax.legend()

    # Clear plot
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    # Update data 
    def update(frame):
        position_matrix = np.column_stack([origin, position_tensor[:, :, frame]])
        xs, ys, zs = zip(*position_matrix.T) # need to transpose to get matrix that can be iterated over its rows of length 3 (required for zip())
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        return line,
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=250)
    ani.save('thumb_movement.gif', writer='Pillow', fps=5)
    # Show the animation
    plt.show()



# ----- ITERATIVE METHOD -----
# iterate until theta values are optimized to reach 
# desired endpoint qf and minimize the delta_theta
# of every step
def iterative_ik(theta_vals,qf):

    # Ensure our given input is inside the thumb's reach
    if (np.linalg.norm(qf) > MAX_LENGTH): 
        print("Input distance outside of the thumb's range!")
        return

    # Intialize guess for new theta
    tol = .1    # norm of error between qf and thumb's final position
    delta_q = np.inf
    theta_g = theta_vals

    #for error plotting
    error_list = []  # To store error at each step
    step_list = []   # To store step indices
    step = 0

    # For 3D thumb plotting
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    position_tensor = np.zeros((3, 3, 100)) # store positions over time

    while np.linalg.norm(delta_q) > tol:

        lam = 10
        position_matrix, thumb_tip_position = kinematics5_simulator_dh(theta_g)
        position_tensor[:, :, step] = position_matrix
        plot_thumb_3d(ax, position_matrix, qf)

        delta_q = qf - thumb_tip_position # difference between desired final position and current pos

        # Minimize l(delta_theta ) = ||qf-f(theta_g)+J*(theta_g)delta_theta||^2 + lambda||delta_theta||^2

        # Implement damped jacobian pseudoinverse
        J = numerical_jacobian(theta_g)
        JT = np.transpose(J)
        delta_theta = np.linalg.inv((JT@J)+(lam*np.identity(n=5)))@(JT@delta_q)
        
        theta_g = optimize_theta(theta_g + delta_theta, theta_g)
        
        error_list.append(np.linalg.norm(delta_q))  # Track the norm of delta_q
        step_list.append(step)  # Track the current step
        step += 1
    
    # trim position tensor
    position_tensor = position_tensor[:, :, :len(step_list)]

    plot_error(step_list, error_list)
    create_animation(position_tensor, qf)
    plt.show()

initial_theta_vals = np.array( [np.pi/6, np.pi/6, 0, -np.pi/4, -np.pi/6])
initial_theta_vals = rand_params()
_, qf = kinematics5_simulator_dh(rand_params())
iterative_ik(initial_theta_vals, qf)