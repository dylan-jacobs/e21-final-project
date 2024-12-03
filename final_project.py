import numpy as np
import sympy as sp

def kinematics5_simulator_dh(theta_cmc_horiz, theta_cmc, theta_mcp_horiz, theta_mcp, theta_ip, mc_length, pp_length, dp_length):
    # Define the total length
    length = mc_length + pp_length + dp_length
    og_pos = np.array([length, 0, 0])
    
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
    CMC_Flex_Matrix = DH_Param.subs({theta: theta_cmc, alph: -sp.pi/2, d: 0, r: mc_length})
    MCP_Abd_Matrix = DH_Param.subs({theta: theta_mcp_horiz, alph: sp.pi/2, d: 0, r: 0})
    MCP_Flex_Matrix = DH_Param.subs({theta: theta_mcp, alph: 0, d: 0, r: pp_length})
    IP_Flex_Matrix = DH_Param.subs({theta: theta_ip, alph: 0, d: 0, r: dp_length})
    
    # Compute the transformation matrices
    EP_Matrix = sp.simplify(CMC_Abd_Matrix * CMC_Flex_Matrix * MCP_Abd_Matrix * MCP_Flex_Matrix * IP_Flex_Matrix)
    CMC_net = sp.simplify(CMC_Abd_Matrix * CMC_Flex_Matrix)
    MCP_net = sp.simplify(CMC_net * MCP_Abd_Matrix * MCP_Flex_Matrix)
    
    MCP_Pos = np.array(CMC_net[:-1, 3].evalf(), dtype=float).flatten()
    IP_Pos = np.array(MCP_net[:-1, 3].evalf(), dtype=float).flatten()
    EP_Pos = np.array(EP_Matrix[:-1, 3].evalf(), dtype=float).flatten()
    
    results = np.column_stack((EP_Pos, IP_Pos, MCP_Pos))
    EP_Change = EP_Pos - og_pos
    
    return results, EP_Change


## STEPS TO PROJECT

#1) WRITE CONSTRAINT EQUATIONS FOR EACH JOINT
#2) APPLY CONSTRAINT EQNS TO POS ARRAYS
        #-) POS ARRAYS USED WILL BE DIFFERENCE BETWEEN SYMBOLIC ABOVE AND DESIRED LOCATION
        #-) 9 EQNS, 5 VARS CAN ELIMINATE SOME
#3) USE NUMERICAL METHODS OF CHOICE TO MINIMIZE ALL VARIABLES IN EQNS WITH CONSTRAINTS
#4) CHECK WITH FUSION MODEL