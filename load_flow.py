from create_bus_admittance_matrix import create_bus_admittance_matrix
import json
import numpy as np
from scipy.sparse import lil_matrix, csc_array, csr_array, csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve



def load_flow(filename):
    #init

    y_bus = create_bus_admittance_matrix(filename)

    g_bus = np.real(y_bus)
    b_bus = np.imag(y_bus)

    with open(filename) as f:
        data = json.load(f)

    s_base = float(data["system_data"]["S_Base_MVA"])


    #Power Flow Equations

    def Pi_x(i, V, theta, G, B):
        V = V.tocsr()
        theta = theta.tocsr()
        G = G.tocsr()
        B = B.tocsr()

        # Calculate delta_theta matrix
        delta_theta = theta[i,0] - theta.toarray().flatten()

        # Compute P_i  using dot product
        V_i = V[i,0]
        V_all = V.toarray().flatten()
        G_row = G.getrow(i).toarray().flatten()
        B_row = B.getrow(i).toarray().flatten()

        P_i = V_i * np.dot(V_all.flatten(),G_row * np.cos(delta_theta) + B_row * np.sin(delta_theta))

        return P_i

    def Qi_x(i, V, theta, G, B):
        V = V.tocsr()
        theta = theta.tocsr()
        G = G.tocsr()
        B = B.tocsr()

        # Calculate delta_theta matrix
        delta_theta = theta[i,0] - theta.toarray().flatten()

        # Compute Q_i  using dot product
        V_i = V[i,0]
        V_all = V.toarray().flatten()
        G_row = G.getrow(i).toarray().flatten()
        B_row = B.getrow(i).toarray().flatten()

        Q_i = V_i * np.dot(V_all.flatten(),G_row * np.sin(delta_theta) - B_row * np.cos(delta_theta))

        return Q_i



    #---- Extract variables, power injections and bus types ----
    swing_bus_name_to_index = {}
    pv_bus_names_to_indexes = {}
    pq_bus_names_to_indexes = {}
    non_swing_bus_names_to_indexes = {}

    active_power_injections_pu = []
    reactive_power_injections_pu = []

    v_variables = [] #pu
    theta_variables = [] #radians

    p_equation_count = 0
    q_equation_count = 0


    variable_index = 0
    for bus_name, bus_info in data["bus_data"].items():
        bus_type = bus_info["load_flow_type"]
        if bus_type == "3": #swing bus
            if swing_bus_name_to_index == {}:
                swing_bus_name_to_index.update({bus_name:variable_index})
            else:
                print("Double Swing Detected")

            v_variables.append(1.06)
            theta_variables.append(0)

        elif bus_type == "2": #pv bus
            v_variables.append(float(bus_info["voltage_magnitude"])) #scheduled power
            theta_variables.append(0) #flat start
            pv_bus_names_to_indexes.update({bus_name: variable_index})
            non_swing_bus_names_to_indexes.update({bus_name: variable_index})
            p_equation_count += 1

        elif bus_type == "0" or bus_type == "1": #pq bus
            v_variables.append(1) #flat start
            theta_variables.append(0) #flat start
            pq_bus_names_to_indexes.update({bus_name: variable_index})
            non_swing_bus_names_to_indexes.update({bus_name: variable_index})
            p_equation_count += 1
            q_equation_count += 1

        active_power_injections_pu.append((float(bus_info["active_generation_MW"]) - float(bus_info["active_load_MW"])) / s_base)
        reactive_power_injections_pu.append((float(bus_info["reactive_generation_MVAR"]) - float(bus_info["reactive_load_MVAR"])) / s_base)

        variable_index = variable_index + 1

    #Data Conversion
    active_power_injections_pu = np.array(active_power_injections_pu).reshape(-1, 1)
    active_power_injections_pu = csc_array(active_power_injections_pu)
    reactive_power_injections_pu = np.array(reactive_power_injections_pu).reshape(-1, 1)
    reactive_power_injections_pu = csc_array(reactive_power_injections_pu)

    v_variables = np.array(v_variables).reshape(-1, 1)
    v_variables = csc_array(v_variables)
    theta_variables = np.array(theta_variables).reshape(-1, 1)
    theta_variables = csc_array(theta_variables)

    #Decleration of constants
    pv_bus_amount = len(pv_bus_names_to_indexes)
    pq_bus_amount =  len(pq_bus_names_to_indexes)

    v_variable_count = pq_bus_amount
    theta_variable_count = pv_bus_amount + pq_bus_amount

    f_x_minus_f_zero = [0] * (p_equation_count + q_equation_count)


    #---- F(x) - Fx Equations ----
    def delta_F_x():
        P_x = []
        Q_x = []

        for bus_index in non_swing_bus_names_to_indexes.values(): #P Equations
            P_x.append(Pi_x(bus_index,v_variables,theta_variables,g_bus,b_bus) - active_power_injections_pu[bus_index, 0])

        for bus_index in pq_bus_names_to_indexes.values(): #Q Equations
            Q_x.append(Qi_x(bus_index,v_variables,theta_variables,g_bus,b_bus) - reactive_power_injections_pu[bus_index, 0])

        P_x = np.array(P_x).reshape(-1, 1)
        Q_x = np.array(Q_x).reshape(-1, 1)
        return csc_array(P_x),csc_array(Q_x)


    #Extract Unknowns
    def extract_unkonwns(theta_variables,v_variables):
        x_k = []

        #Extract theta
        for theta_idx in non_swing_bus_names_to_indexes.values():
            x_k.append([theta_variables[theta_idx,0]])

        # Extract V
        for v_idx in pq_bus_names_to_indexes.values():
            x_k.append([v_variables[v_idx,0]])

        return csc_array(x_k)

    #Insert Unknowns
    def insert_unknowns(x_k,theta_variables,v_variables):
        index = 0

        #Insert theta
        theta_variables_list = theta_variables.toarray().flatten().tolist()

        for theta_idx in non_swing_bus_names_to_indexes.values():
            theta_variables_list[theta_idx] = x_k[index,0]
            index = index + 1


        theta_variables = np.array(theta_variables_list).reshape(-1, 1)
        theta_variables = csc_array(theta_variables)

        # Insert V
        v_variables_list =  v_variables.toarray().flatten().tolist()

        for v_idx in pq_bus_names_to_indexes.values():
            v_variables_list[v_idx] = x_k[index,0]
            index = index + 1

        v_variables = np.array(v_variables_list).reshape(-1, 1)
        v_variables = csc_array(v_variables)
        return theta_variables, v_variables


    ##---- Compute Jacobian ----
    def compute_jacobian():
        J_11 = lil_matrix((theta_variable_count,theta_variable_count))
        J_12 = lil_matrix((theta_variable_count,v_variable_count))
        J_21 = lil_matrix((v_variable_count,theta_variable_count))
        J_22 = lil_matrix((v_variable_count,v_variable_count))

        #Jacobian 1_1
        for j_row_num in range(0,p_equation_count):
            i = list(non_swing_bus_names_to_indexes.values())[j_row_num]

            for j_col_num in range(0,theta_variable_count):
                j = list(non_swing_bus_names_to_indexes.values())[j_col_num]

                if i != j:
                    J_11[j_row_num,j_col_num] = v_variables[i, 0] * v_variables[j, 0] * (g_bus[i,j] * np.sin(theta_variables[i, 0] - theta_variables[j, 0]) - b_bus[i,j]* np.cos(theta_variables[i,0] - theta_variables[j,0]))
                else:
                    J_11[j_row_num, j_col_num] = -1* Qi_x(i, v_variables, theta_variables, g_bus, b_bus)- b_bus[i,i] * v_variables[i, 0] * v_variables[j, 0]
                    print("J11:" + str(i))

        #Jacobian 1_2
        for j_row_num in range(0,p_equation_count):
            i = list(non_swing_bus_names_to_indexes.values())[j_row_num]

            for j_col_num in range(0,v_variable_count):
                j = list(pq_bus_names_to_indexes.values())[j_col_num]

                if i != j:
                    J_12[j_row_num,j_col_num] = v_variables[i, 0]* (g_bus[i,j]* np.cos(theta_variables[i, 0] - theta_variables[j, 0]) + b_bus[i,j]* np.sin(theta_variables[i, 0] - theta_variables[j, 0]))
                else:
                    J_12[j_row_num, j_col_num] = (Pi_x(i, v_variables, theta_variables, g_bus, b_bus) /  v_variables[i, 0]) + g_bus[i,i] * v_variables[i, 0]
                    print("J12:" + str(i))

        #Jacobian 2_1
        for j_row_num in range(0,q_equation_count):
            i = list(pq_bus_names_to_indexes.values())[j_row_num]

            for j_col_num in range(0,theta_variable_count):
                j = list(non_swing_bus_names_to_indexes.values())[j_col_num]

                if i != j:
                    J_21[j_row_num,j_col_num] = -1*v_variables[i, 0] * v_variables[j, 0] * (g_bus[i,j]* np.cos(theta_variables[i, 0] - theta_variables[j, 0]) + b_bus[i,j]* np.sin(theta_variables[i, 0] - theta_variables[j, 0]))
                else:
                    J_21[j_row_num, j_col_num] = Pi_x(i, v_variables, theta_variables, g_bus, b_bus) - g_bus[i,i] * v_variables[i, 0] * v_variables[i, 0]
                    print("J21:" + str(i))

        #Jacobian 2_2
        for j_row_num in range(0,q_equation_count):
            i = list(pq_bus_names_to_indexes.values())[j_row_num]

            for j_col_num in range(0,v_variable_count):
                j = list(pq_bus_names_to_indexes.values())[j_col_num]

                if i != j:
                    J_22[j_row_num,j_col_num] = v_variables[i, 0] * (g_bus[i,j]* np.sin(theta_variables[i, 0] - theta_variables[j, 0]) - b_bus[i,j]* np.cos(theta_variables[i, 0] - theta_variables[j, 0]))
                else:
                    J_22[j_row_num, j_col_num] = Qi_x(i, v_variables, theta_variables, g_bus, b_bus)/v_variables[i, 0] - b_bus[i,i] * v_variables[i, 0]
                    print("J22:" + str(i))


        J_1 = hstack([J_11,J_12])
        J_2 = hstack([J_21,J_22])
        jacobian = vstack([J_1, J_2])
        return jacobian


    #LU Factorization
    def sp_solver(jacobian, delta_Fx):
        jacobian_csc = jacobian.tocsc()
        delta_Fx_csc = (-1) * delta_Fx.toarray().flatten()
        X_dense = spsolve(jacobian_csc, delta_Fx_csc)
        X_lil = lil_matrix(X_dense.reshape(-1, 1))

        return X_lil

    #---- N-R Iteration Init ----
    n_max_iterations = 10
    tolerance = 10 ** -4

    #Initial F_(0)
    pi_x_minus_pi,qi_x_minus_qi = delta_F_x()
    F_delta_x = vstack([pi_x_minus_pi,qi_x_minus_qi])
    print("F_delta_x")
    print(F_delta_x)

    for k in range(1,n_max_iterations+1):
        print("Iteration Number : " + str(k))
        jacobian = compute_jacobian()
        print("jacobian")
        print(jacobian)

        x_k = extract_unkonwns(theta_variables, v_variables)
        print("x_k")
        print(x_k)

        delta_x_k = sp_solver(jacobian, F_delta_x)

        x_k_p_1 = x_k + delta_x_k
        print("x_k_p_1")
        print(x_k_p_1)

        theta_variables, v_variables = insert_unknowns(x_k_p_1, theta_variables, v_variables)
        print("theta_variables")
        print(theta_variables)
        print("v_variables")
        print(v_variables)

        pi_x_minus_pi, qi_x_minus_qi = delta_F_x()
        F_delta_x = vstack([pi_x_minus_pi, qi_x_minus_qi])
        print("F_delta_x")
        print(F_delta_x)
        if np.all(np.abs(F_delta_x.data) < tolerance) == True:
            print("System Converged After" + str(k) + "Iteration: " )
            break
        if k == n_max_iterations:
            print("System Diverged After " + str(k) + " Iteration: " )

    return theta_variables,v_variables


a,b = load_flow("ieee14bus.json")
print(a)
# print(b)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
# print(g)



