import json
import numpy as np
from scipy.sparse import csc_array, lil_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve
from create_bus_admittance_matrix_solution import create_bus_admittance_matrix


def load_flow(filename):
    # Generate the Y-Bus matrix
    y_bus = create_bus_admittance_matrix(filename)
    g_bus = y_bus.real
    b_bus = y_bus.imag

    with open(filename, 'r') as f:
        data = json.load(f)

    s_base = float(data["system_data"]["S_Base_MVA"])

    # --- Helper Functions ---
    def calculate_power(i, V, theta, G, B, is_reactive=False):
        delta_theta = theta[i, 0] - theta.toarray().flatten()
        V_i = V[i, 0]
        V_all = V.toarray().flatten()
        G_row = G.getrow(i).toarray().flatten()
        B_row = B.getrow(i).toarray().flatten()

        if is_reactive:
            return V_i * np.dot(V_all, G_row * np.sin(delta_theta) - B_row * np.cos(delta_theta))
        return V_i * np.dot(V_all, G_row * np.cos(delta_theta) + B_row * np.sin(delta_theta))

    def delta_F_x():
        P_x = [
            calculate_power(i, v_variables, theta_variables, g_bus, b_bus) -
            active_power_injections_pu[i, 0]
            for i in non_swing_bus_indexes.values()
        ]

        Q_x = [
            calculate_power(i, v_variables, theta_variables, g_bus, b_bus, is_reactive=True) -
            reactive_power_injections_pu[i, 0]
            for i in pq_bus_indexes.values()
        ]

        return csc_array(np.array(P_x).reshape(-1, 1)), csc_array(np.array(Q_x).reshape(-1, 1))

    def compute_jacobian(
        p_equation_count, q_equation_count, theta_variable_count, v_variable_count,
        g_bus, b_bus, v_variables, theta_variables, pq_bus_indexes, non_swing_bus_indexes
    ):
        J_11 = lil_matrix((p_equation_count, theta_variable_count))
        J_12 = lil_matrix((p_equation_count, v_variable_count))
        J_21 = lil_matrix((q_equation_count, theta_variable_count))
        J_22 = lil_matrix((q_equation_count, v_variable_count))

        # Jacobian 1_1
        for row, i in enumerate(non_swing_bus_indexes.values()):
            for col, j in enumerate(non_swing_bus_indexes.values()):
                if i != j:
                    J_11[row, col] = (
                        v_variables[i, 0]
                        * v_variables[j, 0]
                        * (
                            g_bus[i, j] * np.sin(theta_variables[i, 0] - theta_variables[j, 0])
                            - b_bus[i, j] * np.cos(theta_variables[i, 0] - theta_variables[j, 0])
                        )
                    )
                else:
                    J_11[row, col] = (
                        -calculate_power(i, v_variables, theta_variables, g_bus, b_bus, is_reactive=True)
                        - b_bus[i, i] * v_variables[i, 0] ** 2
                    )

        # Jacobian 1_2
        for row, i in enumerate(non_swing_bus_indexes.values()):
            for col, j in enumerate(pq_bus_indexes.values()):
                if i != j:
                    J_12[row, col] = (
                        v_variables[i, 0]
                        * (
                            g_bus[i, j] * np.cos(theta_variables[i, 0] - theta_variables[j, 0])
                            + b_bus[i, j] * np.sin(theta_variables[i, 0] - theta_variables[j, 0])
                        )
                    )
                else:
                    J_12[row, col] = (
                        calculate_power(i, v_variables, theta_variables, g_bus, b_bus) / v_variables[i, 0]
                        + g_bus[i, i] * v_variables[i, 0]
                    )

        # Jacobian 2_1
        for row, i in enumerate(pq_bus_indexes.values()):
            for col, j in enumerate(non_swing_bus_indexes.values()):
                if i != j:
                    J_21[row, col] = (
                        -v_variables[i, 0]
                        * v_variables[j, 0]
                        * (
                            g_bus[i, j] * np.cos(theta_variables[i, 0] - theta_variables[j, 0])
                            + b_bus[i, j] * np.sin(theta_variables[i, 0] - theta_variables[j, 0])
                        )
                    )
                else:
                    J_21[row, col] = (
                        calculate_power(i, v_variables, theta_variables, g_bus, b_bus)
                        - g_bus[i, i] * v_variables[i, 0] ** 2
                    )

        # Jacobian 2_2
        for row, i in enumerate(pq_bus_indexes.values()):
            for col, j in enumerate(pq_bus_indexes.values()):
                if i != j:
                    J_22[row, col] = (
                        v_variables[i, 0]
                        * (
                            g_bus[i, j] * np.sin(theta_variables[i, 0] - theta_variables[j, 0])
                            - b_bus[i, j] * np.cos(theta_variables[i, 0] - theta_variables[j, 0])
                        )
                    )
                else:
                    J_22[row, col] = (
                        calculate_power(i, v_variables, theta_variables, g_bus, b_bus, is_reactive=True)
                        / v_variables[i, 0]
                        - b_bus[i, i] * v_variables[i, 0]
                    )

        J_1 = hstack([J_11, J_12])
        J_2 = hstack([J_21, J_22])
        return vstack([J_1, J_2])

    def update_variables(delta_x_k, theta_variables, v_variables):
        """
        Updates theta_variables and v_variables based on the corrections in delta_x_k.
        """
        theta_variables = theta_variables.toarray().astype(float)  # Convert to float64 for compatibility
        v_variables = v_variables.toarray().astype(float)

        # Update theta variables (non-swing buses)
        delta_theta = delta_x_k[:len(non_swing_bus_indexes)].toarray().flatten()
        for idx, bus_idx in enumerate(non_swing_bus_indexes.values()):
            theta_variables[bus_idx, 0] += delta_theta[idx]

        # Update V variables (PQ buses only)
        delta_v = delta_x_k[len(non_swing_bus_indexes):].toarray().flatten()
        pq_indices = list(pq_bus_indexes.values())  # Indices of PQ buses
        for idx, pq_idx in enumerate(pq_indices):
            v_variables[pq_idx, 0] += delta_v[idx]  # Update only PQ buses

        # Convert back to sparse format
        theta_variables = csc_array(theta_variables)
        v_variables = csc_array(v_variables)

        return theta_variables, v_variables

    # --- Extract Variables ---
    swing_bus = {}
    pv_bus_indexes = {}
    pq_bus_indexes = {}
    non_swing_bus_indexes = {}
    all_bus_indexes = {}

    v_variables = []
    theta_variables = []
    active_power_injections_pu = []
    reactive_power_injections_pu = []

    p_equation_count = 0
    q_equation_count = 0

    variable_index = 0
    for bus_name, bus_info in data["bus_data"].items():
        bus_type = int(bus_info["load_flow_type"])
        v_variables.append(
            float(bus_info.get("voltage_magnitude", 1)) if bus_type != 3 else 1.06
        )
        theta_variables.append(0 if bus_type != 3 else 0)

        if bus_type == 3:  # Swing Bus
            swing_bus[bus_name] = variable_index
        elif bus_type == 2:  # PV Bus
            pv_bus_indexes[bus_name] = variable_index
            non_swing_bus_indexes[bus_name] = variable_index
            p_equation_count += 1
        elif bus_type in [0, 1]:  # PQ Bus
            pq_bus_indexes[bus_name] = variable_index
            non_swing_bus_indexes[bus_name] = variable_index
            p_equation_count += 1
            q_equation_count += 1

        all_bus_indexes[bus_name] = variable_index
        active_power_injections_pu.append(
            (float(bus_info["active_generation_MW"]) - float(bus_info["active_load_MW"])) / s_base
        )
        reactive_power_injections_pu.append(
            (float(bus_info["reactive_generation_MVAR"]) - float(bus_info["reactive_load_MVAR"])) / s_base
        )
        variable_index += 1

    # Convert Data to Sparse Arrays
    active_power_injections_pu = csc_array(np.array(active_power_injections_pu).reshape(-1, 1))
    reactive_power_injections_pu = csc_array(np.array(reactive_power_injections_pu).reshape(-1, 1))
    v_variables = csc_array(np.array(v_variables, dtype=float).reshape(-1, 1))
    theta_variables = csc_array(np.array(theta_variables, dtype=float).reshape(-1, 1))

    theta_variable_count = len(non_swing_bus_indexes)
    v_variable_count = len(pq_bus_indexes)

    # --- NR Iteration ---
    max_iterations = 100
    tolerance = 0.0001

    pi_x_minus_pi, qi_x_minus_qi = delta_F_x()
    F_delta_x = vstack([pi_x_minus_pi, qi_x_minus_qi])

    for iteration in range(1, max_iterations + 1):
        print(f"Iteration Number: {iteration}")
        jacobian = compute_jacobian(
            p_equation_count, q_equation_count, theta_variable_count, v_variable_count,
            g_bus, b_bus, v_variables, theta_variables, pq_bus_indexes, non_swing_bus_indexes
        )

        delta_x_k = spsolve(jacobian.tocsc(), (-F_delta_x).toarray().flatten())
        delta_x_k = csc_array(delta_x_k.reshape(-1, 1))

        # Apply updates to variables
        theta_variables, v_variables = update_variables(delta_x_k, theta_variables, v_variables)

        # Recompute F(x)
        pi_x_minus_pi, qi_x_minus_qi = delta_F_x()
        F_delta_x = vstack([pi_x_minus_pi, qi_x_minus_qi])

        # Check Convergence
        if np.all(np.abs(F_delta_x.data) < tolerance):
            print(f"{filename}: System Converged After {iteration} Iterations")
            break
        elif iteration == max_iterations:
            print(f"{filename}: System Did Not Converge After " + str(iteration) + " Iterations.")

    # --- Calculate Losses ---
    total_p_loss = sum(
        calculate_power(i, v_variables, theta_variables, g_bus, b_bus)
        for i in range(len(all_bus_indexes))
    )
    total_q_loss = sum(
        calculate_power(i, v_variables, theta_variables, g_bus, b_bus, is_reactive=True)
        for i in range(len(all_bus_indexes))
    )
    p_loss = total_p_loss * s_base
    q_loss = total_q_loss * s_base

    # Convert to Degrees
    voltage_angles = np.degrees(theta_variables.toarray().flatten())
    voltage_magnitudes = v_variables.toarray().flatten()

    # Final Output
    return dict(zip(all_bus_indexes.keys(), voltage_magnitudes)), dict(
        zip(all_bus_indexes.keys(), voltage_angles)
    ), p_loss, q_loss



a,b,c,d = load_flow("ieee14bus.json")
print(a)
print(b)
print(c)
print(d)
