import json
import numpy as np
from scipy import sparse

def create_bus_admittance_matrix(filename):
    with open(filename) as f:
        data = json.load(f)

    #init
    y_bus_size = len(data["bus_data"])
    bus_name_to_index = {bus_name: index for index, bus_name in enumerate(data["bus_data"])}
    y_bus = np.zeros((y_bus_size, y_bus_size), dtype=complex)

    # add line elements
    for line_info in data["branch_data"].values():
        from_index = bus_name_to_index[line_info["from_bus"]]
        to_index = bus_name_to_index[line_info["to_bus"]]

        R_pu = float(line_info["R_pu"])
        X_pu = float(line_info["X_pu"])
        B_pu = float(line_info["B_pu"])
        series_admittance = 1 / complex(R_pu, X_pu)
        shunt_admittance = 1j * B_pu /2
        line_type = line_info["type"]

        if line_type == "0":  # Transmission Line
            y_ii = series_admittance + shunt_admittance
            y_ij = y_ji = -series_admittance
            y_jj = series_admittance + shunt_admittance
        elif line_type == "1":  # Standard Transformer
            y_ii = series_admittance
            y_ij = y_ji = -series_admittance
            y_jj = series_admittance
        elif line_type == "2" or line_type == "3":  # Tap Changer Transformer
            tap = float(line_info["tap"])
            tap_inv = 1 / tap
            y_ii = series_admittance / (tap ** 2)
            y_ij = y_ji = -series_admittance * tap_inv
            y_jj = series_admittance
        elif line_type == "4":  # Phase Shifter Transformer
            phase_shift_rad = np.deg2rad(float(line_info["phase_shift_degree"]))
            phase_shift_complex = np.exp(1j * phase_shift_rad)
            y_ii = series_admittance
            y_ij = -series_admittance / np.conj(phase_shift_complex)
            y_ji = -series_admittance / phase_shift_complex
            y_jj = series_admittance

        y_bus[from_index, from_index] += y_ii
        y_bus[from_index, to_index] += y_ij
        y_bus[to_index, from_index] += y_ji
        y_bus[to_index, to_index] += y_jj

    # add bus shunt elements
    for bus_name, bus_info in data["bus_data"].items():
        bus_index = bus_name_to_index[bus_name]
        G_pu = float(bus_info["G_pu"])
        B_pu = float(bus_info["B_pu"])

        if G_pu != 0 or B_pu != 0:
            y_bus[bus_index, bus_index] += complex(G_pu, B_pu)

    return sparse.lil_matrix(y_bus)