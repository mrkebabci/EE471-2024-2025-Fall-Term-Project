import json
from math import (cos,
sin,
pi)

from scipy import sparse
import numpy as np

def create_bus_admittance_matrix(filename):
    with open(filename) as f:
        data = json.load(f)

    #init
    y_bus_size = len(data["bus_data"])
    bus_name_to_index = {bus_name: index for index, bus_name in enumerate(data["bus_data"])}
    y_bus = np.zeros((y_bus_size,y_bus_size), dtype=complex)


    for k in data["branch_data"]:

        line_info = data["branch_data"][str(k)]
        from_bus_name = line_info["from_bus"] #from bus: i
        to_bus_name = line_info["to_bus"] #to bus: j

        if (from_bus_name in bus_name_to_index) and (to_bus_name in bus_name_to_index): #Check if input names are valid
            line_series_admittance = 1 / complex(float(line_info["R_pu"]),float(line_info["X_pu"]))
            line_parallel_admittance =  complex(0,float(line_info["B_pu"])) / 2

            if  line_info["type"] == "0": #Transmission Line Case

                y_ii = line_series_admittance + line_parallel_admittance
                y_ij = y_ji = -line_series_admittance
                y_jj = y_ii

            elif line_info["type"] == "1":  #Standard Transformer

                y_ii = line_series_admittance
                y_ij = y_ji = -line_series_admittance
                y_jj = y_ii

            elif line_info["type"] == "2":  #Tap Changer Transformer

                tap = float(line_info["tap"])
                y_ii = line_series_admittance / tap ** 2
                y_ij = y_ji = -line_series_admittance / tap
                y_jj = line_series_admittance

            elif line_info["type"] == "3":  #Phase Shifter Transformer

                y_ii = 0 #kemal hocaya soru
                y_ij = 0
                y_ji = 0
                y_jj = 0

            elif line_info["type"] == "4":  # Phase Shifter Transformer

                phase_shift_radian = np.deg2rad(float(line_info["phase_shift_degree"]))
                phase_shift_complex = complex(np.cos(phase_shift_radian), np.sin(phase_shift_radian))
                y_ii = line_series_admittance
                y_ij = -line_series_admittance / np.conjugate(phase_shift_complex)
                y_ji = -line_series_admittance / phase_shift_complex
                y_jj = y_ii

        from_index = bus_name_to_index[from_bus_name]
        to_index = bus_name_to_index[to_bus_name]


        y_bus[from_index, from_index] += y_ii
        y_bus[from_index, to_index] += y_ij
        y_bus[to_index, from_index] += y_ji
        y_bus[to_index, to_index] += y_jj
    return sparse.lil_matrix(y_bus)

filename = "ieee14bus.json"
ans = create_bus_admittance_matrix(filename)
print(ans)
