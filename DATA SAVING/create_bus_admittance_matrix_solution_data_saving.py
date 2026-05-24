import json
from scipy.sparse import lil_matrix
from scipy.io import savemat
from numpy import deg2rad, cos, sin, conjugate, clongdouble

def create_bus_admittance_matrix(topology_file):
    # Insert your code here
    with open(topology_file, 'r') as f:
        data = json.load(f)
    f.close()
    del f

    busData = data["bus_data"]
    branchData = data["branch_data"]
    y_bus = lil_matrix((len(busData), len(busData)), dtype=clongdouble)

    count = 0
    busMap = {}
    for idx, bus in busData.items():
        i = int(idx)
        y_shunt = complex(float(bus["G_pu"]), float(bus["B_pu"]))
        busMap[i] = count
        y_bus[count, count] += y_shunt
        count += 1

    for idx, line in branchData.items():
        i = busMap[int(line["from_bus"])]
        j = busMap[int(line["to_bus"])]
        resistance = float(line["R_pu"])
        reactance = float(line["X_pu"])
        line_charging_susceptance = float(line["B_pu"])

        admittance = 1 / complex(resistance, reactance)  # Y=1/Z
        shunt_admittance = complex(0, line_charging_susceptance)  # B_total
        tap = float(line["tap"])
        phaseShift = deg2rad(float(line["phase_shift_degree"]))
        if tap == 0:
            tap = 1
        a = tap*(cos(phaseShift) + 1j*sin(phaseShift))

        y_bus[i, i] += admittance/(tap**2) + shunt_admittance / 2
        y_bus[i, j] -= admittance / conjugate(a)
        y_bus[j, i] -= admittance / a
        y_bus[j, j] += admittance + shunt_admittance / 2

    return y_bus

a = create_bus_admittance_matrix("ieee14bus.json")


bus14 = a.tocsr()

bus30 = create_bus_admittance_matrix("ieee30bus.json")
bus30 = bus30.tocsr()

bus57 = create_bus_admittance_matrix("ieee57bus.json")
bus57 = bus57.tocsr()

bus118 = create_bus_admittance_matrix("ieee118bus.json")
bus118 = bus118.tocsr()

bus300 = create_bus_admittance_matrix("ieee300bus.json")
bus300 = bus300.tocsr()


savemat("y_bus_14.mat", {"ieee14bus": bus14})
savemat("y_bus_30.mat", {"ieee30bus": bus30})
savemat("y_bus_57.mat", {"ieee57bus": bus57})
savemat("y_bus_118.mat", {"ieee118bus": bus118})
savemat("y_bus_300.mat", {"ieee300bus": bus300})
# print(a)