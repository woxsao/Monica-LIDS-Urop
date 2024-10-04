import pybamm
import csv

# Set up the DFN model with degradation mechanisms
num_cycles = 1000  # We defined 10 cycles in the experiment
model = pybamm.lithium_ion.DFN(
    {
        "SEI": "ec reaction limited",    }
)

# Load the parameter set
param = pybamm.ParameterValues("OKane2022")

# Define a constant current discharge experiment
experiment = pybamm.Experiment([
    (f"Discharge at 1C until 3V",
     "Rest for 1 hour",
    f"Charge at 1C until 4.2V",
    f"Hold at 4.2V until C/50")
] * num_cycles,
)

# Create a PyBaMM simulation
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

# Run the experiment
sol = sim.solve()
pybamm.plot_summary_variables(sol)

# Extract all the variables from the solution
variables_dict = sol.summary_variables
print("variables dict")
print(variables_dict)
time_data = sol["Time [s]"].entries


# Prepare data for writing to CSV
headers = list(variables_dict.keys())

# Split the data into cycles based on the time array

cycle_data = []

for cycle in range(num_cycles):
    # Extract time indices for the current cycle

    # Find the indices for the current cycle
    # Extract variables for each time step within the cycle
    row = {}
    for var in headers:
        # print("var", var)
        # print("variables_dict[var]", variables_dict[var])
        row[var] = variables_dict[var][cycle]
    cycle_data.append(row)
# print("CYcle data length")
# print(len(cycle_data))
# Write cycle data to CSV
csv_file = "cycle_variables_data_5.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(cycle_data)

print(f"All variable data for cycles saved to {csv_file}")
sim.plot()