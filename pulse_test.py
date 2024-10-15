import pybamm
import csv
import numpy as np

# Set up the DFN model with degradation mechanisms
model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})

# Load the parameter set
param = pybamm.ParameterValues("OKane2022")

# Define the aging experiment
aging_experiment_template = [
    ("Discharge at 1C until 3V", "Rest for 1 hour", "Charge at 1C until 4.2V", "Hold at 4.2V until C/50")
] * 100  # Repeat this pattern for 100 cycles (or adjust as needed)

# Define the pulse test experiment with longer durations
pulse_experiment = pybamm.Experiment([
    ("Charge at 0.5A for 10 seconds", "Rest for 25 seconds"),
    ("Discharge at 0.5A for 10 seconds", "Rest for 25 seconds"),
    ("Charge at 1A for 10 seconds", "Rest for 25 seconds"),
    ("Discharge at 1A for 10 seconds", "Rest for 25 seconds")
], period="0.1 seconds")

# Prepare CSV file for saving results
csv_file = "pulse_test_data_per_cycle.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Cycle", "Time [s]", "Terminal voltage [V]"])

# Loop through each cycle and run both the aging and pulse test
num_cycles = 50

for cycle in range(1, num_cycles + 1):
    print(f"Running cycle {cycle}...")

    # Define the aging experiment for this cycle
    aging_experiment = pybamm.Experiment(aging_experiment_template)

    # Create a PyBaMM simulation for the aging experiment
    aging_sim = pybamm.Simulation(model, parameter_values=param, experiment=aging_experiment)

    # Run the aging simulation
    aging_solution = aging_sim.solve(calc_esoh=False)

    # Set the initial conditions for the pulse test from the aging solution
    pulse_model = model.set_initial_conditions_from(aging_solution, inplace=False)

    # Create a new simulation for the pulse test
    pulse_sim = pybamm.Simulation(pulse_model, parameter_values=param, experiment=pulse_experiment)

    # Increase time resolution by specifying the time points explicitly
    #pulse_sim.build_solver("CasADi")  # Specify the solver if necessary
    t_eval = np.linspace(0, 100, 200)  # Fine time resolution

    # Run the pulse test simulation
    pulse_solution = pulse_sim.solve(calc_esoh=False)

    # Extract data from the pulse test
    time_data = pulse_solution["Time [s]"].entries
    voltage_data = pulse_solution["Terminal voltage [V]"].entries

    # Write results to CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for t, v in zip(time_data, voltage_data):
            writer.writerow([cycle, t, v])
    model = pulse_model.set_initial_conditions_from(pulse_solution, inplace=False)
print(f"Pulse test data for all {num_cycles} cycles saved to {csv_file}")
