import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load model results
results = pd.read_csv("model_comparison_results.csv")

print("Model results loaded successfully!")
print(results)

# ---------------------------------------------------------
# LOGISTICS OPTIMIZATION PROBLEM
# ---------------------------------------------------------
# We have 3 drivers and 4 delivery routes.
# Goal: minimize total predicted delivery time.

drivers = ["Driver 1", "Driver 2", "Driver 3"]
routes = ["Route A", "Route B", "Route C", "Route D"]

# Predicted delivery time (hours)
delivery_time = np.array([
    [1.8, 2.2, 1.5, 2.5],
    [2.0, 1.7, 2.4, 2.1],
    [1.6, 2.3, 1.9, 2.0]
])

# Flatten matrix
cost = delivery_time.flatten()

# Each route must be assigned exactly once
A_eq = []
b_eq = []

for route in range(4):
    constraint = np.zeros(12)

    for driver in range(3):
        constraint[driver * 4 + route] = 1

    A_eq.append(constraint)
    b_eq.append(1)

# Each driver can handle at most 2 routes
A_ub = []
b_ub = []

for driver in range(3):
    constraint = np.zeros(12)

    for route in range(4):
        constraint[driver * 4 + route] = 1

    A_ub.append(constraint)
    b_ub.append(2)

# Binary assignment variables
bounds = [(0, 1)] * 12

# Solve optimization
result = linprog(
    cost,
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=bounds,
    method="highs"
)

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if result.success:

    print("\n===================================")
    print("OPTIMIZATION SUCCESSFUL")
    print("===================================")

    assignment = result.x.reshape(3, 4)

    total_time = 0

    for i, driver in enumerate(drivers):

        assigned_routes = []

        for j, route in enumerate(routes):

            if assignment[i, j] > 0.5:
                assigned_routes.append(route)
                total_time += delivery_time[i, j]

        print(f"{driver}: {assigned_routes}")

    print(f"\nMinimum Total Delivery Time: {total_time:.2f} hours")

    # Save assignment
    rows = []

    for i, driver in enumerate(drivers):

        for j, route in enumerate(routes):

            if assignment[i, j] > 0.5:

                rows.append({
                    "Driver": driver,
                    "Route": route,
                    "Predicted_Time_Hours": delivery_time[i, j]
                })

    assignment_df = pd.DataFrame(rows)

    assignment_df.to_csv(
        "optimized_driver_assignment.csv",
        index=False
    )

    print("\nOptimization result saved as:")
    print("optimized_driver_assignment.csv")

else:
    print("Optimization failed.")