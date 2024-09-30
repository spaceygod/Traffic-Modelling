import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from tqdm import tqdm

# Parameters for BPR function
alpha = 0.15
beta = 4
sigma = 2
l_car = 4.5 # Length of a car in meters
d_spacing = 55 # Minimum safe spacing between cars in meters

# Define the road network (length in meters, speed limit in km/h, number of lanes)
edges = {
    "City 1 → A": {"length": 30000, "speed_limit": 100, "lanes": 2},
    "City 1 → B": {"length": 50000, "speed_limit": 100, "lanes": 2},
    "A → C": {"length": 40000, "speed_limit": 100, "lanes": 2},
    "B → C": {"length": 60000, "speed_limit": 100, "lanes": 2},
    "C → D": {"length": 20000, "speed_limit": 100, "lanes": 2},
    "C → E": {"length": 30000, "speed_limit": 100, "lanes": 2},
    "C → City 2": {"length": 20000, "speed_limit": 100, "lanes": 1},
    "D → City 2": {"length": 10000, "speed_limit": 100, "lanes": 2},
    "E → City 2": {"length": 5000, "speed_limit": 100, "lanes": 2},
}

# Calculate the base travel time tt_0(e) for each edge (so we just add tt_0 and capacity to the edge dictionary as they only have to be calculated once)
for edge, properties in edges.items():
    length = properties["length"]
    speed_limit = properties["speed_limit"]
    tt_0 = (length / speed_limit) * (60 / 1000)  # minutes
    properties["tt_0"] = tt_0
    properties["capacity"] = int(properties["lanes"] * properties["length"] / (l_car + d_spacing))
    properties["current_capacity"] = int(properties["capacity"])

# Simulation settings
num_minutes = 240
warmup_steps = 120
car_distribution_mean = 85
car_distribution_std = 1.5

# Sample the number of cars arriving at City 1 each minute
np.random.seed(42)
cars_arriving_each_minute = np.random.normal(car_distribution_mean, car_distribution_std, num_minutes)
cars_arriving_each_minute = np.round(cars_arriving_each_minute).astype(int)

# Initialize data for cars
cars = []
car_id = 0
for minute, num_cars in enumerate(cars_arriving_each_minute):
    for _ in range(num_cars):
        car = {"id": car_id, "start_time": minute, "location": "", "arrived_at_node": 0, "left_at_node": 0, "was_on_route": ""}  # Initialize car data
        cars.append(car)
        car_id += 1

def indicator(x, y):
    return 1 if x == y else 0

def calculate_rt_i(arrival_next_node, arrived_at_last_node):
    return arrival_next_node - arrived_at_last_node

# Optimized Probability function using the defined formula
def compute_edge_probability(outgoing_edges, vehicle_counts, edges, cars):
    probability_numerators = []
    probability_denominator_sum = 0

    # Group cars by their current edge to avoid filtering multiple times
    cars_by_edge = {edge: [] for edge in outgoing_edges}
    for car in cars:
        if car["location"] in ["A", "B", "C", "D", "E"]:
            if car["was_on_route"] in outgoing_edges:
                cars_by_edge[car["was_on_route"]].append(car)
        elif car["location"] in outgoing_edges:
            if car["location"] in outgoing_edges:
                cars_by_edge[car["location"]].append(car)

    for edge in outgoing_edges:
        # Get travel time tt_0 for the edge
        tt_0 = edges[edge]["tt_0"]
        N_e = vehicle_counts[edge]

        if N_e != 0:
            # Vectorized calculation of rt_i for all cars on this edge
            arrival_next_node = np.array([car["arrived_at_node"] for car in cars_by_edge[edge]])
            arrived_times = np.array([car["left_at_node"] for car in cars_by_edge[edge]])
            rt_i_values = calculate_rt_i(arrival_next_node, arrived_times)

            probability_numerator_sum = np.sum(rt_i_values)

            if probability_numerator_sum != 0:
                probability_numerator = (1 / ((1 / N_e) * probability_numerator_sum))[-1]
            else:
                probability_numerator = 1 / tt_0
        else:
            probability_numerator = 1 / tt_0

        probability_numerators.append(probability_numerator)
        probability_denominator_sum += probability_numerator

    # Perform element-wise division
    normalized_probabilities = np.array(probability_numerators) / probability_denominator_sum
    
    return normalized_probabilities

# Function to choose the next edge based on the probabilities
def choose_next_edge(location, vehicle_counts, edges, cars):
    if location == "City 1":
        outgoing_edges = ["City 1 → A", "City 1 → B"]
    elif location == "A":
        outgoing_edges = ["A → C"]
    elif location == "B":
        outgoing_edges = ["B → C"]
    elif location == "C":
        outgoing_edges = ["C → D", "C → City 2", "C → E", ]
    elif location == "D":
        outgoing_edges = ["D → City 2"]
    elif location == "E":
        outgoing_edges = ["E → City 2"]
    
    # Check whether the capacity of the outgoing edges is reached
    outgoing_edges = [edge for edge in outgoing_edges if vehicle_counts[edge] < edges[edge]["current_capacity"]]

    # If all outgoing edges are at capacity, return False
    if not outgoing_edges:
        return False

    # Compute probabilities for each outgoing edge
    probabilities = compute_edge_probability(outgoing_edges, vehicle_counts, edges, cars)
    
    # Choose an edge based on the computed probabilities
    next_edge = np.random.choice(outgoing_edges, p=probabilities)
    return next_edge

def travel_time_bpr(tt_0, N_e, C_e, alpha, beta, sigma):
    return tt_0 * (1 + alpha * (N_e / C_e) ** beta) + np.random.normal(0, sigma**2)

# Node positions for plotting
node_positions = {
    "City 1": (0, 0),
    "A": (2, 2),
    "B": (2, -2),
    "C": (4, 0),
    "D": (6, 2),
    "E": (6, -2),
    "City 2": (8, 0)
}

# Plotting function to visualize nodes, edges, and car counts
def initialize_plot():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-3, 3)
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_facecolor('black')         # Black background for the axes
    ax.set_title("Real-Time Simulation of Car Movements", color='white')  # Title in white

    # Plot nodes (cities)
    for node, pos in node_positions.items():
        ax.plot(pos[0], pos[1], 'o', markersize=20, label=node)
        ax.text(pos[0], pos[1], node, fontsize=12, ha='center', va='center', color='white')

    # Store edge and node text for dynamic updating
    edge_texts = {}
    for edge in edges:
        start_node, end_node = edge.split(" → ")
        start_pos, end_pos = node_positions[start_node], node_positions[end_node]
        mid_pos = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='gray', linestyle='--')
        edge_texts[edge] = ax.text(mid_pos[0], mid_pos[1], '', fontsize=10, color='gray')

    node_texts = {node: ax.text(pos[0], pos[1] - 0.3, '', fontsize=10, color="white") for node, pos in node_positions.items()}

    # Add a text annotation to display the current timestep
    timestep_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='blue')

    return fig, ax, edge_texts, node_texts, timestep_text

# Update function for the animation
def update_plot(t, cars, vehicle_counts, edge_texts, node_texts, timestep_text, num_minutes):
    # Update edge texts with the number of cars and capacity
    for edge, text in edge_texts.items():
        num_cars = int(vehicle_counts[edge][0])
        capacity = edges[edge]["current_capacity"]
        text.set_text(f"{num_cars}/{capacity}")

    # Update the timestep text
    timestep_text.set_text(f"Timestep: {t}/{num_minutes}")

# Simulation and visualization combined
def simulate_and_visualize(cars, edges, num_minutes, warmup_steps=warmup_steps, most_congested_edge=None, track_most_congested=True, capacity_multiplier=1.0, animate=False):
    fig, ax, edge_texts, node_texts, timestep_text = initialize_plot()

    # Initialize vehicle counts on each edge
    vehicle_counts = {edge: np.zeros(1, dtype=int) for edge in edges}

    # Track travel times to City 2
    car_reach_times = []

    # Track the congestion of each edge
    if track_most_congested:
        congestion_data = {edge: [] for edge in edges}

    # Track the congestion of each edge
    if most_congested_edge is not None:
        edges[most_congested_edge]["current_capacity"] = int(edges[most_congested_edge]["capacity"] * capacity_multiplier)

    # Function to update the plot at each timestep
    def update(t):
        for car in cars:
            if car["location"] == "City 2":
                cars.remove(car)
                continue
            # Check if the car should start yet
            if car["start_time"] == t:
                car["location"] = "City 1"
            # Skip cars that haven't started yet or have already finished
            elif t < car["start_time"]:
                continue

            # Check if the car has finished its route
            if t-car["start_time"] >= car["arrived_at_node"] and not car["location"] in ["City 1", "A", "B", "C", "D", "E"]:
                # Store the edge the car was on before reaching a node
                car["was_on_route"] = car["location"]
                # Update the car's location
                car["location"] = car["location"].split(" → ")[-1]
                # Check if the reached location is City 2
                if car["location"] == "City 2":
                    # Remove the car from the route it just finished if it reached City 2
                    vehicle_counts[car["was_on_route"]] -= 1
                    if car["start_time"] > warmup_steps:
                        car_reach_times.append(car["arrived_at_node"])

            # Choose the next edge if the car is at a node
            if car["location"] in ["City 1", "A", "B", "C", "D", "E"]:
                next_edge = choose_next_edge(car["location"], vehicle_counts, edges, cars)

                if next_edge is False:
                    continue

                # Calculate travel time for this edge
                tt_0 = edges[next_edge]["tt_0"]
                N_e = vehicle_counts[next_edge]
                C_e = edges[next_edge]["current_capacity"]
                travel_time = travel_time_bpr(tt_0, N_e, C_e, alpha, beta, sigma)

                # Update the car's left- and arrived_at_node and assign a new route
                car["left_at_node"] = t - car["start_time"]
                car["arrived_at_node"] = car["left_at_node"] + round(travel_time[0])
                car["location"] = next_edge

                # Update vehicle count for the edge
                vehicle_counts[next_edge] += 1
                if car["was_on_route"]:
                    vehicle_counts[car["was_on_route"]] -= 1

            # Record congestion level
            if track_most_congested:
                # If warmup period is over, record congestion data
                if t > warmup_steps:
                    for edge in edges:
                        congestion_data[edge].append(vehicle_counts[edge][0] / edges[edge]["capacity"])

        # Update the plot with the current state
        update_plot(t, cars, vehicle_counts, edge_texts, node_texts, timestep_text, num_minutes)

    if animate:
        # Animate the plot over time
        anim = FuncAnimation(fig, update, frames=range(num_minutes), repeat=False, interval=100)
        plt.show()
    else:
        # Run the simulation without animation using tqdm for a progress bar
        for t in tqdm(range(num_minutes), desc=f"Simulating"):
            update(t)

    # Calculate average congestion per edge
    if track_most_congested:
        avg_congestion = {edge: np.mean(congestion_data[edge]) for edge in edges}
    else:
        avg_congestion = None

    return car_reach_times, avg_congestion

def reset_car_states(cars):
    for car in cars:
        car["location"] = ""
        car["arrived_at_node"] = 0
        car["left_at_node"] = 0
        car["was_on_route"] = ""

# Run the function multiple times with different capacity values
def simulate_and_compare(cars, edges, num_minutes, deltas=[1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]):
    # Ask the user if it wants the traffic simulation to be animated
    animate = input("Do you want to animate the traffic simulation in real-time? You will have to close the plots for the code to continue if a full simulation is done. (y/n): ").lower() == "y"

    # Initial run to find the most congested edge
    print(f"Running simulation with capacity multiplier: 1.0")

    car_reach_times, avg_congestion = simulate_and_visualize(cars.copy(), edges.copy(), num_minutes, warmup_steps=warmup_steps, animate=animate)
    most_congested_edge = max(avg_congestion, key=avg_congestion.get)
    print(f"Most congested edge: {most_congested_edge}")

    all_car_reach_times = [(1.0, car_reach_times)]

    # Run the simulation multiple times with different capacity multipliers
    for delta in deltas:
        # Reset vehicle counts and car states before running the simulation
        reset_car_states(cars)

        print(f"Running simulation with capacity multiplier: {delta}")
        car_reach_times, avg_congestion = simulate_and_visualize(cars.copy(), edges.copy(), num_minutes, most_congested_edge=most_congested_edge, capacity_multiplier=delta, warmup_steps=warmup_steps, animate=animate)
        print(f"Most congested edge: {max(avg_congestion, key=avg_congestion.get)}")
        all_car_reach_times.append((delta, car_reach_times))
        print(f"Simulation with capacity multiplier {delta} completed.")

    return all_car_reach_times, most_congested_edge

# Plotting 3D Surface of Travel Times with interpolation for smoothness
def plot_3D_surface(all_car_reach_times, bins=25, grid_res=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare lists for surface plotting
    travel_times = []
    multipliers = []
    frequencies = []

    # For each delta, create the travel time distribution and store it
    for delta, car_reach_times in all_car_reach_times:
        hist, xedges = np.histogram(car_reach_times, bins=bins)
        xpos = (xedges[:-1] + xedges[1:]) / 2  # Midpoints of the bins (travel times)
        ypos = np.full_like(xpos, delta)  # Capacity multiplier (same for all travel times)
        travel_times.append(xpos)
        multipliers.append(ypos)
        frequencies.append(hist)

    # Convert lists to arrays
    travel_times = np.concatenate(travel_times)
    multipliers = np.concatenate(multipliers)
    frequencies = np.concatenate(frequencies)

    # Create a finer grid for interpolation
    travel_time_grid = np.linspace(min(travel_times), max(travel_times), grid_res)
    multiplier_grid = np.linspace(min(multipliers), max(multipliers), grid_res)
    grid_x, grid_y = np.meshgrid(travel_time_grid, multiplier_grid)

    # Interpolate the frequencies on the finer grid
    grid_z = griddata((travel_times, multipliers), frequencies, (grid_x, grid_y), method='cubic')

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # Add labels
    ax.set_xlabel("Travel Time (minutes)")
    ax.set_ylabel("Capacity Multiplier")
    ax.set_zlabel("Number of Cars")
    plt.title("Effect of Capacity Increase on Travel Times")

    # Add a color bar to indicate frequency
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

def plot_scatter_with_correlation(all_car_reach_times):
    # Prepare data for scatter plots
    deltas = []
    avg_reach_times = []
    cars_reached_city_2 = []

    for delta, car_reach_times in all_car_reach_times:
        deltas.append(delta)
        avg_reach_times.append(np.mean(car_reach_times))  # Average reach time for each delta
        cars_reached_city_2.append(len(car_reach_times))  # Number of cars that reached City 2 for each delta

    # Plot scatter plot for delta vs avg reach time
    plt.figure(figsize=(14, 6))

    # Scatter plot 1: Delta vs. Average Car Reach Time
    plt.subplot(1, 2, 1)
    plt.scatter(deltas, avg_reach_times, color='b')
    plt.title("Delta vs. Average Car Reach Time")
    plt.xlabel("Capacity Multiplier (Delta)")
    plt.ylabel("Average Car Reach Time (minutes)")
    
    # Compute correlation coefficient for delta vs avg reach time
    correlation_coef1, _ = pearsonr(deltas, avg_reach_times)
    plt.text(1.1, max(avg_reach_times) - 1, f"Correlation Coefficient: {correlation_coef1:.2f}", fontsize=12, color='blue')

    # Scatter plot 2: Delta vs. Number of Cars Reached City 2
    plt.subplot(1, 2, 2)
    plt.scatter(deltas, cars_reached_city_2, color='g')
    plt.title("Delta vs. Number of Cars Reached City 2")
    plt.xlabel("Capacity Multiplier (Delta)")
    plt.ylabel("Number of Cars Reached City 2")

    # Compute correlation coefficient for delta vs cars reached
    correlation_coef2, _ = pearsonr(deltas, cars_reached_city_2)
    plt.text(1.1, max(cars_reached_city_2) - 10, f"Correlation Coefficient: {correlation_coef2:.2f}", fontsize=12, color='green')

    plt.tight_layout()
    plt.show()

# Run the simulation and plot the results
all_car_reach_times, most_congested_edge = simulate_and_compare(cars, edges, num_minutes)
plot_3D_surface(all_car_reach_times)
plot_scatter_with_correlation(all_car_reach_times)