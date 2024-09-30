from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.functional import choose_next_edge, travel_time_bpr, reset_car_states
from utils.visualization import initialize_plot, update_plot

# Simulation and visualization combined
def simulate_and_visualize(cars, edges, node_positions, num_minutes, warmup_steps=120, most_congested_edge=None, track_most_congested=True, capacity_multiplier=1.0, animate=False, alpha=0.15, beta=4, sigma=2):
    fig, ax, edge_texts, timestep_text = initialize_plot(edges, node_positions)

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
        update_plot(t, edges, vehicle_counts, edge_texts, timestep_text, num_minutes)

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

# Run the function multiple times with different capacity values
def simulate_and_compare(cars, edges, node_positions, num_minutes, warmup_steps, deltas, alpha=0.15, beta=4, sigma=2):
    # Ask the user if it wants the traffic simulation to be animated
    animate = input("Do you want to animate the traffic simulation in real-time? You will have to close the plots for the code to continue if a full simulation is done. (y/n): ").lower() == "y"

    # Initial run to find the most congested edge
    print(f"Running simulation with capacity multiplier: 1.0")

    car_reach_times, avg_congestion = simulate_and_visualize(cars.copy(), edges.copy(), node_positions, num_minutes, warmup_steps=warmup_steps, animate=animate)
    most_congested_edge = max(avg_congestion, key=avg_congestion.get)
    print(f"Most congested edge: {most_congested_edge}")

    all_car_reach_times = [(1.0, car_reach_times)]

    # Run the simulation multiple times with different capacity multipliers
    for delta in deltas:
        # Reset vehicle counts and car states before running the simulation
        reset_car_states(cars)

        print(f"Running simulation with capacity multiplier: {delta}")
        car_reach_times, avg_congestion = simulate_and_visualize(cars.copy(), edges.copy(), node_positions, num_minutes, most_congested_edge=most_congested_edge, capacity_multiplier=delta, warmup_steps=warmup_steps, animate=animate, alpha=alpha, beta=beta, sigma=sigma)
        print(f"Most congested edge: {max(avg_congestion, key=avg_congestion.get)}")
        all_car_reach_times.append((delta, car_reach_times))
        print(f"Simulation with capacity multiplier {delta} completed.")

    return all_car_reach_times, most_congested_edge
