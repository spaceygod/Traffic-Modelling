import numpy as np
import copy
import csv
from tqdm import tqdm

# Add neighboring nodes to each node
def add_properties_to_nodes(nodes, edges):
    for node, node_properties in nodes.items():
        neighboring_nodes = []
        for edge, edge_properties in edges.items():
            if edge.split(" → ")[0] == node:
                neighboring_nodes.append(edge.split(" → ")[1])
        nodes[node]["neighboring nodes"] = neighboring_nodes

# Adding the tt_0, capacity, number of cars on the edge at each time, and (expected) travel time at each time as properties of each edge
def add_properties_to_edges(edges, l_car, d_spacing, num_minutes):
    for edge, properties in edges.items():
        length = properties["length"]
        speed_limit = properties["speed_limit"]
        tt_0 = (length / speed_limit) * (60 / 1000)  # minutes
        properties["tt_0"] = tt_0
        properties["capacity"] = int(properties["lanes"] * properties["length"] / (l_car + d_spacing))
        properties["cars on edge"] = [0 for minute in range(num_minutes + 100000)] # list of the amount of cars occupying the edge at each time
        properties["travel time"] = [tt_0 for minute in range(num_minutes + 100000)] # list of the (expected) travel time on the edge at each time

# Change the capacity of each edge if the total number of cars in the simulation is reduced
def change_capacity(edges, reduction_factor):
    for edge, properties in edges.items():
        properties["capacity"] = int(reduction_factor * properties["capacity"])

## Functions that print simulation data
def print_nodes(nodes):
    print("=" * 40)
    print("Nodes")
    print("=" * 40)
    for node, properties in nodes.items():
        print(f"{'Node:':<20} {node}")
        print(f"{'Coordinates:':<20} {properties['coordinates']}")
        print(f"{'Population:':<20} {properties['population']}")
        print(f"{'Neighboring nodes':<20} {properties['neighboring nodes']}")
        print("-" * 40)

def print_edges(edges):
    print("\n" + "=" * 40)
    print("Edges (Road Network Details)")
    print("=" * 40)
    for edge, properties in edges.items():
        print(f"{'Edge:':<20} {edge}")
        print(f"{'Length (m):':<20} {properties['length']}")
        print(f"{'Speed Limit (km/h):':<20} {properties['speed_limit']}")
        print(f"{'Lanes:':<20} {properties['lanes']}")
        print(f"{'tt_0:':<20} {properties['tt_0']}")
        print(f"{'Capacity:':<20} {properties['capacity']}")
        print(f"{'Cars on edge:':<20} {properties['cars on edge']}")
        print(f"{'Travel time:':<20} {properties['travel time']}")
        print("-" * 40)

def print_distance_matrix(distance_matrix):
    print("\n" + "=" * 40)
    print("Distance Matrix (Euclidean Distance)")
    print("=" * 40)
    for node_pair, distance in distance_matrix.items():
        print(f"{node_pair:<20} {distance:.2f} units")
    print("-" * 40)

def print_travel_matrix(travel_matrix):
    print("\n" + "=" * 40)
    print("Travel Matrix (Fraction of Cars Traveling)")
    print("=" * 40)
    for route, fraction in travel_matrix.items():
        print(f"{route:<20} {fraction:.6f}")
    print("-" * 40)

def print_cars_spawned_each_minute(cars_spawned_each_minute):
    print("\n" + "=" * 40)
    print("Cars Spawned Each Minute")
    print("=" * 40)
    for route, cars_over_time in cars_spawned_each_minute.items():
        print(f"Route: {route}")
        print(f"{'Minute':<8} {'Cars Spawned'}")
        print("-" * 20)
        for minute, count in enumerate(cars_over_time):
            print(f"{minute:<8} {count}")
        print("-" * 40)

def print_cars(cars, range):
    range_of_cars = [cars[i] for i in range]
    for car in range_of_cars:
        print("=" * 40)  
        print(f"Car ID: {car['id']}")
        print("-" * 40)
        print(f"{'Origin:':<25} {car['origin']}")
        print(f"{'Destination:':<25} {car['destination']}")
        print(f"{'Optimal Path:':<25} {car['optimal path']}")
        print(f"{'Optimal Travel Time:':<25} {car['optimal travel time']}")
        print(f"{'Trajectory:':<25} {car['trajectory']}")
        print(f"{'Time Spawned:':<25} {car['time spawned']}")
        print(f"{'Time Arrived:':<25} {car['time arrived']}")
        print(f"{'Active:':<25} {car['active']}")
        print(f"{'Location:':<25} {car['location']}")
        print(f"{'Time Entered Last Edge:':<25} {car['time entered last edge']}")
        print(f"{'Finished Edge:':<25} {car['finished edge']}")
        print(f"{'Next Edge:':<25} {car['next edge']}")
        print("=" * 40 + "\n")  


# Using A*-algorithm to determine the optimal path for a car
## One iteration of A*
def iterate_A_star(current_route, current_queue, nodes, edges, time, heuristic_constant, distance_matrix, destination):
    current_path = current_route["path"]
    current_travel_time = current_route["travel time"]
    last_node_of_path = current_path[-1]
    updated_queue = current_queue

    # # Printing the iteration
    # print("=" * 40)
    # print(f"Iterating A* on {last_node_of_path}")
    # print(f"{'Path:':<25} {current_path}")
    # print(f"{'Travel time:':<25} {current_travel_time}")

    # For each neighbor of the last node of the route, make a new route which is the old route plus this neighbor
    neighboring_nodes = nodes[last_node_of_path]["neighboring nodes"] # list of the neighboring nodes of the last node of the route that is currently checked

    for neighbor in neighboring_nodes:
        new_path = current_path + [neighbor]
        next_edge = last_node_of_path + " → " + neighbor

        # Increase the cost due to travel time whenever the edge is full (so that the car would have to wait)
        if edges[next_edge]['cars on edge'][time] >= edges[next_edge]['capacity'] - 2:
            new_travel_time = current_travel_time + edges[next_edge]['travel time'][time] + 1000
        else:
            new_travel_time = current_travel_time + edges[next_edge]["travel time"][time]
        
        new_heuristic = heuristic_constant * distance_matrix[neighbor + " → " + destination]
        new_total_cost = new_travel_time + new_heuristic

        new_route = {
            "path": new_path, 
            "travel time": new_travel_time, 
            "heuristic": new_heuristic, 
            "total cost": new_total_cost
            }

        # # Printing the new route found
        # print("-" * 40)
        # print(f"Extended route from {new_path[-2]} to its neighbor {new_path[-1]}")
        # print(f"{'New path:':<25} {new_route['path']}")
        # print(f"{'New travel time:':<25} {new_route['travel time']}")
        # print(f"{'New heuristic:':<25} {new_route['heuristic']}")
        # print(f"{'New total cost:':<25} {new_route['total cost']}")

        updated_queue.append(new_route)
    
    # Remove the current route from the queue
    updated_queue.remove(current_route)

    # Sort the queue by total cost (lowest cost first)
    updated_queue = sorted(updated_queue, key=lambda x: x["total cost"])

    # Remove routes with higher cost going to the same neighbor
    unique_queue = []
    seen_nodes = set()  # To track nodes that we've already added the lowest cost route for

    for route in updated_queue:
        last_node = route["path"][-1]
        if last_node not in seen_nodes:
            unique_queue.append(route)  # Keep only the lowest-cost route to each node
            seen_nodes.add(last_node)

    # print("-" * 40)
    # print("-" * 40)
    # print(f"New queue")
    # for route in unique_queue:
    #     print(f"{'Path:':<25} {route['path']}")
    #     print(f"{'Travel time:':<25} {route['travel time']}")
    #     print(f"{'Heuristic:':<25} {route['heuristic']}")
    #     print(f"{'Total cost:':<25} {route['total cost']} \n")
    # print("=" * 40)

    return unique_queue

## Iterating A* until the destination is reached
def determine_optimal_route(car, nodes, edges, time, heuristic_constant, distance_matrix):
    origin = car["origin"]
    destination = car["destination"]
    queue = []  # List of routes checked by the algorithm. Each route is a dictionary with keys:
                # "path" (list of nodes traversed), "travel time", "heuristic" (heuristic estimate),
                # and "total cost" (sum of travel time and heuristic)

    # Initialize the starting route from the origin
    queue.append({
        "path": [origin], 
        "travel time": 0, 
        "heuristic": heuristic_constant * distance_matrix.get(origin + " → " + destination, float('inf')), 
        "total cost": heuristic_constant * distance_matrix.get(origin + " → " + destination, float('inf'))
    })

    # Run A* algorithm until queue is empty or destination is reached
    while queue:
        # Check if the current route reaches the destination
        if queue[0]["path"][-1] == destination:
            optimal_path = queue[0]["path"]
            optimal_travel_time = queue[0]["travel time"]
            return optimal_path, optimal_travel_time

        # Iterate through the A* algorithm on the current route
        queue = iterate_A_star(queue[0], queue, nodes, edges, time, heuristic_constant, distance_matrix, destination)

    # If the queue is empty, no path to the destination was found
    return None, None

# Find the node after a certain node in the trajectory
def find_next_node(node_list, target_node):
    for i in range(len(node_list) - 1):
        if node_list[i][0] == target_node:
            return node_list[i + 1][0]
    return None  # Returns None if target_node is not found or is the last node

# Converting the node database of the form {node: {coordinates: ..., population}} to {node: coordinates}
def convert_nodes(nodes):
    return {node: nodes[node]["coordinates"] for node in nodes} 

# Switch x and y coordinates of locations in nodes
def switch_x_y(nodes):
    new_nodes = copy.deepcopy(nodes)

    for node, properties in new_nodes.items():
        x = properties['coordinates'][0]
        y = properties['coordinates'][1]

        new_nodes[node]['coordinates'] = (y, x)

    return new_nodes
# Saving the simulation results to a CSV file
def save_simulation_results(cars, nodes, edges, distance_matrix, heuristic_constant, filename="simulation_results.csv"):
    # Prepare data for each car
    car_data = []

    # Optimal path and travel time if the system were empty
    # Temporarily set all edges' travel times to their minimum (tt_0) values
    empty_edges = {edge: {**properties, "travel time": [properties["tt_0"]] * len(properties["travel time"])}
                       for edge, properties in edges.items()}
    
    # Cache for storing computed optimal paths and travel times for each unique (origin, destination)
    optimal_path_cache = {}

    for car in tqdm(cars, desc="Saving to CSV"):
        # Path the car actually took
        actual_path = car["optimal path"] if car["optimal path"] is not None else car["trajectory"]
        time_taken = car["time arrived"] - car["time spawned"] if car["time arrived"] is not None else None
        
        # Check if the optimal path for the car's (origin, destination) pair is already cached
        origin, destination = car["origin"], car["destination"]
        if (origin, destination) in optimal_path_cache:
            # Retrieve from cache
            optimal_path_empty, optimal_travel_time_empty = optimal_path_cache[(origin, destination)]
        else:
            # Compute the optimal path and travel time if the system were empty
            optimal_path_empty, optimal_travel_time_empty = determine_optimal_route(
                car, nodes, empty_edges, car["time spawned"], heuristic_constant, distance_matrix
            )
            
            # Cache the result for this (origin, destination) pair
            optimal_path_cache[(origin, destination)] = (optimal_path_empty, optimal_travel_time_empty)
        
        # Add the car data to the list
        car_data.append({
            "Car ID": car["id"],
            "Chosen Path": actual_path if car["optimal path"] is not None else [actual_path[0] for actual_path in actual_path],
            "Time Taken": time_taken,
            "Optimal Path (Empty System)": optimal_path_empty,
            "Optimal Time (Empty System)": optimal_travel_time_empty
        })
    
    # Save the data to a CSV file
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Car ID", "Chosen Path", "Time Taken", "Optimal Path (Empty System)", "Optimal Time (Empty System)"])
        writer.writeheader()
        for car in car_data:
            writer.writerow(car)

# Change node population
def change_population(nodes):
    new_nodes = copy.deepcopy(nodes)

    for node, properties in new_nodes.items():
        if properties['population'] != None:
            if properties['population'] <= 200000:
                new_nodes[node]['population'] = None
            else:
                new_nodes[node]['population'] = 1

    return new_nodes