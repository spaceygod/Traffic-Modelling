import numpy as np

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
        properties["cars on edge"] = [0 for minute in range(num_minutes)] # list of the amount of cars occupying the edge at each time
        properties["travel time"] = [tt_0 for minute in range(num_minutes)] # list of the (expected) travel time on the edge at each time

# Using A*-algorithm to determine the optimal path for a car
## One iteration of A*
def iterate_A_star(current_route, current_queue, nodes, edges, time, heuristic_constant, distance_matrix, destination):
    current_path = current_route["path"]
    current_travel_time = current_route["travel time"]
    last_node_of_path = current_path[-1]
    updated_queue = current_queue

    # For each neighbor of the last node of the route, make a new route which is the old route plus this neighbor
    neighboring_nodes = nodes[last_node_of_path]["neighboring nodes"] # list of the neighboring nodes of the last node of the route that is currently checked

    for neighbor in neighboring_nodes:
        new_path = current_path + [neighbor]
        new_travel_time = current_travel_time + edges[last_node_of_path + " → " + neighbor]["travel time"][time]
        new_heuristic = heuristic_constant * distance_matrix[neighbor + " → " + destination]
        new_total_cost = new_travel_time + new_heuristic

        new_route = {
            "path": new_path, 
            "travel time": new_travel_time, 
            "heuristic": new_heuristic, 
            "total cost": new_total_cost
            }

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

    return unique_queue

## Iterating A* until the destination is reached
def determine_optimal_route(car, nodes, edges, time, heuristic_constant, distance_matrix):
    origin = car["origin"]
    destination = car["destination"]
    queue = [] # list of routes checked by the algorithm. A route is a dictionary {"path": path, "travel time": travel time, "heuristic": heuristic, "total cost": travel time + heuristic} where path is a list of the traversed nodes (e.g. [origin, A, B, ...]) and heuristic is the heuristic constant * the Euclidean distance from the last node of the path to the destination

    # Starting point of the algorithm
    queue.append({
        "path": [origin], 
        "travel time": 0, 
        "heuristic": heuristic_constant * distance_matrix[origin + " → " + destination], 
        "total cost": heuristic_constant * distance_matrix[origin + " → " + destination]
        })
    
    # Running the algorithm
    while queue[0]["path"][-1] != destination:
        queue = iterate_A_star(queue[0], queue, nodes, edges, time, heuristic_constant, distance_matrix)
    
    # Returning the optimal route
    optimal_path = queue[0]["path"]
    optimal_travel_time = queue[0]["travel time"]

    return optimal_path, optimal_travel_time
