import numpy as np
from utils.functional import travel_time_bpr

# Run modified A* algorithm
def run_A_mod(nodes, edges, car, heuristic_constant, distance_matrix, num_minutes):

    # Initialize the queue of checked routes
    queue = [{
        "trajectory": [(car["origin"], car["time spawned"])],
        "travel time": 0,
        "heuristic": heuristic_constant * distance_matrix.get(car["origin"] + " → " + car["destination"], float('inf')),
        "total cost": heuristic_constant * distance_matrix.get(car["origin"] + " → " + car["destination"], float('inf'))
    }]

    # Run modified A* algorithm until queue is empty or destination is reached
    while queue:

        # Check if the current route reaches the destination
        if queue[0]["trajectory"][-1][0] == car["destination"]:
            # If so, return this optimal trajectory
            optimal_trajectory = queue[0]["trajectory"]
            return optimal_trajectory
        else:
            # If not,iterate the modified A* algorithm
            queue = iterate_A_mod(queue, nodes, edges, car, heuristic_constant, distance_matrix, num_minutes)

    # If the queue is empty, no path to the destination was found
    return None

# Iterate the modified A* algorithm
def iterate_A_mod(current_queue, nodes, edges, car, heuristic_constant, distance_matrix, num_minutes):

    best_route = current_queue[0] # the route with the lowest total cost so far
    travel_time = best_route["travel time"] # the travel time of this route
    last_node_of_path = best_route["trajectory"][-1][0] # the last node of this route
    neighboring_nodes = nodes[last_node_of_path]["neighboring nodes"] # list of the neighboring nodes of the last node of this route

    # For each neighbor of the last node of the route, make a new route which is the old route plus this neighbor
    for neighbor in neighboring_nodes:

        new_travel_time = travel_time + edges[last_node_of_path + " → " + neighbor]["travel time"][round(car["time spawned"] + travel_time)] # the new travel time is the old travel time plus the travel time of the new edge at time t = (time the car is spawned) + (time it took the car to get to the last node of the route)
        new_trajectory = best_route["trajectory"] + [(neighbor, car["time spawned"] + new_travel_time)] # the new trajectory
        new_heuristic = heuristic_constant * distance_matrix.get(neighbor + " → " + car["destination"], float('inf')) # the heuristic is proportional to the distance from the neighbor to the destination
        new_total_cost = new_heuristic + new_travel_time # the total cost is the travel time plus the heuristic

        new_route = {
            "trajectory": new_trajectory,
            "travel time": new_travel_time,
            "heuristic": new_heuristic,
            "total cost": new_total_cost
        }

        # Add the new route to the queue
        current_queue.append(new_route)
    
    # Remove the route with the lowest total cost from the queue
    current_queue.pop(0)

    # Sort the queue by total cost
    current_queue.sort(key=lambda x: x["total cost"])

    # Remove routes with higher total cost going to the same node
    seen_nodes = set()
    for route in current_queue:
        if route["trajectory"][-1][0] in seen_nodes:
            current_queue.remove(route)
        else:
            seen_nodes.add(route["trajectory"][-1][0])

    return current_queue

# Update the number of cars on each edge in the future based on the trajectory
def update_future_edges(edges, trajectory, alpha, beta, sigma):
    
    # Iterate over all nodes in the trajectory
    for i in range(len(trajectory) - 1):
        edge = trajectory[i][0] + " → " + trajectory[i + 1][0] # the edge between the two nodes
        times_on_edge = list(range(round(trajectory[i][1]), int(trajectory[i + 1][1]))) # the times the car was on this edge

        # Update the number of cars on this edge for each time
        for time in times_on_edge:
            edges[edge]["cars on edge"][time] += 1
            edges[edge]["travel time"][time] = travel_time_bpr(edges[edge]["tt_0"], edges[edge]["cars on edge"][time], edges[edge]["capacity"], alpha, beta, sigma)
        