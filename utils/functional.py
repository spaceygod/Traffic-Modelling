import numpy as np

def indicator(x, y):
    return 1 if x == y else 0

def calculate_rt_i(arrival_next_node, arrived_at_last_node):
    return arrival_next_node - arrived_at_last_node

# Optimized Probability function using the defined formula
def compute_edge_probability(outgoing_edges, vehicle_counts, edges, cars, all_nodes, starting_city, ending_city):
    probability_numerators = []
    probability_denominator_sum = 0

    # Group cars by their current edge to avoid filtering multiple times
    cars_by_edge = {edge: [] for edge in outgoing_edges}
    for car in cars:
        if car["location"] in [node for node in all_nodes if node not in [starting_city, ending_city]]:
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
def choose_next_edge(location, vehicle_counts, edges, cars, all_nodes, starting_city, ending_city):
    # Find all outgoing edges from the current location
    outgoing_edges = [edge for edge in edges if edge.startswith(f"{location} →")]
    
    # Check whether the capacity of the outgoing edges is reached
    outgoing_edges = [edge for edge in outgoing_edges if vehicle_counts[edge] < edges[edge]["current_capacity"]]

    # If all outgoing edges are at capacity, return False
    if not outgoing_edges:
        return False

    # Compute probabilities for each outgoing edge
    probabilities = compute_edge_probability(outgoing_edges, vehicle_counts, edges, cars, all_nodes, starting_city, ending_city)
    
    # Choose an edge based on the computed probabilities
    next_edge = np.random.choice(outgoing_edges, p=probabilities)
    return next_edge

def travel_time_bpr(tt_0, N_e, C_e, alpha, beta, sigma):
    return max(0, tt_0 * (1 + alpha * (N_e / C_e) ** beta) + np.random.normal(0, sigma**2)) # The max is to avoid negative travel times

# Calculate the base travel time tt_0(e) for each edge (so we just add tt_0 and capacity to the edge dictionary as they only have to be calculated once) and the capacity of each edge
def add_properties_to_edges(edges, l_car, d_spacing):
    for edge, properties in edges.items():
        length = properties["length"]
        speed_limit = properties["speed_limit"]
        tt_0 = (length / speed_limit) * (60 / 1000)  # minutes
        properties["tt_0"] = tt_0
        properties["capacity"] = int(properties["lanes"] * properties["length"] / (l_car + d_spacing))
        properties["current_capacity"] = int(properties["capacity"])
    return edges

def reset_car_states(cars):
    for car in cars:
        car["location"] = ""
        car["arrived_at_node"] = 0
        car["left_at_node"] = 0
        car["was_on_route"] = ""