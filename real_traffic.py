import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import matplotlib.image as mpimg

# Parameters for BPR function
alpha = 0.15
beta = 4
sigma = 2
l_car = 4.5 # Length of a car in meters
d_spacing = 55 # Minimum safe spacing between cars in meters

edges = {
'Amsterdam → Hilversum': {'length': 22000, 'speed_limit': 100, 'lanes': 3},
'Hilversum → Amersfoort': {'length': 15000, 'speed_limit': 100, 'lanes': 3},
'Amersfoort → Apeldoorn': {'length': 42500, 'speed_limit': 100, 'lanes': 3},
'Apeldoorn → Deventer': {'length': 16500, 'speed_limit': 100, 'lanes': 3}, 
'Deventer → Hengelo': {'length': 60000, 'speed_limit': 100, 'lanes': 3}, 
'Hengelo → Duitsland': {'length': 18200, 'speed_limit': 100, 'lanes': 2}, 
'Amsterdam → Utrecht': {'length': 26600, 'speed_limit': 100, 'lanes': 5},
'Utrecht → Waardenburg': {'length': 26800, 'speed_limit': 100, 'lanes': 4},
"Waardenburg → 's-Hertogenbosch": {'length': 17200, 'speed_limit': 100, 'lanes': 4},
"'s-Hertogenbosch → Eindhoven": {'length': 37100, 'speed_limit': 100, 'lanes': 3},
'Eindhoven → Weert': {'length': 32400, 'speed_limit': 100, 'lanes': 2},
'Weert → Echt': {'length': 20600, 'speed_limit': 100, 'lanes': 2},
'Echt → Geleen': {'length': 19900, 'speed_limit': 100, 'lanes': 2},
'Geleen → Maastricht': {'length': 14200, 'speed_limit': 100, 'lanes': 2},
'Maastricht → België': {'length': 12700, 'speed_limit': 100, 'lanes': 2},
'Amsterdam → Schiphol': {'length': 7400, 'speed_limit': 100, 'lanes': 3},
'Schiphol → Leiden': {'length': 26700, 'speed_limit': 100, 'lanes': 4},
'Leiden → Den Haag': {'length': 13800, 'speed_limit': 100, 'lanes': 3},
'Den Haag → Delft': {'length': 3400, 'speed_limit': 100, 'lanes': 3},
'Delft → Schiedam': {'length': 6900, 'speed_limit': 100, 'lanes': 3},
'Schiedam → Vlaardingen': {'length': 1900, 'speed_limit': 100, 'lanes': 3},
'Vlaardingen → Knooppunt Benelux': {'length': 4400, 'speed_limit': 100, 'lanes': 2},
'Knooppunt Sabina → Bergen op Zoom': {'length': 23900, 'speed_limit': 100, 'lanes': 2},
'Bergen op Zoom → België': {'length': 15800, 'speed_limit': 100, 'lanes': 2},
'Hoofddorp → Westpoort': {'length': 13000, 'speed_limit': 100, 'lanes': 2},
'Westpoort → Amsterdam': {'length': 16200, 'speed_limit': 100, 'lanes': 2},
'Knooppunt Muiderberg → Almere': {'length': 11600, 'speed_limit': 100, 'lanes': 3},
'Almere → Lelystad': {'length': 21900, 'speed_limit': 100, 'lanes': 3},
'Lelystad → Emmeloord': {'length': 32200, 'speed_limit': 100, 'lanes': 2},
'Emmeloord → Lemmer': {'length': 19000, 'speed_limit': 100, 'lanes': 2},
'Lemmer → Joure': {'length': 14200, 'speed_limit': 100, 'lanes': 2},
'Zaanstad → Purmerend': {'length': 8400, 'speed_limit': 100, 'lanes': 3},
'Purmerend → Hoorn': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
'Hoorn → Afsluitdijk': {'length': 45800, 'speed_limit': 100, 'lanes': 2},
'Afsluitdijk → Sneek': {'length': 36900, 'speed_limit': 100, 'lanes': 2},
'Sneek → Joure': {'length': 11700, 'speed_limit': 100, 'lanes': 2},
'Joure → Heerenveen': {'length': 9400, 'speed_limit': 100, 'lanes': 2},
'Heerenveen → Drachten': {'length': 20600, 'speed_limit': 100, 'lanes': 2},
'Drachten → Groningen': {'length': 33900, 'speed_limit': 100, 'lanes': 2},
'Groningen → Hoogezand': {'length': 15000, 'speed_limit': 100, 'lanes': 2},
'Hoogezand → Winschoten': {'length': 21600, 'speed_limit': 100, 'lanes': 2},
'Winschoten → Bad Nieuweschan': {'length': 11400, 'speed_limit': 100, 'lanes': 2},
'Bad Nieuweschans → Duitsland': {'length': 1700, 'speed_limit': 100, 'lanes': 2},
'Amsterdam → Zaanstad': {'length': 4300, 'speed_limit': 100, 'lanes': 3},
'Diemen → Amstelveen': {'length': 14000, 'speed_limit': 100, 'lanes': 3},
'Amstelveen → Haarlem': {'length': 14900, 'speed_limit': 100, 'lanes': 3},
'Haarlem → Beverwijk': {'length': 12100, 'speed_limit': 100, 'lanes': 2},
'Duitsland → Zevenaar': {'length': 7200, 'speed_limit': 100, 'lanes': 2},
'Zevenaar → Arnhem': {'length': 8600, 'speed_limit': 100, 'lanes': 3},
'Arnhem → Ede': {'length': 24900, 'speed_limit': 100, 'lanes': 3},
'Ede → Utrecht': {'length': 38400, 'speed_limit': 100, 'lanes': 2},
'Utrecht → Gouda': {'length': 31000, 'speed_limit': 100, 'lanes': 3},
'Gouda → Zoetermeer': {'length': 13400, 'speed_limit': 100, 'lanes': 3},
'Zoetermeer → Den Haag': {'length': 12700, 'speed_limit': 100, 'lanes': 3},
'Delft → Rotterdam': {'length': 10600, 'speed_limit': 100, 'lanes': 3},
'Maasvlakte → Europoort': {'length': 3600, 'speed_limit': 100, 'lanes': 2},
'Europoort → Rotterdam': {'length': 27500, 'speed_limit': 100, 'lanes': 3},
'Rotterdam → Gorinchem': {'length': 36100, 'speed_limit': 100, 'lanes': 3},
'Gorinchem → Waardenburg': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
'Waardenburg → Tiel': {'length': 15000, 'speed_limit': 100, 'lanes': 2},
'Tiel → Nijmegen': {'length': 24400, 'speed_limit': 100, 'lanes': 2},
'Nijmegen → Bemmel': {'length': 10100, 'speed_limit': 100, 'lanes': 2},
'Rotterdam → Dordrecht': {'length': 15200, 'speed_limit': 100, 'lanes': 3},
'Dordrecht → Moerdijk': {'length': 15200, 'speed_limit': 100, 'lanes': 2},
'Moerdijk → Breda': {'length': 11300, 'speed_limit': 100, 'lanes': 2},
'Breda → België': {'length': 10800, 'speed_limit': 100, 'lanes': 2},
'Moerdijk → Roosendaal': {'length': 23500, 'speed_limit': 100, 'lanes': 2},
'Zevenaar → Doetinchem': {'length': 21000, 'speed_limit': 100, 'lanes': 2},
'Doetinchem → Varsseveld': {'length': 9500, 'speed_limit': 100, 'lanes': 2},
'Gouda → Rotterdam': {'length': 16400, 'speed_limit': 100, 'lanes': 3},
'Rotterdam → Vlaardingen': {'length': 8800, 'speed_limit': 100, 'lanes': 3},
'Vlaardingen → Maassluis': {'length': 7400, 'speed_limit': 100, 'lanes': 2},
'IJmuiden → Beverwijk': {'length': 3800, 'speed_limit': 100, 'lanes': 2},
'Breda → Raamsdonk': {'length': 11900, 'speed_limit': 100, 'lanes': 2},
'Raamsdonk → Gorinchem': {'length': 18600, 'speed_limit': 100, 'lanes': 2},
'Gorinchem → Utrecht': {'length': 29100, 'speed_limit': 100, 'lanes': 3},
'Utrecht → Hilversum': {'length': 19000, 'speed_limit': 100, 'lanes': 3},
'Hilversum → Huizen': {'length': 10300, 'speed_limit': 100, 'lanes': 2},
'Huizen → Almere': {'length': 12600, 'speed_limit': 100, 'lanes': 2},
'Utrecht → Amersfoort': {'length': 18900, 'speed_limit': 100, 'lanes': 3},
'Amersfoort → Harderwijk': {'length': 27700, 'speed_limit': 100, 'lanes': 2},
'Harderwijk → Zwolle': {'length': 39100, 'speed_limit': 100, 'lanes': 2},
'Zwolle → Meppel': {'length': 21100, 'speed_limit': 100, 'lanes': 2},
'Meppel → Hoogeveen': {'length': 20800, 'speed_limit': 100, 'lanes': 2},
'Hoogeveen → Assen': {'length': 31800, 'speed_limit': 100, 'lanes': 2},
'Assen → Groningen': {'length': 26100, 'speed_limit': 100, 'lanes': 2},
'Rotterdam → Knooppunt Sabina': {'length': 25900, 'speed_limit': 100, 'lanes': 2},
'Barneveld → Ede': {'length': 12400, 'speed_limit': 100, 'lanes': 2},
'Harlingen → Leeuwarden': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
'Meppel → Steenwijk': {'length': 13200, 'speed_limit': 100, 'lanes': 2},
'Steenwijk → Heerenveen': {'length': 23800, 'speed_limit': 100, 'lanes': 2},
'Heerenveen → Leeuwarden': {'length': 23900, 'speed_limit': 100, 'lanes': 2},
'Enschede → Hengelo': {'length': 13600, 'speed_limit': 100, 'lanes': 2},
'Hengelo → Almelo': {'length': 9600, 'speed_limit': 100, 'lanes': 2},
'Almelo → Wierden': {'length': 5400, 'speed_limit': 100, 'lanes': 2},
'Hoogeveen → Emmen': {'length': 23000, 'speed_limit': 100, 'lanes': 2},
'Emmen → Duitsland': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
'Rotterdam → Ridderkerk': {'length': 1800, 'speed_limit': 100, 'lanes': 2},
'Wassenaar → Leiden': {'length': 4300, 'speed_limit': 100, 'lanes': 2},
'Leiden → Nieuw-Vennep': {'length': 17800, 'speed_limit': 100, 'lanes': 2},
'Eindhoven → Oss': {'length': 35900, 'speed_limit': 100, 'lanes': 2},
'Oss → Ewijk': {'length': 16700, 'speed_limit': 100, 'lanes': 2},
'Ewijk → Arnhem': {'length': 21200, 'speed_limit': 100, 'lanes': 2},
'Arnhem → Apeldoorn': {'length': 31200, 'speed_limit': 100, 'lanes': 2},
'Apeldoorn → Zwolle': {'length': 32299, 'speed_limit': 100, 'lanes': 2},
'Eindhoven → Tilburg': {'length': 19300, 'speed_limit': 100, 'lanes': 2},
'Tilburg → Breda': {'length': 22500, 'speed_limit': 100, 'lanes': 2},
'Breda → Roosendaal': {'length': 34000, 'speed_limit': 100, 'lanes': 2},
'Roosendaal → Bergen op Zoom': {'length': 13700, 'speed_limit': 100, 'lanes': 2},
'Bergen op Zoom → Goes': {'length': 39500, 'speed_limit': 100, 'lanes': 2},
'Goes → Middelburg': {'length': 17600, 'speed_limit': 100, 'lanes': 2},
'Middelburg → Vlissingen': {'length': 5400, 'speed_limit': 100, 'lanes': 2},
'Knooppunt Sabina → Raamsdonk': {'length': 14000, 'speed_limit': 100, 'lanes': 2},
'Raamsdonk → Waalwijk': {'length': 12200, 'speed_limit': 100, 'lanes': 2},
"Waalwijk → 's-Hertogenbosch": {'length': 17600, 'speed_limit': 100, 'lanes': 2},
"'s-Hertogenbosch → Oss": {'length': 18700, 'speed_limit': 100, 'lanes': 2},
'Tilburg → Berkel-Enschot': {'length': 3100, 'speed_limit': 100, 'lanes': 2},
"Vught → 's-Hertogenbosch": {'length': 6900, 'speed_limit': 100, 'lanes': 2},
'België → Eindhoven': {'length': 18000, 'speed_limit': 100, 'lanes': 2},
'Eindhoven → Venlo': {'length': 51800, 'speed_limit': 100, 'lanes': 2},
'Venlo → Duitsland': {'length': 1500, 'speed_limit': 100, 'lanes': 2},
'Ewijk → Nijmegen': {'length': 6600, 'speed_limit': 100, 'lanes': 2},
'Nijmegen → Boxmeer': {'length': 20300, 'speed_limit': 100, 'lanes': 2},
'Boxmeer → Venlo': {'length': 35300, 'speed_limit': 100, 'lanes': 2},
'Venlo → Roermond': {'length': 28400, 'speed_limit': 100, 'lanes': 2},
'Roermond → Echt': {'length': 13300, 'speed_limit': 100, 'lanes': 2},
'België → Stein': {'length': 1400, 'speed_limit': 100, 'lanes': 2},
'Stein → Geleen': {'length': 3500, 'speed_limit': 100, 'lanes': 2},
'Geleen → Hoensbroek': {'length': 8300, 'speed_limit': 100, 'lanes': 2},
'Hoensbroek → Heerlen': {'length': 5400, 'speed_limit': 100, 'lanes': 2},
'Heerlen → Duitsland': {'length': 8100, 'speed_limit': 100, 'lanes': 2},
'Boxmeer → Duitsland': {'length': 6400, 'speed_limit': 100, 'lanes': 2},
'Maastricht → Heerlen': {'length': 16600, 'speed_limit': 100, 'lanes': 2},
'Eindhoven → Helmond': {'length': 11800, 'speed_limit': 100, 'lanes': 2},
'Arnhem → Bemmel': {'length': 7500, 'speed_limit': 100, 'lanes': 2},
'Arnhem → Dieren': {'length': 10200, 'speed_limit': 100, 'lanes': 2}
}

# Calculate the base travel time tt_0(e) for each edge (so we just add tt_0 and capacity to the edge dictionary as they only have to be calculated once)
for edge, properties in edges.items():
    length = properties["length"]
    speed_limit = properties["speed_limit"]
    tt_0 = (length / speed_limit) * (60 / 1000)  # minutes
    properties["tt_0"] = tt_0
    properties["capacity"] = int(properties["lanes"] * properties["length"] / (l_car + d_spacing))

# Simulation settings
num_minutes = 240
car_distribution_mean = 95
car_distribution_std = 1
start_location = "VLissingen"
end_location = "Bad Nieuweschans"

# Sample the number of cars arriving at start_location each minute
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
    # Find all outgoing edges from the current location
    outgoing_edges = [edge for edge in edges if edge.startswith(f"{location} →")]
    
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

# City positions using real coordinates
node_positions = {
    "Hoensbroek": (50.9180, 5.9310),
    "Lemmer": (52.8475, 5.7194),
    "Schiedam": (51.9198, 4.3987),
    "Dieren": (52.0556, 6.1000),
    "Arnhem": (51.9851, 5.8987),
    "Knooppunt Sabina": (51.6352, 4.3728),
    "Ewijk": (51.8664, 5.7387),
    "Maastricht": (50.8514, 5.6909),
    "Meppel": (52.6951, 6.1940),
    "Assen": (53.0010, 6.5622),
    "Gouda": (52.0112, 4.7111),
    "Utrecht": (52.0907, 5.1214),
    "België": (50.85, 4.50),
    "Bad Nieuweschan": (53.1842, 7.2102),
    "Zevenaar": (51.9315, 6.0722),
    "Lelystad": (52.5185, 5.4714),
    "Geleen": (50.9745, 5.8292),
    "Bad Nieuweschans": (53.1842, 7.2102),
    "Weert": (51.2510, 5.7064),
    "Haarlem": (52.3874, 4.6462),
    "Leeuwarden": (53.2012, 5.7999),
    "Zwolle": (52.5168, 6.0830),
    "Huizen": (52.2995, 5.2413),
    "Hilversum": (52.2292, 5.1803),
    "Delft": (52.0116, 4.3571),
    "Emmeloord": (52.7107, 5.7480),
    "Zaanstad": (52.4574, 4.7510),
    "Deventer": (52.2552, 6.1639),
    "Tilburg": (51.5555, 5.0913),
    "Echt": (51.1069, 5.8741),
    "Diemen": (52.3392, 4.9622),
    "Venlo": (51.3704, 6.1724),
    "Bergen op Zoom": (51.4936, 4.2871),
    "Apeldoorn": (52.2112, 5.9699),
    "Stein": (50.9725, 5.7707),
    "Duitsland": (52.17, 7.10),
    "Nijmegen": (51.8126, 5.8372),
    "Westpoort": (52.4056, 4.8167),
    "Maasvlakte": (51.9554, 4.0236),
    "Moerdijk": (51.7031, 4.6139),
    "Amsterdam": (52.3676, 4.9041),
    "Zoetermeer": (52.0570, 4.4931),
    "Doetinchem": (51.9654, 6.2880),
    "Heerlen": (50.8882, 5.9795),
    "Hoogezand": (53.1630, 6.7628),
    "Drachten": (53.1122, 6.0989),
    "IJmuiden": (52.4607, 4.6103),
    "Steenwijk": (52.7853, 6.1191),
    "Maassluis": (51.9224, 4.2492),
    "Eindhoven": (51.4416, 5.4697),
    "Harlingen": (53.1736, 5.4208),
    "Beverwijk": (52.4873, 4.6563),
    "Purmerend": (52.5053, 4.9592),
    "Bemmel": (51.8886, 5.9029),
    "Almelo": (52.3564, 6.6626),
    "Waardenburg": (51.8314, 5.2671),
    "Berkel-Enschot": (51.5816, 5.1666),
    "Vught": (51.6558, 5.2873),
    "Wassenaar": (52.1397, 4.4019),
    "Groningen": (53.2194, 6.5665),
    "Hoofddorp": (52.3061, 4.6907),
    "Amersfoort": (52.1560, 5.3878),
    "Roosendaal": (51.5306, 4.4654),
    "Amstelveen": (52.3089, 4.8495),
    "Ridderkerk": (51.8720, 4.6021),
    "Varsseveld": (51.9421, 6.4603),
    "Winschoten": (53.1427, 7.0344),
    "Tiel": (51.8858, 5.4294),
    "Sneek": (53.0319, 5.6583),
    "Vlissingen": (51.4426, 3.5735),
    "Knooppunt Benelux": (51.8997, 4.3706),
    "Europoort": (51.9525, 4.1719),
    "Gorinchem": (51.8352, 4.9704),
    "Hoorn": (52.6425, 5.0597),
    "Ede": (52.0468, 5.6640),
    "Waalwijk": (51.6826, 5.0729),
    "Boxmeer": (51.6458, 5.9475),
    "Raamsdonk": (51.7192, 4.9006),
    "Den Haag": (52.0705, 4.3007),
    "Oss": (51.7645, 5.5180),
    "Afsluitdijk": (53.0713, 5.0409),
    "Schiphol": (52.3105, 4.7683),
    "Almere": (52.3508, 5.2647),
    "Hengelo": (52.2659, 6.7930),
    "Goes": (51.5047, 3.8883),
    "Middelburg": (51.4988, 3.6136),
    "Rotterdam": (51.9225, 4.47917),
    "Enschede": (52.2215, 6.8937),
    "Joure": (52.9633, 5.8051),
    "Hoogeveen": (52.7221, 6.4866),
    "Wierden": (52.3581, 6.5933),
    "Harderwijk": (52.3410, 5.6208),
    "Barneveld": (52.1416, 5.5805),
    "Nieuw-Vennep": (52.2635, 4.6342),
    "Emmen": (52.7850, 6.8950),
    "Helmond": (51.4793, 5.6576),
    "Heerenveen": (52.9602, 5.9195),
    "Leiden": (52.1601, 4.4970),
    "'s-Hertogenbosch": (51.6978, 5.3037),
    "Knooppunt Muiderberg": (52.3294, 5.0797),
    "Vlaardingen": (51.9121, 4.3419),
    "Roermond": (51.1942, 5.9870),
    "Dordrecht": (51.8133, 4.6901),
    "Breda": (51.5719, 4.7683)
}

all_cities_list = list(node_positions.keys())
all_cities_lits_without_end_location = [city for city in all_cities_list if city != end_location]
all_cities_list_without_start_location = [city for city in all_cities_list if city != start_location]

# Reloading the image and setting up dimensions
img = mpimg.imread('./blank_netherlands.png')

# Set the correct lat/lon extent for the map of the Netherlands
# Here we assume these lat/lon bounds (min_lat, max_lat, min_lon, max_lon) for the image:
lat_min, lat_max = 50.75, 53.55  # Approx latitude range of the Netherlands
lon_min, lon_max = 3.36, 7.22    # Approx longitude range of the Netherlands

# Plotting function to visualize nodes, edges, and car counts
def initialize_plot():
    fig, ax = plt.subplots(figsize=(12, 16))

    # Set the plot limits to match the extent of the image
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])

    # Load and display the background image (scale according to the map)
    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto')  # Fit image to the latitude/longitude extent

    # Plot cities on the map using their actual lat/lon coordinates
    for node, (lat, lon) in node_positions.items():
        ax.plot(lon, lat, 'o', markersize=8, color='red')  # Directly use lon (x) and lat (y) for plotting
        ax.text(lon, lat, node, fontsize=9, ha='center', color='black')  # Add city names at lon (x), lat (y)

    ax.set_facecolor('black')         # Black background for the axes
    ax.set_title("Real-Time Simulation of Car Movements", color='black')  # Title in white

    # Store edge and node text for dynamic updating
    edge_texts = {}
    for edge in edges:
        start_node, end_node = edge.split(" → ")
        start_pos, end_pos = node_positions[start_node], node_positions[end_node]
        mid_pos = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)

        # Increase line width for better visibility
        ax.plot([start_pos[1], end_pos[1]], [start_pos[0], end_pos[0]], color='cyan', linestyle='--', linewidth=2)
        # Increase font size for edge texts
        edge_texts[edge] = ax.text(mid_pos[1], mid_pos[0], '', fontsize=12, color='purple')

    # Adjust node text color and size
    node_texts = {node: ax.text(pos[1], pos[0] - 0.05, '', fontsize=12, color="blue") for node, pos in node_positions.items()}

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

    # Update node texts with the number of cars waiting at each node
    cars_at_node = {node: 0 for node in node_positions}
    for car in cars:
        if car["location"] in all_cities_list:
            cars_at_node[car["location"]] += 1

    for node, text in node_texts.items():
        text.set_text(f"Waiting: {cars_at_node[node]}")

    # Update the timestep text
    timestep_text.set_text(f"Timestep: {t}/{num_minutes}")

# Simulation and visualization combined
def simulate_and_visualize(cars, edges, num_minutes, warmup_steps=120, most_congested_edge=None, capacity_multiplier=1.0, animate=False):
    fig, ax, edge_texts, node_texts, timestep_text = initialize_plot()

    # Initialize vehicle counts on each edge
    vehicle_counts = {edge: np.zeros(1, dtype=int) for edge in edges}

    # Track travel times to end_location
    car_reach_times = []

    # Track the congestion of each edge
    if most_congested_edge is None:
        congestion_data = {edge: [] for edge in edges}
        for edge in edges:
            edges[edge]["current_capacity"] = edges[edge]["capacity"]
    else:
        edges[most_congested_edge]["current_capacity"] = int(edges[most_congested_edge]["capacity"] * capacity_multiplier)

    # Function to update the plot at each timestep
    def update(t):
        for car in cars:
            # Check if the car should start yet
            if car["start_time"] == t:
                car["location"] = start_location
            # Skip cars that haven't started yet or have already finished
            elif t < car["start_time"] or car["location"] == end_location:
                continue

            # Check if the car has finished its route
            if t-car["start_time"] >= car["arrived_at_node"] and not car["location"] in all_cities_lits_without_end_location:
                # Store the edge the car was on before reaching a node
                car["was_on_route"] = car["location"]
                # Update the car's location
                car["location"] = car["location"].split(" → ")[-1]
                # Check if the reached location is end_location
                if car["location"] == end_location:
                    # Remove the car from the route it just finished if it reached end_location
                    vehicle_counts[car["was_on_route"]] -= 1
                    if car["start_time"] > warmup_steps:
                        car_reach_times.append(car["arrived_at_node"])
                continue

            # Choose the next edge if the car is at a node
            if car["location"] in all_cities_lits_without_end_location:
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
            if most_congested_edge is None:
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
        # Run the simulation without animation
        for t in range(num_minutes):
            update(t)

    # Calculate average congestion per edge
    if most_congested_edge is None:
        avg_congestion = {edge: np.mean(congestion_data[edge]) for edge in edges}
    else:
        avg_congestion = None

    return car_reach_times, avg_congestion

def reset_vehicle_counts(edges):
    for edge in edges:
        edges[edge]["current_capacity"] = edges[edge]["capacity"]  # Reset to original capacity

def reset_car_states(cars):
    for car in cars:
        car["location"] = ""
        car["arrived_at_node"] = 0
        car["left_at_node"] = 0
        car["was_on_route"] = ""

# Run the function multiple times with different capacity values
def simulate_and_compare(cars, edges, num_minutes, deltas=[1.5, 2]):
    # Ask the user if it wants the traffic simulation to be animated
    animate = input("Do you want to animate the traffic simulation in real-time? You will have to close the plots for the code to continue if a full simulation is done. (y/n): ").lower() == "y"

    # Initial run to find the most congested edge
    print(f"Running simulation with capacity multiplier: 1.0")

    car_reach_times, avg_congestion = simulate_and_visualize(cars, edges, num_minutes, warmup_steps=120, animate=animate)
    most_congested_edge = max(avg_congestion, key=avg_congestion.get)
    print(f"Most congested edge: {most_congested_edge}")

    all_car_reach_times = [(1.0, car_reach_times)]

    # Run the simulation multiple times with different capacity multipliers
    for delta in deltas:
        # Reset vehicle counts and car states before running the simulation
        reset_vehicle_counts(edges)
        reset_car_states(cars)

        print(f"Running simulation with capacity multiplier: {delta}")
        car_reach_times, _ = simulate_and_visualize(cars, edges, num_minutes, most_congested_edge=most_congested_edge, capacity_multiplier=delta, warmup_steps=10, animate=animate)
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
        cars_reached_city_2.append(len(car_reach_times))  # Number of cars that reached end_location for each delta

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

    # Scatter plot 2: Delta vs. Number of Cars Reached end_location
    plt.subplot(1, 2, 2)
    plt.scatter(deltas, cars_reached_city_2, color='g')
    plt.title(f"Delta vs. Number of Cars Reached {end_location}")
    plt.xlabel("Capacity Multiplier (Delta)")
    plt.ylabel(f"Number of Cars Reached {end_location}")

    # Compute correlation coefficient for delta vs cars reached
    correlation_coef2, _ = pearsonr(deltas, cars_reached_city_2)
    plt.text(1.1, max(cars_reached_city_2) - 10, f"Correlation Coefficient: {correlation_coef2:.2f}", fontsize=12, color='green')

    plt.tight_layout()
    plt.show()

# Run the simulation and plot the results
all_car_reach_times, most_congested_edge = simulate_and_compare(cars, edges, num_minutes)
plot_3D_surface(all_car_reach_times)
plot_scatter_with_correlation(all_car_reach_times)