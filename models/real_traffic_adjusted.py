import numpy as np
from utils.plotting import plot_3D_surface, plot_scatter_with_correlation
from utils.functional import add_properties_to_edges
from utils.simulate import simulate_and_compare
import matplotlib.image as mpimg
from real_data.parse_edges import parse_highway_data

# Parameters for BPR function
alpha = 0.15
beta = 4
sigma = 2
l_car = 4.5 # Length of a car in meters
d_spacing = 55 # Minimum safe spacing between cars in meters

# edges = {
# 'Amsterdam → Hilversum': {'length': 22000, 'speed_limit': 100, 'lanes': 3},
# 'Hilversum → Amersfoort': {'length': 15000, 'speed_limit': 100, 'lanes': 3},
# 'Amersfoort → Apeldoorn': {'length': 42500, 'speed_limit': 100, 'lanes': 3},
# 'Apeldoorn → Deventer': {'length': 16500, 'speed_limit': 100, 'lanes': 3}, 
# 'Deventer → Hengelo': {'length': 60000, 'speed_limit': 100, 'lanes': 3}, 
# 'Hengelo → Duitsland': {'length': 18200, 'speed_limit': 100, 'lanes': 2}, 
# 'Duitsland → Zevenaar': {'length': 7200, 'speed_limit': 100, 'lanes': 2},
# 'Zevenaar → Arnhem': {'length': 8600, 'speed_limit': 100, 'lanes': 3},
# 'Arnhem → Ede': {'length': 24900, 'speed_limit': 100, 'lanes': 3},
# 'Ede → Utrecht': {'length': 38400, 'speed_limit': 100, 'lanes': 2},
# 'Utrecht → Gouda': {'length': 31000, 'speed_limit': 100, 'lanes': 3},
# 'Gouda → Zoetermeer': {'length': 13400, 'speed_limit': 100, 'lanes': 3},
# 'Zoetermeer → Den Haag': {'length': 12700, 'speed_limit': 100, 'lanes': 3},
# 'Den Haag → Delft': {'length': 3400, 'speed_limit': 100, 'lanes': 3},
# 'Delft → Schiedam': {'length': 6900, 'speed_limit': 100, 'lanes': 3},
# 'Schiedam → Vlaardingen': {'length': 1900, 'speed_limit': 100, 'lanes': 3},
# 'Vlaardingen → Knooppunt Benelux': {'length': 4400, 'speed_limit': 100, 'lanes': 2},

# 'Amsterdam → Utrecht': {'length': 26600, 'speed_limit': 100, 'lanes': 5},
# 'Utrecht → Waardenburg': {'length': 26800, 'speed_limit': 100, 'lanes': 4},
# "Waardenburg → 's-Hertogenbosch": {'length': 17200, 'speed_limit': 100, 'lanes': 4},
# "'s-Hertogenbosch → Eindhoven": {'length': 37100, 'speed_limit': 100, 'lanes': 3},
# 'Eindhoven → Weert': {'length': 32400, 'speed_limit': 100, 'lanes': 2},
# 'Weert → Echt': {'length': 20600, 'speed_limit': 100, 'lanes': 2},
# 'Echt → Geleen': {'length': 19900, 'speed_limit': 100, 'lanes': 2},
# 'Geleen → Maastricht': {'length': 14200, 'speed_limit': 100, 'lanes': 2},
# 'Maastricht → België': {'length': 12700, 'speed_limit': 100, 'lanes': 2},
# 'Amsterdam → Schiphol': {'length': 7400, 'speed_limit': 100, 'lanes': 3},
# 'Schiphol → Leiden': {'length': 26700, 'speed_limit': 100, 'lanes': 4},
# 'Leiden → Den Haag': {'length': 13800, 'speed_limit': 100, 'lanes': 3},

# 'Knooppunt Sabina → Bergen op Zoom': {'length': 23900, 'speed_limit': 100, 'lanes': 2},
# 'Bergen op Zoom → België': {'length': 15800, 'speed_limit': 100, 'lanes': 2},
# 'Hoofddorp → Westpoort': {'length': 13000, 'speed_limit': 100, 'lanes': 2},
# 'Westpoort → Amsterdam': {'length': 16200, 'speed_limit': 100, 'lanes': 2},
# 'Knooppunt Muiderberg → Almere': {'length': 11600, 'speed_limit': 100, 'lanes': 3},
# 'Almere → Lelystad': {'length': 21900, 'speed_limit': 100, 'lanes': 3},
# 'Lelystad → Emmeloord': {'length': 32200, 'speed_limit': 100, 'lanes': 2},
# 'Emmeloord → Lemmer': {'length': 19000, 'speed_limit': 100, 'lanes': 2},
# 'Lemmer → Joure': {'length': 14200, 'speed_limit': 100, 'lanes': 2},
# 'Zaanstad → Purmerend': {'length': 8400, 'speed_limit': 100, 'lanes': 3},
# 'Purmerend → Hoorn': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
# 'Hoorn → Afsluitdijk': {'length': 45800, 'speed_limit': 100, 'lanes': 2},
# 'Afsluitdijk → Sneek': {'length': 36900, 'speed_limit': 100, 'lanes': 2},
# 'Sneek → Joure': {'length': 11700, 'speed_limit': 100, 'lanes': 2},
# 'Joure → Heerenveen': {'length': 9400, 'speed_limit': 100, 'lanes': 2},
# 'Heerenveen → Drachten': {'length': 20600, 'speed_limit': 100, 'lanes': 2},
# 'Drachten → Groningen': {'length': 33900, 'speed_limit': 100, 'lanes': 2},
# 'Groningen → Hoogezand': {'length': 15000, 'speed_limit': 100, 'lanes': 2},
# 'Hoogezand → Winschoten': {'length': 21600, 'speed_limit': 100, 'lanes': 2},
# 'Winschoten → Bad Nieuweschan': {'length': 11400, 'speed_limit': 100, 'lanes': 2},
# 'Bad Nieuweschans → Duitsland': {'length': 1700, 'speed_limit': 100, 'lanes': 2},
# 'Amsterdam → Zaanstad': {'length': 4300, 'speed_limit': 100, 'lanes': 3},
# 'Diemen → Amstelveen': {'length': 14000, 'speed_limit': 100, 'lanes': 3},
# 'Amstelveen → Haarlem': {'length': 14900, 'speed_limit': 100, 'lanes': 3},
# 'Haarlem → Beverwijk': {'length': 12100, 'speed_limit': 100, 'lanes': 2},

# 'Delft → Rotterdam': {'length': 10600, 'speed_limit': 100, 'lanes': 3},
# 'Maasvlakte → Europoort': {'length': 3600, 'speed_limit': 100, 'lanes': 2},
# 'Europoort → Rotterdam': {'length': 27500, 'speed_limit': 100, 'lanes': 3},
# 'Rotterdam → Gorinchem': {'length': 36100, 'speed_limit': 100, 'lanes': 3},
# 'Gorinchem → Waardenburg': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
# 'Waardenburg → Tiel': {'length': 15000, 'speed_limit': 100, 'lanes': 2},
# 'Tiel → Nijmegen': {'length': 24400, 'speed_limit': 100, 'lanes': 2},
# 'Nijmegen → Bemmel': {'length': 10100, 'speed_limit': 100, 'lanes': 2},
# 'Rotterdam → Dordrecht': {'length': 15200, 'speed_limit': 100, 'lanes': 3},
# 'Dordrecht → Moerdijk': {'length': 15200, 'speed_limit': 100, 'lanes': 2},
# 'Moerdijk → Breda': {'length': 11300, 'speed_limit': 100, 'lanes': 2},
# 'Breda → België': {'length': 10800, 'speed_limit': 100, 'lanes': 2},
# 'Moerdijk → Roosendaal': {'length': 23500, 'speed_limit': 100, 'lanes': 2},
# 'Zevenaar → Doetinchem': {'length': 21000, 'speed_limit': 100, 'lanes': 2},
# 'Doetinchem → Varsseveld': {'length': 9500, 'speed_limit': 100, 'lanes': 2},
# 'Gouda → Rotterdam': {'length': 16400, 'speed_limit': 100, 'lanes': 3},
# 'Rotterdam → Vlaardingen': {'length': 8800, 'speed_limit': 100, 'lanes': 3},
# 'Vlaardingen → Maassluis': {'length': 7400, 'speed_limit': 100, 'lanes': 2},
# 'IJmuiden → Beverwijk': {'length': 3800, 'speed_limit': 100, 'lanes': 2},
# 'Breda → Raamsdonk': {'length': 11900, 'speed_limit': 100, 'lanes': 2},
# 'Raamsdonk → Gorinchem': {'length': 18600, 'speed_limit': 100, 'lanes': 2},
# 'Gorinchem → Utrecht': {'length': 29100, 'speed_limit': 100, 'lanes': 3},
# 'Utrecht → Hilversum': {'length': 19000, 'speed_limit': 100, 'lanes': 3},
# 'Hilversum → Huizen': {'length': 10300, 'speed_limit': 100, 'lanes': 2},
# 'Huizen → Almere': {'length': 12600, 'speed_limit': 100, 'lanes': 2},
# 'Utrecht → Amersfoort': {'length': 18900, 'speed_limit': 100, 'lanes': 3},
# 'Amersfoort → Harderwijk': {'length': 27700, 'speed_limit': 100, 'lanes': 2},
# 'Harderwijk → Zwolle': {'length': 39100, 'speed_limit': 100, 'lanes': 2},
# 'Zwolle → Meppel': {'length': 21100, 'speed_limit': 100, 'lanes': 2},
# 'Meppel → Hoogeveen': {'length': 20800, 'speed_limit': 100, 'lanes': 2},
# 'Hoogeveen → Assen': {'length': 31800, 'speed_limit': 100, 'lanes': 2},
# 'Assen → Groningen': {'length': 26100, 'speed_limit': 100, 'lanes': 2},
# 'Rotterdam → Knooppunt Sabina': {'length': 25900, 'speed_limit': 100, 'lanes': 2},
# 'Barneveld → Ede': {'length': 12400, 'speed_limit': 100, 'lanes': 2},
# 'Harlingen → Leeuwarden': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
# 'Meppel → Steenwijk': {'length': 13200, 'speed_limit': 100, 'lanes': 2},
# 'Steenwijk → Heerenveen': {'length': 23800, 'speed_limit': 100, 'lanes': 2},
# 'Heerenveen → Leeuwarden': {'length': 23900, 'speed_limit': 100, 'lanes': 2},
# 'Enschede → Hengelo': {'length': 13600, 'speed_limit': 100, 'lanes': 2},
# 'Hengelo → Almelo': {'length': 9600, 'speed_limit': 100, 'lanes': 2},
# 'Almelo → Wierden': {'length': 5400, 'speed_limit': 100, 'lanes': 2},
# 'Hoogeveen → Emmen': {'length': 23000, 'speed_limit': 100, 'lanes': 2},
# 'Emmen → Duitsland': {'length': 18300, 'speed_limit': 100, 'lanes': 2},
# 'Rotterdam → Ridderkerk': {'length': 1800, 'speed_limit': 100, 'lanes': 2},
# 'Wassenaar → Leiden': {'length': 4300, 'speed_limit': 100, 'lanes': 2},
# 'Leiden → Nieuw-Vennep': {'length': 17800, 'speed_limit': 100, 'lanes': 2},
# 'Eindhoven → Oss': {'length': 35900, 'speed_limit': 100, 'lanes': 2},
# 'Oss → Ewijk': {'length': 16700, 'speed_limit': 100, 'lanes': 2},
# 'Ewijk → Arnhem': {'length': 21200, 'speed_limit': 100, 'lanes': 2},
# 'Arnhem → Apeldoorn': {'length': 31200, 'speed_limit': 100, 'lanes': 2},
# 'Apeldoorn → Zwolle': {'length': 32299, 'speed_limit': 100, 'lanes': 2},
# 'Eindhoven → Tilburg': {'length': 19300, 'speed_limit': 100, 'lanes': 2},
# 'Tilburg → Breda': {'length': 22500, 'speed_limit': 100, 'lanes': 2},
# 'Breda → Roosendaal': {'length': 34000, 'speed_limit': 100, 'lanes': 2},
# 'Roosendaal → Bergen op Zoom': {'length': 13700, 'speed_limit': 100, 'lanes': 2},
# 'Bergen op Zoom → Goes': {'length': 39500, 'speed_limit': 100, 'lanes': 2},
# 'Goes → Middelburg': {'length': 17600, 'speed_limit': 100, 'lanes': 2},
# 'Middelburg → Vlissingen': {'length': 5400, 'speed_limit': 100, 'lanes': 2},
# 'Knooppunt Sabina → Raamsdonk': {'length': 14000, 'speed_limit': 100, 'lanes': 2},
# 'Raamsdonk → Waalwijk': {'length': 12200, 'speed_limit': 100, 'lanes': 2},
# "Waalwijk → 's-Hertogenbosch": {'length': 17600, 'speed_limit': 100, 'lanes': 2},
# "'s-Hertogenbosch → Oss": {'length': 18700, 'speed_limit': 100, 'lanes': 2},
# 'Tilburg → Berkel-Enschot': {'length': 3100, 'speed_limit': 100, 'lanes': 2},
# "Vught → 's-Hertogenbosch": {'length': 6900, 'speed_limit': 100, 'lanes': 2},
# 'België → Eindhoven': {'length': 18000, 'speed_limit': 100, 'lanes': 2},
# 'Eindhoven → Venlo': {'length': 51800, 'speed_limit': 100, 'lanes': 2},
# 'Venlo → Duitsland': {'length': 1500, 'speed_limit': 100, 'lanes': 2},
# 'Ewijk → Nijmegen': {'length': 6600, 'speed_limit': 100, 'lanes': 2},
# 'Nijmegen → Boxmeer': {'length': 20300, 'speed_limit': 100, 'lanes': 2},
# 'Boxmeer → Venlo': {'length': 35300, 'speed_limit': 100, 'lanes': 2},
# 'Venlo → Roermond': {'length': 28400, 'speed_limit': 100, 'lanes': 2},
# 'Roermond → Echt': {'length': 13300, 'speed_limit': 100, 'lanes': 2},
# 'België → Stein': {'length': 1400, 'speed_limit': 100, 'lanes': 2},
# 'Stein → Geleen': {'length': 3500, 'speed_limit': 100, 'lanes': 2},
# 'Geleen → Hoensbroek': {'length': 8300, 'speed_limit': 100, 'lanes': 2},
# 'Hoensbroek → Heerlen': {'length': 5400, 'speed_limit': 100, 'lanes': 2},
# 'Heerlen → Duitsland': {'length': 8100, 'speed_limit': 100, 'lanes': 2},
# 'Boxmeer → Duitsland': {'length': 6400, 'speed_limit': 100, 'lanes': 2},
# 'Maastricht → Heerlen': {'length': 16600, 'speed_limit': 100, 'lanes': 2},
# 'Eindhoven → Helmond': {'length': 11800, 'speed_limit': 100, 'lanes': 2},
# 'Arnhem → Bemmel': {'length': 7500, 'speed_limit': 100, 'lanes': 2},
# 'Arnhem → Dieren': {'length': 10200, 'speed_limit': 100, 'lanes': 2}
# }

# Use the function from real_data to get the real edges
edges = parse_highway_data('./real_data/real_highway_data.txt')

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
    "Afsluitdijk": (53.0713, 5.2409), #5.0409
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

# Add properties to edges
edges = add_properties_to_edges(edges, l_car, d_spacing)

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

# Reloading the image and setting up dimensions
bg_image = mpimg.imread('./Images/netherlands/blank_netherlands_adjusted.png')

# Set the correct lat/lon extent for the map of the Netherlands
# Here we assume these lat/lon bounds (min_lat, max_lat, min_lon, max_lon) for the image:
lat_min, lat_max = 50.75, 53.55  # Approx latitude range of the Netherlands
lon_min, lon_max = 3.36, 7.22    # Approx longitude range of the Netherlands

starting_city = "Middelburg"
ending_city = "Groningen"

# Run the simulation and plot the results
all_car_reach_times, most_congested_edge = simulate_and_compare(cars, edges, node_positions, num_minutes, warmup_steps, deltas=[1.0, 1.5], alpha=0.15, beta=4, sigma=2, bg_image=bg_image, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, starting_city=starting_city, ending_city=ending_city)
plot_3D_surface(all_car_reach_times)#[1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
plot_scatter_with_correlation(all_car_reach_times)