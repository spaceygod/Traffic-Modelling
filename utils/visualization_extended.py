import matplotlib.pyplot as plt
from matplotlib import cm, colors

import numpy as np

nodes = {
    "Hoensbroek": {"coordinates": (50.9180, 5.9310), "population": 18860},
    "Lemmer": {"coordinates": (52.8475, 5.7194), "population": 10000},
    "Schiedam": {"coordinates": (51.9198, 4.3987), "population": 78000},
    "Dieren": {"coordinates": (52.0556, 6.1000), "population": 15000},
    "Arnhem": {"coordinates": (51.9851, 5.8987), "population": 160000},
    "Knooppunt Sabina": {"coordinates": (51.6352, 4.3728), "population": None},
    "Ewijk": {"coordinates": (51.8664, 5.7387), "population": 3500},
    "Maastricht": {"coordinates": (50.8514, 5.6909), "population": 122000},
    "Meppel": {"coordinates": (52.6951, 6.1940), "population": 34000},
    "Assen": {"coordinates": (53.0010, 6.5622), "population": 68000},
    "Gouda": {"coordinates": (52.0112, 4.7111), "population": 74000},
    "Utrecht": {"coordinates": (52.0907, 5.1214), "population": 360000},
    # "België": {"coordinates": (50.85, 4.50), "population": None},
    "Bad Nieuweschans": {"coordinates": (53.1842, 7.2102), "population": 1500},
    "Zevenaar": {"coordinates": (51.9315, 6.0722), "population": 24000},
    "Lelystad": {"coordinates": (52.5185, 5.4714), "population": 79000},
    "Geleen": {"coordinates": (50.9745, 5.8292), "population": 32000},
    "Weert": {"coordinates": (51.2510, 5.7064), "population": 50000},
    "Haarlem": {"coordinates": (52.3874, 4.6462), "population": 162000},
    "Leeuwarden": {"coordinates": (53.2012, 5.7999), "population": 128810},
    "Zwolle": {"coordinates": (52.5168, 6.0830), "population": 133000},
    "Huizen": {"coordinates": (52.2995, 5.2413), "population": 42100},
    "Hilversum": {"coordinates": (52.2292, 5.1803), "population": 92000},
    "Delft": {"coordinates": (52.0116, 4.3571), "population": 104000},
    "Emmeloord": {"coordinates": (52.7107, 5.7480), "population": 25500},
    "Zaanstad": {"coordinates": (52.4574, 4.7510), "population": 156000},
    "Deventer": {"coordinates": (52.2552, 6.1639), "population": 101000},
    "Tilburg": {"coordinates": (51.5555, 5.0913), "population": 224000},
    "Echt": {"coordinates": (51.1069, 5.8741), "population": 33000},
    "Diemen": {"coordinates": (52.3392, 4.9622), "population": 32000},
    "Venlo": {"coordinates": (51.3704, 6.1724), "population": 101000},
    "Bergen op Zoom": {"coordinates": (51.4936, 4.2871), "population": 67000},
    "Apeldoorn": {"coordinates": (52.2112, 5.9699), "population": 165000},
    "Stein": {"coordinates": (50.9725, 5.7707), "population": 25000},
    # "Duitsland": {"coordinates": (52.17, 7.10), "population": None},
    "Nijmegen": {"coordinates": (51.8126, 5.8372), "population": 178000},
    "Westpoort": {"coordinates": (52.4056, 4.8167), "population": None},
    "Maasvlakte": {"coordinates": (51.9554, 4.0236), "population": None},
    "Moerdijk": {"coordinates": (51.7031, 4.6139), "population": 7400},
    "Amsterdam": {"coordinates": (52.3676, 4.9041), "population": 931000},
    "Zoetermeer": {"coordinates": (52.0570, 4.4931), "population": 125000},
    "Doetinchem": {"coordinates": (51.9654, 6.2880), "population": 57000},
    "Heerlen": {"coordinates": (50.8882, 5.9795), "population": 88000},
    "Hoogezand": {"coordinates": (53.1630, 6.7628), "population": 35000},
    "Drachten": {"coordinates": (53.1122, 6.0989), "population": 45000},
    "IJmuiden": {"coordinates": (52.4607, 4.6103), "population": 31000},
    "Steenwijk": {"coordinates": (52.7853, 6.1191), "population": 18000},
    "Maassluis": {"coordinates": (51.9224, 4.2492), "population": 33000},
    "Eindhoven": {"coordinates": (51.4416, 5.4697), "population": 240000},
    "Harlingen": {"coordinates": (53.1736, 5.4208), "population": 16000},
    "Beverwijk": {"coordinates": (52.4873, 4.6563), "population": 41000},
    "Purmerend": {"coordinates": (52.5053, 4.9592), "population": 80000},
    "Bemmel": {"coordinates": (51.8886, 5.9029), "population": 12000},
    "Almelo": {"coordinates": (52.3564, 6.6626), "population": 72000},
    "Waardenburg": {"coordinates": (51.8314, 5.2671), "population": 2000},
    "Berkel-Enschot": {"coordinates": (51.5816, 5.1666), "population": 11000},
    "Vught": {"coordinates": (51.6558, 5.2873), "population": 27000},
    "Wassenaar": {"coordinates": (52.1397, 4.4019), "population": 26000},
    "Groningen": {"coordinates": (53.2194, 6.5665), "population": 235000},
    "Hoofddorp": {"coordinates": (52.3061, 4.6907), "population": 78590},
    "Amersfoort": {"coordinates": (52.1560, 5.3878), "population": 159130},
    "Roosendaal": {"coordinates": (51.5306, 4.4654), "population": 77700},
    "Amstelveen": {"coordinates": (52.3089, 4.8495), "population": 94211},
    "Ridderkerk": {"coordinates": (51.8720, 4.6021), "population": 46582},
    "Varsseveld": {"coordinates": (51.9421, 6.4603), "population": 6025},
    "Winschoten": {"coordinates": (53.1427, 7.0344), "population": 18210},
    "Tiel": {"coordinates": (51.8858, 5.4294), "population": 42200},
    "Sneek": {"coordinates": (53.0319, 5.6583), "population": 34195},
    "Vlissingen": {"coordinates": (51.4426, 3.5735), "population": 44600},
    "Knooppunt Benelux": {"coordinates": (51.8997, 4.3706), "population": None},
    "Europoort": {"coordinates": (51.9525, 4.1719), "population": None},
    "Gorinchem": {"coordinates": (51.8352, 4.9704), "population": 37575},
    "Hoorn": {"coordinates": (52.6425, 5.0597), "population": 75825},
    "Ede": {"coordinates": (52.0468, 5.6640), "population": 118630},
    "Waalwijk": {"coordinates": (51.6826, 5.0729), "population": 48900},
    "Boxmeer": {"coordinates": (51.6458, 5.9475), "population": 28700},
    "Raamsdonk": {"coordinates": (51.7192, 4.9006), "population": 1685},
    "Den Haag": {"coordinates": (52.0705, 4.3007), "population": 552995},
    "Oss": {"coordinates": (51.7645, 5.5180), "population": 93350},
    "Afsluitdijk": {"coordinates": (53.0713, 5.2409), "population": None},
    "Schiphol": {"coordinates": (52.3105, 4.7683), "population": 45},
    "Almere": {"coordinates": (52.3508, 5.2647), "population": 220614},
    "Hengelo": {"coordinates": (52.2659, 6.7930), "population": 81709},
    "Goes": {"coordinates": (51.5047, 3.8883), "population": 38435},
    "Middelburg": {"coordinates": (51.4988, 3.6136), "population": 48732},
    "Rotterdam": {"coordinates": (51.9225, 4.47917), "population": 655468},
    "Enschede": {"coordinates": (52.2215, 6.8937), "population": 159286},
    "Joure": {"coordinates": (52.9633, 5.8051), "population": 13110},
    "Hoogeveen": {"coordinates": (52.7221, 6.4866), "population": 55949},
    "Wierden": {"coordinates": (52.3581, 6.5933), "population": 24533},
    "Harderwijk": {"coordinates": (52.3410, 5.6208), "population": 48916},
    "Barneveld": {"coordinates": (52.1416, 5.5805), "population": 59411},
    "Nieuw-Vennep": {"coordinates": (52.2635, 4.6342), "population": 31835},
    "Emmen": {"coordinates": (52.7850, 6.8950), "population": 107866},
    "Helmond": {"coordinates": (51.4793, 5.6576), "population": 93998},
    "Heerenveen": {"coordinates": (52.9602, 5.9195), "population": 50845},
    "Leiden": {"coordinates": (52.1601, 4.4970), "population": 124091},
    "'s-Hertogenbosch": {"coordinates": (51.6978, 5.3037), "population": 157476},
    "Knooppunt Muiderberg": {"coordinates": (52.3294, 5.0797), "population": None},
    "Vlaardingen": {"coordinates": (51.9121, 4.3419), "population": 74073},
    "Roermond": {"coordinates": (51.1942, 5.9870), "population": 61163},
    "Dordrecht": {"coordinates": (51.8133, 4.6901), "population": 119522},
    "Breda": {"coordinates": (51.5719, 4.7683), "population": 184409}
}


def initialize_plot(edges, node_positions, bg_image=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    edge_texts = {}
    edge_lines = {}

    # Filter out cities with None population, but include België and Duitsland
    valid_nodes = {city: info for city, info in nodes.items() if info['population'] is not None or city in ['België', 'Duitsland']}
    # print("Number of valid nodes:", len(valid_nodes))
    top_cities = sorted(valid_nodes.items(), key=lambda item: item[1]['population'] if item[1]['population'] is not None else 0, reverse=True)[:30]
    top_city_names = {city: info['coordinates'] for city, info in top_cities}

    if bg_image is None:
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim(3.5, 7.7)
        ax.set_ylim(50.5, 54)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.set_title("Real-Time Simulation of Car Movements", color='black', fontsize=14)

        # Plot all nodes (cities) with light gray color
        for node, pos in nodes.items():
            ax.plot(pos['coordinates'][1], pos['coordinates'][0], 'o', markersize=8, color='lightgray')
            # Highlight only top cities with their names
            if node in top_city_names:
                ax.text(pos['coordinates'][1], pos['coordinates'][0] + 0.05, node, fontsize=6, ha='center', va='bottom', color='black')
            elif node in ['België', 'Duitsland']:  # Always show names for these two
                ax.text(pos['coordinates'][1], pos['coordinates'][0] + 0.05, node, fontsize=6, ha='center', va='bottom', color='red')

        # Initialize edge lines (if edges are provided)
        for edge in edges:
            start_node, end_node = edge.split(" → ")
            if start_node in nodes and end_node in nodes:  # Check for existing nodes
                start_pos, end_pos = nodes[start_node]['coordinates'], nodes[end_node]['coordinates']
                mid_pos = ((start_pos[1] + end_pos[1]) / 2, (start_pos[0] + end_pos[0]) / 2)
                line, = ax.plot([start_pos[1], end_pos[1]], [start_pos[0], end_pos[0]], color='green', linestyle='-', linewidth=1)
                edge_lines[edge] = line
                edge_texts[edge] = ax.text(mid_pos[0], mid_pos[1], '', fontsize=8, color='purple')
    else:
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim([lon_min, lon_max])
        ax.set_ylim([lat_min, lat_max])
        ax.imshow(bg_image, extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto')
        ax.set_title("Real-Time Simulation of Car Movements", color='white', fontsize=14)

        for node, pos in nodes.items():
            ax.plot(pos['coordinates'][1], pos['coordinates'][0], 'o', markersize=8, color='lightgray')
            if node in top_city_names:
                ax.text(pos['coordinates'][1], pos['coordinates'][0] + 0.05, node, fontsize=8, ha='center', va='bottom', color='purple')
            elif node in ['België', 'Duitsland']:
                ax.text(pos['coordinates'][1], pos['coordinates'][0] + 0.05, node, fontsize=8, ha='center', va='bottom', color='red')

        for edge in edges:
            start_node, end_node = edge.split(" → ")
            if start_node in nodes and end_node in nodes:
                start_pos, end_pos = nodes[start_node]['coordinates'], nodes[end_node]['coordinates']
                mid_pos = ((start_pos[1] + end_pos[1]) / 2, (start_pos[0] + end_pos[0]) / 2)
                line, = ax.plot([start_pos[1], end_pos[1]], [start_pos[0], end_pos[0]], color='green', linestyle='-', linewidth=1)
                edge_lines[edge] = line
                edge_texts[edge] = ax.text(mid_pos[0], mid_pos[1], '', fontsize=8, color='purple')

    timestep_text = ax.text(0.5, 0.98, '', transform=ax.transAxes, fontsize=10, color='black', ha='center', verticalalignment='top')


    return fig, ax, edge_texts, timestep_text, edge_lines

# The update_plot function remains the same
def update_plot(t, edges, vehicle_counts, edge_texts, timestep_text, num_minutes, edge_lines):
    cmap = cm.get_cmap('RdYlGn_r')
    norm = colors.Normalize(vmin=0, vmax=1)

    for edge, text in edge_texts.items():
        num_cars = int(vehicle_counts[edge][0])
        capacity = edges[edge]["capacity"]
        congestion = num_cars / capacity if capacity > 0 else 0
        congestion = min(max(congestion, 0), 1)

        text.set_text(f"{num_cars}/{capacity}")
        # Make the font size smaller
        text.set_fontsize(6)
        # Set color to orange
        text.set_color('orange')
        edge_color = cmap(norm(congestion))
        edge_lines[edge].set_color(edge_color)

    timestep_text.set_text(f"Timestep: {t}/{num_minutes}")
