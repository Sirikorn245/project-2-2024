import random
import heapq
import folium
import streamlit as st

# A* algorithm implementation
def a_star_search(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor, cost in graph[current].items():
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Create map function
def create_map(station_locations, paths, agents_positions, destination_positions):
    m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=15)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    for station in station_locations:
        folium.Marker(
            location=[station[0], station[1]],
            popup=f"Station: {station}",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    for path in paths:
        folium.PolyLine(
            locations=[(pos[0], pos[1]) for pos in path],
            color="green",
            weight=2.5,
            opacity=0.8,
        ).add_to(m)

    for i, (agent_pos, dest_pos) in enumerate(zip(agents_positions, destination_positions)):
        folium.Marker(
            location=[agent_pos[0], agent_pos[1]],
            popup=f"Agent {i + 1} Position",
            icon=folium.Icon(color="red", icon="user"),
        ).add_to(m)

        folium.Marker(
            location=[dest_pos[0], dest_pos[1]],
            popup=f"Destination {i + 1}",
            icon=folium.Icon(color="green", icon="flag"),
        ).add_to(m)

    return m

# Streamlit UI
st.title("Traffic Simulation with Folium Map")

# Input from user
num_agents = st.number_input("Number of Agents:", min_value=1, value=5)
num_bikes_per_station = st.number_input("Number of Bikes per Station:", min_value=1, value=10)

# Example station locations
station_locations = [
    (13.727868667926447, 100.77068388462067),
    (13.727668037525024, 100.76436460018158),
    (13.729528421937207, 100.77500224113464),
    (13.7295518720667, 100.77996164560318),
    (13.730672264410334, 100.78087896108627),
    (13.726521574796058, 100.77518731355667),
    (13.729210542172659, 100.77740550041199),
]

# Define map boundaries
min_lat = min([coord[0] for coord in station_locations])
max_lat = max([coord[0] for coord in station_locations])
min_lon = min([coord[1] for coord in station_locations])
max_lon = max([coord[1] for coord in station_locations])

# Generate random agent positions and destinations (ensure not on station locations)
agents_positions = []
destination_positions = []
for _ in range(num_agents):
    while True:
        agent_pos = (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        if agent_pos not in station_locations:  # Ensure agent not at a station
            agents_positions.append(agent_pos)
            break

    while True:
        dest_pos = (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        if dest_pos not in station_locations and dest_pos != agent_pos:  # Ensure destination is not a station and not the same as the start
            destination_positions.append(dest_pos)
            break

# Build graph for A* Search (using distance between stations)
graph = {}
for i, station in enumerate(station_locations):
    graph[station] = {}
    for j, neighbor in enumerate(station_locations):
        if i != j:
            dist = heuristic(station, neighbor)
            graph[station][neighbor] = dist

# Find paths for each agent
paths = []
station_bikes = [num_bikes_per_station] * len(station_locations)
for agent_pos, dest_pos in zip(agents_positions, destination_positions):
    # Find the nearest station with at least one bike
    start_station = min(
        [station for station, bikes in zip(station_locations, station_bikes) if bikes > 0],
        key=lambda station: heuristic(agent_pos, station),
    )
    # Find the nearest station to the destination
    end_station = min(station_locations, key=lambda station: heuristic(dest_pos, station))
    # Perform A* search to find the path
    path = a_star_search(graph, start_station, end_station)
    paths.append(path)

    # Reduce bike count at the start station
    station_index = station_locations.index(start_station)
    station_bikes[station_index] -= 1

# Create and display the map
traffic_map = create_map(station_locations, paths, agents_positions, destination_positions)
st.write("### Traffic Map")
st.components.v1.html(folium.Figure().add_child(traffic_map)._repr_html_(), height=500)
