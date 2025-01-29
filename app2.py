import folium
import streamlit as st
import random
import heapq
import time

import streamlit.components.v1 as components


def interpolate_position(start, end, fraction):
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction
    )

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

def create_map(station_locations, full_paths, start_positions, destination_positions, current_positions):
    m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=15)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Station
    for station in station_locations:
        folium.Marker(
            location=[station[0], station[1]],
            popup=f"Station: {station}",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    # Full paths
    for path in full_paths:
        folium.PolyLine(
            locations=[(pos[0], pos[1]) for pos in path],
            color="green",
            weight=2.5,
            opacity=0.8,
        ).add_to(m)

    # ตำแหน่ง person
    for i, pos in enumerate(current_positions):
        folium.Marker(
            location=[pos[0], pos[1]],
            popup=f"Agent {i + 1} Current Position",
            icon=folium.Icon(color="red", icon="user"),
        ).add_to(m)

    # Des
    for i, dest in enumerate(destination_positions):
        folium.Marker(
            location=[dest[0], dest[1]],
            popup=f"Destination {i + 1}",
            icon=folium.Icon(color="green", icon="flag"),
        ).add_to(m)

    return m

st.title("Traffic Simulation with Folium Map")

num_persons = st.number_input("Number of Persons:", min_value=1, value=5)
num_bikes_per_station = st.number_input("Number of Bikes per Station:", min_value=1, value=10)
max_time_step = st.number_input("Max Time Steps:", min_value=1, value=100)

station_locations = [
    (13.727868667926447, 100.77068388462067),  # เกกี
    (13.727668037525024, 100.76436460018158),  # rnp
    (13.729528421937207, 100.77500224113464),  # หอใน
    (13.7295518720667, 100.77996164560318),    # วิทยา
    (13.730672264410334, 100.78087896108627),  # ไอที
    (13.726521574796058, 100.77518731355667),  # วิดวะ
    (13.729210542172659, 100.77740550041199)]   # พระเทพ

# map boundaries
min_lat = min([coord[0] for coord in station_locations])
max_lat = max([coord[0] for coord in station_locations])
min_lon = min([coord[1] for coord in station_locations])
max_lon = max([coord[1] for coord in station_locations])

# สุ่มจุดเกิด / des
start_positions = []
destination_positions = []
for _ in range(num_persons):
    while True:
        person_pos = (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        if person_pos not in station_locations:  # Ensure agent not at a station
            start_positions.append(person_pos)
            break

    while True:
        dest_pos = (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        if dest_pos not in station_locations and dest_pos != person_pos:  # Ensure destination is not a station and not the same as the start
            destination_positions.append(dest_pos)
            break

# Build graph for A* Search
graph = {}
for i, station in enumerate(station_locations):
    graph[station] = {}
    for j, neighbor in enumerate(station_locations):
        if i != j:
            dist = heuristic(station, neighbor)
            graph[station][neighbor] = dist


if st.button("Start Simulation"):

    station_bikes = [num_bikes_per_station] * len(station_locations)
    
    # Find paths for person
    full_paths = []
    for person_pos, dest_pos in zip(start_positions, destination_positions):
        # Find the nearest station with at least one bike
        start_station = min(
            [station for station, bikes in zip(station_locations, station_bikes) if bikes > 0],
            key=lambda station: heuristic(person_pos, station),
        )
        # หา station ใกล้ des
        end_station = min(station_locations, key=lambda station: heuristic(dest_pos, station))
        
        # รวมพาธ
        complete_path = [person_pos]  # จุดเกิด
        complete_path.append(start_station)  # ไป sta ใกล้สุด
        complete_path.extend(a_star_search(graph, start_station, end_station))  # start_sta to end_sta
        complete_path.append(dest_pos)  # ไป des
        
        full_paths.append(complete_path)

        # Reduce bike count at the start station
        station_index = station_locations.index(start_station)
        station_bikes[station_index] -= 1


    map_display = st.empty()

    # Animate
    for step in range(max_time_step + 1):  # 0 to 100%
        # หาตำแหน่งปจบ.
        current_positions = []
        for path in full_paths:
            # Determine which segment of full path
            segment_length = len(path) - 1
            if segment_length == 0:  # Path has only one point
                current_positions.append(path[0])
                continue

            segment_index = int(step / max_time_step  * segment_length)
            
            # Interpolate position within the current segment
            start_point = path[segment_index]
            end_point = path[segment_index + 1] if segment_index + 1 < len(path) else start_point
            local_progress = (step / max_time_step * segment_length) - segment_index
            
            current_pos = interpolate_position(start_point, end_point, local_progress)
            current_positions.append(current_pos)


        traffic_map = create_map(
            station_locations, 
            full_paths, 
            start_positions, 
            destination_positions, 
            current_positions
        )

        map_display.write("### Traffic Map")
        map_html = folium.Figure().add_child(traffic_map)._repr_html_()
        components.html(map_html, height=500)

        # time-step
        time.sleep(1)