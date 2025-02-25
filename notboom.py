# 1️⃣ ส่วน Import Library
import folium
import streamlit as st
import random
import heapq
import json
import time
import streamlit.components.v1 as components
import math

import osmnx as ox
import networkx as nx

from geopy.distance import geodesic 
# สามารถคำนวณระยะทางจริงบนแผนที่ (โดยคำนึงถึงความโค้งของพื้นผิวโลก) ได้โดยใช้สูตร Haversine 
# หรือใช้ไลบรารีที่มีอยู่ เช่น geopy ซึ่งช่วยคำนวณระยะทางแบบ geodesic (ระยะทางบนพื้นผิวโค้งของโลก) ได้อย่างแม่นยำ


# ดึงข้อมูลถนนจาก OpenStreetMap
road = ox.graph_from_place("Lat Krabang, Bangkok, Thailand", network_type="all")



# 💌🧚‍♀️💗🌨🥡🍥 💌🧚 CBS 🥡🍥 💌🧚‍♀️💗🌨🥡🍥

# CBS Node class to represent a node in the constraint tree
class CBSNode:
    def __init__(self, constraints=None, solution=None, cost=0):
        self.constraints = constraints or {}  # Dictionary of agent_id: [(vertex/edge, timestep)]
        self.solution = solution or {}        # Dictionary of agent_id: path
        self.cost = cost                     # Sum of individual path costs
        self.grid_steps = {}                 # Dictionary to store grid steps for each agent

    def __lt__(self, other):
        # เปรียบเทียบโดยใช้ cost
        return self.cost < other.cost
    
# Function to detect conflicts between two agents' paths
def detect_conflicts(path1, path2):
    """
    Detects vertex and edge conflicts between two paths
    Returns list of conflicts: (timestep, type, location)
    """
    conflicts = []
    min_len = min(len(path1), len(path2))
    
    # Check vertex conflicts (agents at same location at same time)
    for t in range(min_len):
        if path1[t] == path2[t]:
            conflicts.append((t, 'vertex', path1[t]))
    
    # Check edge conflicts (agents swap positions)
    for t in range(min_len - 1):
        if path1[t] == path2[t+1] and path1[t+1] == path2[t]:
            conflicts.append((t, 'edge', (path1[t], path1[t+1])))
    
    return conflicts

def calculate_grid_steps(path, t_per_meter, simulation_time_step, max_time_steps):
    """
    Calculate the actual number of grid steps an agent takes along its path
    """
    # Calculate segment boundaries
    boundaries = compute_segment_boundaries(path, t_per_meter, simulation_time_step)
    
    # Get the total number of steps (last boundary)
    total_steps = boundaries[-1]
    
    # If total steps exceeds max_time_steps, cap it
    return min(total_steps, max_time_steps)

def find_path_with_constraints(graph, start, goal, constraints, t_per_meter, simulation_time_step):
    """
    Modified A* search that respects temporal constraints
    """
    open_set = [(0, start, 0)]  # (f_score, node, timestep)
    came_from = {}
    g_score = {(start, 0): 0}
    f_score = {(start, 0): heuristic(start, goal)}
    
    while open_set:
        current_f, current_node, t = heapq.heappop(open_set)
        
        if current_node == goal:
            path = []
            current_state = (current_node, t)
            while current_state in came_from:
                path.append(current_state[0])
                current_state = came_from[current_state]
            path.append(start)
            return path[::-1]
        
        # ⛰🌿🌻☀️☁️Use NetworkX methods to get neighbors⛰🌿🌻☀️☁️
        for neighbor in graph.neighbors(current_node):
            next_t = t + 1
            
            # Check if this move violates any constraints
            violates_constraint = False
            # Check vertex constraints (node, timestep)
            if (neighbor, next_t) in constraints:
                violates_constraint = True
            
            # Check edge constraints (from_node, to_node, timestep)
            if (current_node, neighbor, t) in constraints:
                violates_constraint = True
                
            if not violates_constraint:
                # ⛰🌿🌻☀️☁️ Get edge weight using NetworkX get_edge_data ⛰🌿🌻☀️☁️
                edge_data = graph.get_edge_data(current_node, neighbor)
                edge_weight = edge_data['weight'] if edge_data else 1

                neighbor_state = (neighbor, next_t)
                tentative_g_score = g_score[(current_node, t)] + edge_weight
                
                if neighbor_state not in g_score or tentative_g_score < g_score[neighbor_state]:
                    came_from[neighbor_state] = (current_node, t)
                    g_score[neighbor_state] = tentative_g_score
                    f_score[neighbor_state] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor_state], neighbor, next_t))
    
    return None  # No path found

def cbs_search(graph, start_positions, destination_positions, station_locations, t_per_meter, simulation_time_step, max_time_steps):
    """
    Main CBS algorithm implementation with grid steps tracking
    """
    root = CBSNode()

    # ⛰🌿🌻☀️☁️ สร้าง temporary graph ที่รวมจุดเริ่มต้นและจุดหมายของทุก agent ⛰🌿🌻☀️☁️
    temp_graph = graph.copy()

    # ⛰🌿🌻☀️☁️ เพิ่มจุดเริ่มต้นและจุดหมายของทุก agent เข้าไปในกราฟ ⛰🌿🌻☀️☁️
    for start, goal in zip(start_positions, destination_positions):
        temp_graph = add_temporary_node(temp_graph, start)
        temp_graph = add_temporary_node(temp_graph, goal)
    
    # Find initial paths for all agents ignoring conflicts
    for agent_id, (start, goal) in enumerate(zip(start_positions, destination_positions)):
        # Find nearest stations for pickup and dropoff
        start_station = min(station_locations, key=lambda s: heuristic(start, s))
        end_station = min(station_locations, key=lambda s: heuristic(goal, s))


        # ⛰🌿🌻☀️☁️ Create complete path using OSM ⛰🌿🌻☀️☁️
        # Create complete path using OSM
        path = []
        # 1. Start position to start station
        path.append(start)
        path.append(start_station)
        
        # 2. Route between stations using OSM
        osm_path = find_route_osm(road, start_station, end_station, 'cbs')
        if osm_path:
            path.extend(osm_path[1:])  # Skip first point as it's already added
        
        # 3. End station to goal
        path.append(goal)
        
        root.solution[agent_id] = path
        # Calculate and store grid steps for this agent
        root.grid_steps[agent_id] = calculate_grid_steps(path, t_per_meter, simulation_time_step, max_time_steps)
    
    # Initialize priority queue with root node
    open_list = [root]
    
    while open_list:
        node = heapq.heappop(open_list)
        
        # Convert paths to timelines for conflict detection
        agent_timelines = {}
        for agent_id, path in node.solution.items():
            timeline = compute_agent_timeline(path, t_per_meter, simulation_time_step, max_time_steps)
            agent_timelines[agent_id] = timeline
        
        # Find conflicts between all pairs of agents
        conflicts = []
        for i in range(len(agent_timelines)):
            for j in range(i + 1, len(agent_timelines)):
                conflicts.extend(detect_conflicts(agent_timelines[i], agent_timelines[j]))
        
        if not conflicts:
            return node.solution, agent_timelines, node.grid_steps
        
        # Handle first conflict
        conflict = conflicts[0]
        timestep, conflict_type, location = conflict
        
        # Create child nodes with new constraints
        for agent_id in range(2):  # Create two children with alternative constraints
            new_constraints = {k: v.copy() for k, v in node.constraints.items()}
            if agent_id not in new_constraints:
                new_constraints[agent_id] = set()
            
            if conflict_type == 'vertex':
                new_constraints[agent_id].add((tuple(location), timestep))
            else:  # edge conflict
                new_constraints[agent_id].add((tuple(location[0]), tuple(location[1]), timestep))
            
            child = CBSNode(new_constraints)
            
            # Find new path for constrained agent
            new_path = find_path_with_constraints(
                temp_graph,  # ใช้ temp_graph แทน graph
                start_positions[agent_id],
                destination_positions[agent_id],
                new_constraints[agent_id],
                t_per_meter,
                simulation_time_step
            )
            
            if new_path:
                child.solution = node.solution.copy()
                child.solution[agent_id] = new_path
                child.cost = sum(len(path) for path in child.solution.values())
                child.grid_steps = node.grid_steps.copy()
                child.grid_steps[agent_id] = calculate_grid_steps(new_path, t_per_meter, simulation_time_step, max_time_steps)
                heapq.heappush(open_list, child)
    
    return None, None, None  # No solution found

# 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥



#! สิ่งที่ทำเพิ่ม
# - คำนวณระยะทางจริง
# - กำหนดความเร็วเดินคงที่
# - แบ่ง time step ตามเวลาที่ใช้เดินจริง

def interpolate_position(start, end, fraction):
    """คำนวณตำแหน่งระหว่าง start กับ end ตาม fraction (0-1)
       คืนค่าเป็น (lat, lon) tuple"""
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction
    )



# 💌🧚‍♀️💗🌨🥡🍥 💌🧚 new 🥡🍥 💌🧚‍♀️💗🌨🥡🍥

def add_temporary_node(graph, point):
    """ ⛰🌿🌻☀️☁️
    Add a temporary node to a NetworkX graph and connect it to its nearest neighbor
    
    Args:
        graph (nx.Graph): NetworkX graph object
        point (tuple): Coordinates of the new point (lat, lon)
    
    Returns:
        nx.Graph: Copy of graph with new temporary node added
    ⛰🌿🌻☀️☁️ """

    # Create a copy of the graph
    temp_graph = graph.copy()
    
    # If point already exists as a node, return the graph as is
    if point in temp_graph.nodes():
        return temp_graph

    # หา node ใกล้เคียงที่สุด
    nearest = min(temp_graph.nodes(), key=lambda node: geodesic(point, node).meters)

    # คำนวณระยะทางไป node นั้น
    distance = geodesic(point, nearest).meters

    # Add new node and edge
    temp_graph.add_node(point)
    temp_graph.add_edge(point, nearest, weight=distance, length=distance)

    return temp_graph

# 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥



def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# 2️⃣ ฟังก์ชันช่วยเหลือสำหรับ A* Search
def heuristic(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# 💌🧚‍♀️💗🌨🥡🍥 💌🧚 new 🥡🍥 💌🧚‍♀️💗🌨🥡🍥
def create_station_graph(station_locations):
    """
    ⛰🌿🌻☀️☁️
    สร้าง NetworkX graph object สำหรับเชื่อมต่อระหว่างสถานี
    
    Args:
        station_locations (list): List of station coordinates (lat, lon)
    
    Returns:
        nx.Graph: Complete graph connecting all stations
        
    ⛰🌿🌻☀️☁️
    """
    G = nx.Graph()
    
    # เพิ่ม nodes (สถานี)
    for station in station_locations:
        G.add_node(station)
    
    # เพิ่ม edges (เส้นทางระหว่างสถานี)
    for i, station1 in enumerate(station_locations):
        for station2 in station_locations[i+1:]:  # Avoid duplicate edges
            distance = geodesic(station1, station2).meters
            G.add_edge(station1, station2, weight=distance, length=distance)
    
    return G
# 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥


# 🧪🧪🧪 ใช้ astar_path ของ networkX 🧪🧪🧪
def a_star_search(graph, start, goal):
    """
    ค้นหาเส้นทางด้วย A* algorithm โดยใช้ NetworkX
    """
    try:
        # ใช้ astar_path จาก NetworkX
        path = nx.astar_path(graph, start, goal, weight='length')
        return path
    except nx.NetworkXNoPath:
        print(f"ไม่พบเส้นทางระหว่าง {start} และ {goal}")
        return []


# 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
# M* Node class for search
class MStarNode:
    def __init__(self, positions, g_score=0, parent=None):
        self.positions = tuple(positions)  # Current positions of all agents
        self.g_score = g_score            # Cost from start to current
        self.parent = parent              # Parent node
        self.colliding_agents = set()     # Set of agents involved in collision

    def __lt__(self, other):
        return self.g_score < other.g_score

def m_star_search(graph, start_positions, goal_positions, t_per_meter, simulation_time_step, max_time_steps):
    """
    M* search implementation for multi-agent path finding
    
    Args:
        graph: NetworkX graph of the road network
        start_positions: List of start positions for each agent
        goal_positions: List of goal positions for each agent
        t_per_meter: Time per meter for movement
        simulation_time_step: Time step duration
        max_time_steps: Maximum number of time steps
        
    Returns:
        Dictionary of paths for each agent, timelines, and grid steps
    """
    num_agents = len(start_positions)

    # ตรวจสอบว่ามีตำแหน่งเริ่มต้นและเป้าหมายที่ถูกต้อง
    if num_agents == 0 or len(goal_positions) != num_agents:
        print("Invalid number of start/goal positions")
        return None, None, None
    
    # ตรวจสอบว่าตำแหน่งทั้งหมดอยู่ในกราฟ
    for pos in start_positions + goal_positions:
        if pos not in graph.nodes():
            print(f"Position {pos} not in graph")
            try:
                # พยายามหาโหนดที่ใกล้ที่สุด
                nearest = min(graph.nodes(), key=lambda node: heuristic(pos, node))
                print(f"Nearest node is {nearest}")
            except:
                print("Cannot find nearest node")
            return None, None, None
    
    # Helper function to check collisions between agents
    def check_collisions(positions):
        collisions = set()
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if positions[i] == positions[j]:
                    collisions.add(i)
                    collisions.add(j)
        return collisions

    # Helper function to get neighbors for a specific agent
    def get_agent_neighbors(pos, agent_id):
        neighbors = list(graph.neighbors(pos))
        neighbors.append(pos)  # Allow waiting at current position
        return neighbors

    # Heuristic function (manhattan distance)
    def h_score(positions):
        return sum(heuristic(pos, goal_positions[i]) for i, pos in enumerate(positions))

    # Initialize start node
    start_node = MStarNode(start_positions)
    start_node.f_score = h_score(start_positions)

    # Priority queue for open set
    open_set = [start_node]
    closed_set = set()
    
    # Initialize path tracking
    came_from = {start_node.positions: None}
    g_score = {start_node.positions: 0}
    f_score = {start_node.positions: h_score(start_positions)}

    while open_set:
        current = heapq.heappop(open_set)
        
        # Check if goal reached
        if all(current.positions[i] == goal_positions[i] for i in range(num_agents)):
            # Reconstruct paths
            paths = {i: [] for i in range(num_agents)}
            node = current
            while node:
                for i in range(num_agents):
                    paths[i].insert(0, node.positions[i])
                node = node.parent

            # Convert paths to timelines and calculate grid steps
            timelines = {}
            grid_steps = {}
            for agent_id, path in paths.items():
                timeline = compute_agent_timeline(path, t_per_meter, simulation_time_step, max_time_steps)
                timelines[agent_id] = timeline
                grid_steps[agent_id] = len(timeline)

            return paths, timelines, grid_steps

        if current.positions in closed_set:
            continue

        closed_set.add(current.positions)

        # Generate neighbor states
        for agent_id in range(num_agents):
            if agent_id in current.colliding_agents:
                neighbor_positions = list(current.positions)
                current_pos = current.positions[agent_id]
                
                for next_pos in get_agent_neighbors(current_pos, agent_id):
                    neighbor_positions[agent_id] = next_pos
                    neighbor_tuple = tuple(neighbor_positions)
                    
                    if neighbor_tuple in closed_set:
                        continue

                    # Calculate new g_score
                    tentative_g_score = current.g_score + 1

                    if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                        # Create new node
                        neighbor_node = MStarNode(neighbor_positions, tentative_g_score, current)
                        
                        # Check for collisions
                        collisions = check_collisions(neighbor_positions)
                        neighbor_node.colliding_agents = collisions
                        
                        # Update scores
                        g_score[neighbor_tuple] = tentative_g_score
                        f_score[neighbor_tuple] = tentative_g_score + h_score(neighbor_positions)
                        
                        heapq.heappush(open_set, neighbor_node)

    return None, None, None  # No solution found

# Function to find route using M*
def find_route_m_star(graph, start_positions, goal_positions, t_per_meter, simulation_time_step, max_time_steps):
    try:
        # ตรวจสอบว่าโหนดเริ่มต้นและเป้าหมายอยู่ในกราฟหรือไม่
        valid_start_positions = []
        valid_goal_positions = []
        
        for i, (start, goal) in enumerate(zip(start_positions, goal_positions)):
            if start not in graph.nodes():
                print(f"Start position {start} for agent {i} not in graph. Skipping agent.")
                continue
                
            if goal not in graph.nodes():
                print(f"Goal position {goal} for agent {i} not in graph. Skipping agent.")
                continue
                
            valid_start_positions.append(start)
            valid_goal_positions.append(goal)
        
        if not valid_start_positions or not valid_goal_positions:
            print("No valid agent positions found. Cannot proceed with M* search.")
            return {}, {}, {}
            
        # เรียกใช้ m_star_search เพื่อหาเส้นทาง
        paths, timelines, grid_steps = m_star_search(
            graph, 
            valid_start_positions, 
            valid_goal_positions, 
            t_per_meter, 
            simulation_time_step, 
            max_time_steps
        )
        
        if paths is None:
            print("M* search could not find a solution")
            # ถ้าหาเส้นทางไม่ได้ ลองหาเส้นทางแยกสำหรับแต่ละ agent
            paths = {}
            timelines = {}
            grid_steps = {}
            
            for i, (start, goal) in enumerate(zip(valid_start_positions, valid_goal_positions)):
                try:
                    # ใช้ A* แทนสำหรับแต่ละ agent
                    path = a_star_search(graph, start, goal)
                    
                    if path:
                        paths[i] = path
                        timeline = compute_agent_timeline(path, t_per_meter, simulation_time_step, max_time_steps)
                        timelines[i] = timeline
                        grid_steps[i] = len(timeline)
                    else:
                        print(f"Could not find path for agent {i} from {start} to {goal}")
                        paths[i] = [start, goal]  # เส้นทางเริ่มต้นถึงเป้าหมายโดยตรง
                        timeline = [start] * max_time_steps  # คงตำแหน่งเริ่มต้น
                        timelines[i] = timeline
                        grid_steps[i] = 0
                except Exception as e:
                    print(f"Error finding path for agent {i}: {e}")
                    paths[i] = [start, goal]
                    timelines[i] = [start] * max_time_steps
                    grid_steps[i] = 0
            
            return paths, timelines, grid_steps
                
        # คงเส้นทางไว้ตามที่ได้จาก m_star_search
        return paths, timelines, grid_steps
        
    except Exception as e:
        print(f"Error in find_route_m_star: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}, {}
# 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°


#🍤🍤🍤🍤🍤🍤หาเส้นทางตามถนน osmnx🍤🍤🍤🍤🍤🍤
def find_route_osm(road, start_latlon, end_latlon, algorithm):
    # แปลงพิกัด latitude, longitude ให้เป็นโหนดที่ใกล้ที่สุดในกราฟถนน
    start_node = ox.distance.nearest_nodes(road, start_latlon[1], start_latlon[0])
    end_node = ox.distance.nearest_nodes(road, end_latlon[1], end_latlon[0])


    def get_edge_weight(current, neighbor):
        """Helper function to safely get edge weight"""
        edge_data = road.get_edge_data(current, neighbor)
        
        # Handle different edge data formats
        if isinstance(edge_data, dict):
            return edge_data.get('length', 1)
        elif isinstance(edge_data, set):
            # If it's a set, try to get the first item's length
            if edge_data:
                first_item = next(iter(edge_data))
                if isinstance(first_item, dict):
                    return first_item.get('length', 1)
            return 1
        elif isinstance(edge_data, list):
            # If it's a list, try to get the first item's length
            if edge_data:
                return edge_data[0].get('length', 1)
        return 1  # Default weight if no length found


    # 📌 เลือกอัลกอริธึมที่ใช้
    if algorithm == 'a_star':
        # Use A* algorithm
        try:
            route = nx.astar_path(road, start_node, end_node, weight='length')
        except nx.NetworkXNoPath:
            print(f"A*: No path found between {start_latlon} and {end_latlon}")
            return []
            
    elif algorithm == 'cbs':
        # Use CBS-specific pathfinding
        try:
            # Initialize priority queue with starting node
            open_set = [(0, start_node)]
            came_from = {}
            g_score = {node: float('inf') for node in road.nodes}
            g_score[start_node] = 0
            
            while open_set:
                current_cost, current = heapq.heappop(open_set)
                
                if current == end_node:
                    # Reconstruct path
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start_node)
                    route = path[::-1]
                    break
                    
                for neighbor in road.neighbors(current):
                    # Get edge weight using helper function
                    weight = get_edge_weight(current, neighbor)
                    
                    tentative_g_score = g_score[current] + weight
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(
                            (road.nodes[neighbor]['y'], road.nodes[neighbor]['x']),
                            end_latlon
                        )
                        heapq.heappush(open_set, (f_score, neighbor))
            else:
                print(f"CBS: No path found between {start_latlon} and {end_latlon}")
                return []
        
        except Exception as e:
            print(f"CBS: Error finding path: {e}")
            return []
    
    elif algorithm == 'm_star':
        try:
            paths, timelines, grid_steps = find_route_m_star(
                road,
                [start_node],
                [end_node],
                t_per_meter=0.1,
                simulation_time_step=1,
                max_time_steps=100
            )
            if paths:
                route = list(paths.values())[0]  # Get first (and only) path
            else:
                print(f"M*: No path found between {start_latlon} and {end_latlon}")
                return []
        except Exception as e:
            print(f"M*: Error finding path: {e}")
            return []

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # แปลง route จาก node ID เป็นพิกัด latitude, longitude
    return [(road.nodes[node]["y"], road.nodes[node]["x"]) for node in route]



# 💌🧚‍♀️💗🌨🥡🍥 💌🧚 new 🥡🍥 💌🧚‍♀️💗🌨🥡🍥

# ฟังก์ชันใหม่สำหรับคำนวณ timeline ของ agent โดยใช้ระยะทางจริง
def compute_segment_boundaries(path, t_per_meter, simulation_time_step):
    """
    - ฟังก์ชันนี้ใช้คำนวณจุดแบ่งเวลา (time boundaries) สำหรับแต่ละส่วน (segment) ของเส้นทาง 
    - แปลงระยะทางจริงให้เป็นจำนวน time steps ในการจำลอง
    path: เส้นทางที่เป็นชุดของพิกัด
    t_per_meter: เวลาที่ใช้ต่อระยะทาง 1 เมตร
    simulation_time_step: ขนาดของ time step ในการจำลอง
    """
    # t_per_meter = 0.1           # เวลาที่ใช้เดิน 1 เมตร (วินาที)
    # simulation_time_step = 1    # 1 วินาทีต่อ time step

    boundaries = [0]
    total_steps = 0
    for i in range(len(path)-1):
        d = geodesic(path[i], path[i+1]).meters #  สำหรับแต่ละ segment คำนวณระยะทางจริงระหว่างจุดด้วย geodesic().meters
        
        seg_time = d * t_per_meter # คำนวณเวลาที่ใช้สำหรับ segment นั้น

        seg_steps = max(1, int(round(seg_time / simulation_time_step))) # แปลงเวลาเป็นจำนวน time steps โดยปัดเศษและกำหนดให้มีอย่างน้อย 1 step
        total_steps += seg_steps # เก็บสะสมจำนวน steps ทั้งหมดไว้ใน total_steps

        boundaries.append(total_steps) 

    return boundaries
    # คืนค่า list boundaries ที่เก็บค่าสะสมของ time steps ณ จุดสิ้นสุดของแต่ละ segment
    # ตัวอย่างเช่น [0, 5, 12, 18] หมายถึง segment แรกใช้ 5 steps, segment ที่สองใช้ 7 steps, และ segment ที่สามใช้ 6 steps

# 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥 💌🧚‍♀️💗🌨🥡🍥



def compute_agent_timeline(path, t_per_meter, simulation_time_step, max_time_steps):
    """
    คำนวณ timeline ของ agent ตาม path ที่ให้
      - ใช้ geodesic distance ในการคำนวณเวลาเดินแต่ละ segment
      - simulation_time_step คือความยาวของ time step (วินาที)
      - max_time_steps คือจำนวน time step สูงสุดที่ agent จะเดิน (เมื่อครบแล้วจะหยุด)
    คืนค่า timeline เป็น list ของตำแหน่ง [lat, lon] สำหรับแต่ละ time step
    """
    timeline = []
    total_steps = 0
    for i in range(len(path)-1):
        start = tuple(path[i]) if isinstance(path[i], list) else path[i]  # แปลงเป็น tuple
        end = tuple(path[i+1]) if isinstance(path[i+1], list) else path[i+1]
        d = geodesic(start, end).meters
        seg_time = d * t_per_meter  # เวลาเดินในหน่วยวินาที
        seg_steps = max(1, int(round(seg_time / simulation_time_step)))
        for step in range(seg_steps):
            # หากจำนวน time steps รวมเกิน max_time_steps ให้หยุดทันที
            if total_steps >= max_time_steps:
                return timeline[:max_time_steps]
            fraction = step / seg_steps
            timeline.append(interpolate_position(start, end, fraction))
            total_steps += 1
    # เติมตำแหน่งสุดท้าย (จุดหมายปลายทาง) ถ้ายังไม่ครบ max_time_steps
    while len(timeline) < max_time_steps:
        timeline.append(path[-1])
    return timeline[:max_time_steps]
# 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
# ส่วน CBS: นิยาม class และฟังก์ชันสำหรับ Conflict-Based Search



# 3️⃣ ฟังก์ชัน create_map(...) → สร้างแผนที่และฝัง JavaScript animation
#    ในส่วนนี้ เราจะไม่สร้าง station markers ด้วย Python แต่จะสร้างและอัปเดตใน JavaScript
def create_map(full_paths, agents_positions, station_locations, station_bikes_timeline, destination_positions):
    # สร้างแผนที่พื้นฐาน
    m = folium.Map(location=[13.728, 100.775], zoom_start=15)

    # วาดเส้นทางของ agent แต่ละคน (full_paths)
    for path in full_paths:
        folium.PolyLine(path, color='yellow', weight=2).add_to(m)


    # Marker Destination
    # destination_positions
    for i, dest in enumerate(destination_positions):
        folium.Marker(
            location=[dest[0], dest[1]],
            popup=f"Destination {i + 1}",
            icon=folium.Icon(color="gray", icon="flag"),
        ).add_to(m)



    # แปลงตัวแปร Python ให้เป็น JSON สำหรับ JavaScript
    agents_positions_json = json.dumps(agents_positions)
    station_locations_json = json.dumps(station_locations)
    station_bikes_timeline_json = json.dumps(station_bikes_timeline)

    map_var = m.get_name()


   
    
    custom_js = f"""
    <script>
    window.addEventListener('load', function() {{
        // ข้อมูล timeline ของตำแหน่ง agent และจำนวนจักรยานในแต่ละสถานี
        var agentsPositions = {agents_positions_json};
        var stationLocations = {station_locations_json};
        var stationBikesTimeline = {station_bikes_timeline_json};
        var mapObj = window["{map_var}"];
        
        // สร้าง icon สำหรับ agent และ station
        var redIcon = L.icon({{
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.4/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        }});
        var greenIcon = L.icon({{
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.3.4/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        }});
        
        // สร้าง marker สำหรับ agent โดยเริ่มต้นจากตำแหน่งแรกใน agentsPositions
        var agentMarkers = [];
        for (var i = 0; i < agentsPositions.length; i++) {{
            var marker = L.marker(agentsPositions[i][0], {{icon: redIcon}}).addTo(mapObj);
            marker.bindPopup("Agent " + (i+1));
            agentMarkers.push(marker);
        }}
        
        // สร้าง marker สำหรับ station โดยอิงจาก stationLocations และ stationBikesTimeline[0]
        var stationMarkers = [];
        for (var i = 0; i < stationLocations.length; i++) {{
            var marker = L.marker(stationLocations[i], {{icon: greenIcon}}).addTo(mapObj);
            marker.bindPopup("Station: " + stationLocations[i] + "<br>Bikes Available: " + stationBikesTimeline[0][i]);
            stationMarkers.push(marker);
        }}

        var timeStep = 0;
        var maxStep = agentsPositions[0].length;
        var interval = null;

        function updateMarkers() {{
            // อัปเดตตำแหน่งของ agent
            for (var i = 0; i < agentMarkers.length; i++) {{
                agentMarkers[i].setLatLng(agentsPositions[i][timeStep]);
            }}
            // อัปเดต popup ของ station ให้แสดงจำนวนจักรยานใน time step ปัจจุบัน
            for (var i = 0; i < stationMarkers.length; i++) {{
                stationMarkers[i].setPopupContent("Station: " + stationLocations[i] + "<br>Bikes Available: " + stationBikesTimeline[timeStep][i]);
            }}
            document.getElementById("timeStepDisplay").innerText = "Time Step: " + timeStep;
        }}

        function startAnimation() {{
            if (!interval) {{
                interval = setInterval(function() {{
                    if (timeStep < maxStep - 1) {{
                        timeStep++;
                        updateMarkers();
                    }} else {{
                        clearInterval(interval);
                        interval = null;
                    }}
                }}, 100);
            }}
        }}

        function pauseAnimation() {{
            clearInterval(interval);
            interval = null;
        }}

        function resetAnimation() {{
            pauseAnimation();
            timeStep = 0;
            updateMarkers();
        }}

        document.getElementById("startBtn").addEventListener("click", startAnimation);
        document.getElementById("pauseBtn").addEventListener("click", pauseAnimation);
        document.getElementById("resetBtn").addEventListener("click", resetAnimation);

        updateMarkers();
    }});
    </script>
    """

    control_html = """
    <div style="text-align:center; margin-top: 10px;">
        <button id="startBtn">Start</button>
        <button id="pauseBtn">Pause</button>
        <button id="resetBtn">Reset</button>
        <p id="timeStepDisplay">Time Step: 0</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(control_html + custom_js))
    return m

# 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
# 4️⃣ ฟังก์ชัน run_simulation() → รันการจำลอง
def run_simulation():
    num_persons = st.session_state.num_persons
    max_time_step = st.session_state.max_time_step
    num_bikes_per_station = st.session_state.num_bikes

    # ตั้งค่าความเร็วเดิน (เวลาต่อเมตร) และ simulation time step (วินาที)
    t_per_meter = 0.1           # กำหนดเวลา (วินาที) ที่ใช้เดิน 1 เมตร
    simulation_time_step = 1    # 1 วินาทีต่อ time step

    # กำหนดสถานีจักรยาน
    station_locations = [
    (13.727868667926447, 100.77068388462067),  # เกกี
    (13.727668037525024, 100.76436460018158),  # rnp
    (13.729528421937207, 100.77500224113464),  # หอใน
    (13.7295518720667, 100.77996164560318),    # วิทยา
    (13.730672264410334, 100.78087896108627),  # ไอที
    (13.726521574796058, 100.77518731355667),  # วิดวะ
    (13.729210542172659, 100.77740550041199)   # พระเทพ
    ]

    # map boundaries
    min_lat = min(x[0] for x in station_locations)
    max_lat = max(x[0] for x in station_locations)
    min_lon = min(x[1] for x in station_locations)
    max_lon = max(x[1] for x in station_locations)

    

    # สุ่มตำแหน่งเริ่มต้นและปลายทางของ agent
    start_positions = [
        (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        for _ in range(num_persons)
    ]
    destination_positions = [
        (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        for _ in range(num_persons)
    ]

    # สร้างกราฟสำหรับ A* Search ระหว่างสถานี
    graph = create_station_graph(station_locations)

    # กำหนดจำนวนจักรยานเริ่มต้นในแต่ละสถานี
    initial_station_bikes = [num_bikes_per_station] * len(station_locations)

    # 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
    #! ส่วน A* Alogorithm
    full_paths_a_star = [] 
    rental_events = []  
    return_events = []  
    agent_grid_steps = []  # จำนวน grid steps ที่แต่ละ agent เดิน (ก่อน padding)

    for start_pos, dest_pos in zip(start_positions, destination_positions):
        print(f"A*: Agent เริ่มที่ {start_pos} และต้องไปที่ {dest_pos}")

        # จัดเรียงสถานีทั้งหมดตามระยะทางจาก agent (ใกล้ -> ไกล) เพราะจะเช็คสถานีที่ใกล้ที่สุดก่อน , ถ้าสถานีนั้น ไม่มีจักรยาน → ให้เลือก สถานีที่ใกล้รองลงมา แต่ถ้ายังไม่มี → เลือกสถานีรองที่เหลือไปเรื่อย ๆ
        sorted_stations = sorted(station_locations, key=lambda s: heuristic(start_pos, s))

        # ค้นหาสถานีที่มีจักรยานเหลืออยู่
        start_station = None
        for station in sorted_stations:
            station_index = station_locations.index(station) # หา index ของสถานีนี้
            if initial_station_bikes[station_index] > 0: # ตรวจสอบว่ามีจักรยานหรือไม่
                start_station = station
                break
        if start_station is None:
            start_station = sorted_stations[0]

        # สำหรับ drop-off (สถานีปลายทาง) เราเลือก 𝘀𝘁𝗮𝘁𝗶𝗼𝗻 ที่ใกล้ 𝗱𝗲𝘀𝘁𝗶𝗻𝗮𝘁𝗶𝗼𝗻 มากที่สุด
        end_station = min(station_locations, key=lambda s: heuristic(dest_pos, s))

        # ขั้นตอน สร้างเส้นทาง: จุดเริ่มต้น → สถานีเช่า → (เส้นทาง A* ระหว่างสถานี) → จุดหมายปลายทาง
        # 1. จุดเริ่มต้น → สถานีเช่า
        # complete_path : list ที่เก็บเส้นทางของ agent ตั้งแต่เริ่มต้น จุดแรกคือ ตำแหน่งเริ่มต้นของ agent, จุดที่สองคือ สถานีเช่าจักรยาน (start_station) ที่เลือกไว้
        complete_path = [start_pos]

        # 2. สถานีเช่า → สถานีคืน (ด้วย a* alogorithm ⛩️)
        #! 🚩 เอาแบบนี้ไปก่อน เราต้องใช้ a* algotithm หาแค่สถานียืมไปสถานีคืน
        # ใช้เส้นทางที่ได้จาก OpenStreetMap จริง
        osm_path = find_route_osm(road, start_station, end_station, 'a_star')
        complete_path.extend(osm_path)

        # 3. สถานีคืน → จุดหมายปลายทาง
        complete_path.append(dest_pos)

        # full_paths เป็น list ที่เก็บเส้นทางของ agent ทุกคน
        full_paths_a_star.append(complete_path)


        # คำนวณเวลาที่ใช้เดินในแต่ละ segment
        # คำนวณ segment boundaries ของ complete_path (ในหน่วย time step)
        boundaries = compute_segment_boundaries(complete_path, t_per_meter, simulation_time_step)

        # จำนวน grid steps ที่ agent เดินจริง (ก่อนถึงจุดสิ้นสุดของ path)
        active_steps = boundaries[-1]

        # หาก active_steps เกิน max_time_step ให้ถือว่าเดิน max_time_step
        if active_steps > max_time_step:
            active_steps = max_time_step
        agent_grid_steps.append(active_steps)

        # rental event: เมื่อ agent ถึงสถานีเช่า (complete_path[1])
        rental_time = boundaries[1] if boundaries[1] < max_time_step else max_time_step - 1

        # return event: เมื่อ agent ถึงสถานีคืน (complete_path[-2])
        return_time = boundaries[-2] if boundaries[-2] < max_time_step else max_time_step - 1

        # บันทึก event
        station_index = station_locations.index(start_station)
        end_station_index = station_locations.index(end_station)
        rental_events.append((rental_time, station_index))
        return_events.append((return_time, end_station_index))

        # ปรับจำนวนจักรยานในสถานีทันที
        initial_station_bikes[station_index] -= 1
        initial_station_bikes[end_station_index] += 1

    st.write("### a_star: จำนวน Grid Steps ที่แต่ละ Agent เดิน")
    for idx, steps in enumerate(agent_grid_steps):
        st.write(f"Agent {idx+1}: {steps} grid steps")

    # คำนวณตำแหน่งของ agent ในแต่ละ time step โดยใช้ compute_agent_timeline
    agents_positions_a_star = []
    for path in full_paths_a_star:
        timeline = compute_agent_timeline(path, t_per_meter, simulation_time_step, max_time_step)
        agents_positions_a_star.append(timeline)
    


    # กำหนด CSS
    map_style = """
    <style>
        .stVerticalBlock { width: 80% !important; }
        .st-emotion-cache-17vd2cm { width: 80% !important; }
        [data-testid="stAppViewContainer"] { }
        [data-testid="stIFrame"] { width: 80% !important; height: 650px !important; }
        [data-testid="stMainBlockContainer"] { width: 80% !important; max-width: 80% !important; }
    </style>
    """
    st.markdown(map_style, unsafe_allow_html=True)
    

    # ส่งข้อมูล A* ไปสร้าง map
    st.write("### A* Traffic Simulation Map")
    traffic_map_a_star = create_map(full_paths_a_star, agents_positions_a_star, station_locations, 
                                 [[num_bikes_per_station]*len(station_locations) for _ in range(max_time_step)],
                                 destination_positions)
    with st.container():
        components.html(traffic_map_a_star._repr_html_(), height=600)



    # 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
    # Run CBS algorithm
    cbs_solution, cbs_timelines, cbs_grid_steps = cbs_search(
        graph,
        start_positions,
        destination_positions,
        station_locations,
        t_per_meter,
        simulation_time_step,
        max_time_step
    )

    if cbs_solution:
        # Display grid steps for CBS
        st.write("### CBS: จำนวน Grid Steps ที่แต่ละ Agent เดิน")
        for agent_id, steps in cbs_grid_steps.items():
            st.write(f"Agent {agent_id+1}: {steps} grid steps")
        
        # Convert CBS solution to format needed for visualization
        full_paths_cbs = list(cbs_solution.values())
        agents_positions_cbs = list(cbs_timelines.values())
        
        # Create CBS visualization
        st.write("### CBS Traffic Simulation Map")
        traffic_map_cbs = create_map(
            full_paths_cbs,
            agents_positions_cbs,
            station_locations,
            [[num_bikes_per_station]*len(station_locations) for _ in range(max_time_step)],
            destination_positions
        )
        with st.container():
            components.html(traffic_map_cbs._repr_html_(), height=600)
    else:
        st.write("CBS could not find a valid solution with the given constraints")



    # 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
    # Run M* algorithm

    # สร้างกราฟสำหรับ M* โดยเฉพาะ
    road_graph = nx.Graph()
    
    # เพิ่มโหนดจาก OSM road network
    for node, data in road.nodes(data=True):
        road_graph.add_node(node)
    
    # เพิ่ม edges
    for u, v, data in road.edges(data=True):
        # ใช้ weight เป็น length ถ้ามี
        if 'length' in data:
            road_graph.add_edge(u, v, weight=data['length'], length=data['length'])
        else:
            # ถ้าไม่มี length ให้คำนวณจากพิกัด
            u_coords = (road.nodes[u]['y'], road.nodes[u]['x']) if 'y' in road.nodes[u] else None
            v_coords = (road.nodes[v]['y'], road.nodes[v]['x']) if 'y' in road.nodes[v] else None
            
            if u_coords and v_coords:
                try:
                    distance = geodesic(u_coords, v_coords).meters
                    road_graph.add_edge(u, v, weight=distance, length=distance)
                except:
                    road_graph.add_edge(u, v, weight=1, length=1)
            else:
                road_graph.add_edge(u, v, weight=1, length=1)
    
    # แปลงตำแหน่งเริ่มต้นและเป้าหมายเป็นโหนดที่มีในกราฟ
    temp_start_positions = []
    temp_goal_positions = []

    for start_pos in start_positions:
        try:
            start_node = ox.distance.nearest_nodes(road, start_pos[1], start_pos[0])
            temp_start_positions.append(start_node)
        except Exception as e:
            print(f"Error finding nearest node for start position {start_pos}: {e}")
            continue
            
    for goal_pos in destination_positions:
        try:
            goal_node = ox.distance.nearest_nodes(road, goal_pos[1], goal_pos[0])
            temp_goal_positions.append(goal_node)
        except Exception as e:
            print(f"Error finding nearest node for goal position {goal_pos}: {e}")
            continue

    # ตรวจสอบว่ามีตำแหน่งเริ่มต้นและเป้าหมายที่ถูกต้องหรือไม่
    if len(temp_start_positions) == 0 or len(temp_goal_positions) == 0:
        st.write("ไม่สามารถหาโหนดที่เหมาะสมสำหรับตำแหน่งเริ่มต้นหรือเป้าหมายได้")
    elif len(temp_start_positions) != len(temp_goal_positions):
        st.write("จำนวนตำแหน่งเริ่มต้นและเป้าหมายไม่เท่ากัน")
    else:
        # เรียกใช้ M* algorithm
        m_star_paths, m_star_timelines, m_star_grid_steps = find_route_m_star(
            road_graph,
            temp_start_positions,
            temp_goal_positions,
            t_per_meter,
            simulation_time_step,
            max_time_step
        )
    
    print("🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤🍤", m_star_paths)

    if m_star_paths and len(m_star_paths) > 0:
        # Display grid steps for M*
        st.write("### M*: จำนวน Grid Steps ที่แต่ละ Agent เดิน")
        for agent_id, steps in m_star_grid_steps.items():
            st.write(f"Agent {agent_id+1}: {steps} grid steps")

        # แปลงเส้นทางจาก node ids เป็นพิกัด lat-lon
        full_paths_m_star = []
        agents_positions_m_star = []

        for agent_id in sorted(m_star_paths.keys()):
            path = m_star_paths[agent_id]
            full_path = []

            for node in path:
                try:
                    full_path.append((road.nodes[node]['y'], road.nodes[node]['x']))
                except:
                    # ใช้ node เดิมถ้ามันเป็นพิกัดอยู่แล้ว
                    if isinstance(node, tuple):
                        full_path.append(node)

            full_paths_m_star.append(full_path)
            agents_positions_m_star.append(m_star_timelines[agent_id])
        
        # Create M* visualization
        st.write("### M* Traffic Simulation Map")
        traffic_map_m_star = create_map(
            full_paths_m_star,
            agents_positions_m_star,
            station_locations,
            [[num_bikes_per_station]*len(station_locations) for _ in range(max_time_step)],
            destination_positions
        )
        with st.container():
            components.html(traffic_map_m_star._repr_html_(), height=600)
    else:
        st.write("M* could not find a valid solution with the given constraints")

    # # ปรับปรุงจำนวนจักรยานในแต่ละสถานีตลอด timeline
    # station_bikes_timeline = [[num_bikes_per_station] * len(station_locations) for _ in range(max_time_step)]
    # for t in range(max_time_step):
    #     for rental_time, station_index in rental_events:
    #         if t >= rental_time:
    #             station_bikes_timeline[t][station_index] -= 1
    #     for return_time, station_index in return_events:
    #         if t >= return_time:
    #             station_bikes_timeline[t][station_index] += 1

    # print("Station bikes at time 0:", station_bikes_timeline[0])
    # print("Station bikes at final time:", station_bikes_timeline[-1])



  

# 🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°🪩🫧🍸🥂🫧✧˖°
# 5️⃣ ส่วนอินพุต Streamlit
st.title("Traffic Simulation with Real-time Station Updates")
st.number_input("Number of Agents:", min_value=1, value=5, key='num_persons')
st.number_input("Max Time Steps:", min_value=1, value=100, key='max_time_step')
st.number_input("Number of Bikes per Station:", min_value=1, value=10, key='num_bikes')

if st.button("Run Simulation"):
    run_simulation()
