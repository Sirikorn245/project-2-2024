import streamlit as st

import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import IPython.display


from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from pathfinding.core.diagonal_movement import DiagonalMovement

import requests

# longdo_token = "6b9e34b8551b12ebbb95956c56383211"

# HTML content for the Longdo Map
html_content = """
<html>
  <head>
    <meta charset="UTF-8" />
    <title>Longdo Map</title>
    <style type="text/css">
      html {
        height: 100%;
      }
      body {
        margin: 0px;
        height: 100%;
      }
      #map {
        height: 100%;
      }
    </style>
    <script
      type="text/javascript"
      src="https://api.longdo.com/map/?key=6b9e34b8551b12ebbb95956c56383211"
    ></script>
    <script>
      function init() {
        var map = new longdo.Map({
          placeholder: document.getElementById("map"),
        });

        // 🌴🍹🍉⛱️🥥กำหนด marks🌴🍹🍉⛱️🥥
        var marker1 = new longdo.Marker({ lon: 100.77068388462067, lat: 13.727868667926447 });
        map.Overlays.add(marker1);

        var marker2 = new longdo.Marker({ lon: 100.76436460018158, lat: 13.727668037525024 });
        map.Overlays.add(marker2);

        var marker3 = new longdo.Marker({ lon: 100.77500224113464, lat: 13.729528421937207 });
        map.Overlays.add(marker3);

        var marker4 = new longdo.Marker({ lon: 100.77996164560318, lat: 13.7295518720667 });
        map.Overlays.add(marker4);

        var marker5 = new longdo.Marker({ lon: 100.78087896108627, lat: 13.730672264410334 });
        map.Overlays.add(marker5);

        var marker6 = new longdo.Marker({ lon: 100.77518731355667, lat: 13.726521574796058 });
        map.Overlays.add(marker6);

        var marker7 = new longdo.Marker({ lon: 100.77740550041199, lat: 13.729210542172659 });
        map.Overlays.add(marker7);

        map.bound({
          minLon: 100.763,
          minLat: 13.724,
          maxLon: 100.784,
          maxLat: 13.7315,
        });

        //map.Route.mode(longdo.RouteMode.Cost);
        //map.Route.mode(longdo.RouteMode.Distance);  // ใช้ Mode ที่คำนวณระยะทาง
        //map.Route.enableRestrict(longdo.RouteRestrict.Bike, true);
        map.Route.mode(longdo.RouteMode.Walk);

        // กำหนดเส้นทางระหว่าง marker1 และ marker5
        map.Route.placeholder(document.getElementById('results'));
        map.Route.add(marker3);
        map.Route.add(marker5);
        map.Route.search(); // เรียกใช้ search เพื่อคำนวณเส้นทาง

      }
    </script>
  </head>
  <body onload="init();">
    <div id="map"></div>
    <div id="results"></div>
  </body>
</html>
"""

# Display the HTML content in Streamlit
st.components.v1.html(html_content, height=600, width=1000)



def find_nearest_path(start_position, road_data , is_road, size_grid, destination_agent):

    # grid_data_Obstacles = np.where(is_road, 1, 0) # คือตอนแรกเราทำแบบนี้ แต่การกำหนดให้ 1 เป็น road แล้ว 0 เป็นสิ่งกีดขวาง มันทำให้ค่า path เป็น None เนื่องจาก ไม่สามารถวิ่งผ่านสิ่งกีดขวาง (Obstacles) บนกริด ได้
    grid_data = np.ones((size_grid, size_grid))
    grid = Grid(matrix=grid_data)

    # ตำแหน่งเริ่มต้น
    start_x, start_y = start_position # เป็นตัวเลข (x, y) ของ station เพราะฟังชันนี้จะคำนวณเส้นทางหลังจากมาถึง Station ไป Destination
    start_node = grid.node(start_x, start_y)

    # คำนวณ A* สำหรับทุกจุดปลายทาง
    finder = AStarFinder()
    best_path_near_Station = None
    shortest_distance = float('inf') # ค่าเป็นอนันต์ (infinity)

    for road_positions in road_data: # วนลูปเพื่อเทียบว่าจาก Station ไปหาถนนที่ใกล้ที่สุดจะมีเส้นทางยังไง
        # print(road_positions) # (0, 10) (1, 10), ... มันคือตำแหน่งกริดสีดำที่เป็นถนน ซึ่งเป็นตำแหน่งใน road_data ที่เป็นลิส
        des_x, des_y = road_positions
        end_node = grid.node(des_x, des_y)
        path, runs = finder.find_path(start_node, end_node, grid)  # start_node  <GridNode(5:7 0x7adbde0ff7c0)>
        # print(len(path)) # 0
        # print(path) # []

        # ตรวจสอบความยาวเส้นทางและเลือกเส้นทางที่สั้นที่สุด
        if path and len(path) < shortest_distance: # ถ้า path ไม่เป็น None
            shortest_distance = len(path)
            best_path_near_Station = path

    if best_path_near_Station is not None:
        best_path_near_Station.pop(0)

    # Set Variable
    x, y = best_path_near_Station[-1].x, best_path_near_Station[-1].y # (x , y) คือตำแหน่งหลังจากออก Station มาอยู่บนถนน
    des_x_agent, des_y_agent = destination_agent
    # print(f"คำแหน่งที่อยู่บนถนนแล้ว : {x} {y} , ตำแหน่งจุดหมายปลายทางของ Person : {des_x_agent} {des_y_agent}") # 10 6


    # print("---------------------")
    # หลังจากมาถึงสถานีแล้ว Person จะเดินทางไปหาถนนที่ติดกับสถานี จึงเริ่มคำนวณหาวิธีให้ Person ไปยัง Des โดยที่เส้นทางที่มาจะต้องอยู่บนถนน แล้วหลังจากนั้นให้เดินเส้นทางปกติต่อ กรณีที่ Des ไม่อยู่บนถนน ซึ่งจะมี 2 ขั้นตอน
    # 1. คำนวณตำแหน่ง best_path ที่ Des อยู่ติดกับถนนก่อน (Des = start , road = End)
    best_path_near_Des = None
    shortest_distance = float('inf')
    start_node = grid.node(des_x_agent, des_y_agent)
    for destination in road_data:
        des_x, des_y = destination
        end_node = grid.node(des_x, des_y)
        path, runs = finder.find_path(start_node, end_node, grid)

        # ตรวจสอบความยาวเส้นทางและเลือกเส้นทางที่สั้นที่สุด
        if path and len(path) < shortest_distance: # ถ้า path ไม่เป็น None
            shortest_distance = len(path)
            best_path_near_Des = path
    best_path_near_Des.reverse() # reverse จากที่จุดเริ่มคือ Des ให้จุด Des กลับไปเป็นปลายทาง
    # print(f"1. ตำแหน่งถนนที่ agent จะปั่นจักรยานมา {best_path_near_Des[0].x} {best_path_near_Des[0].y}")


    # print("---------------------")
    # 2. คำนวณเส้นทางจากตำแหน่งเริ่มต้นบนถนน (ที่อยู่ใกล้กับ Station) ไปยังตำแหน่งบนถนนที่อยู่ติดกับจุดหมาย (Des) -->  (อันนี้ส่งไปคำนวณที่อื่นด้วย return เนื่องจากเหมือนจะนับจุดเริ่มต้นกับจุดendNodeเป็นของก่อนหน้า เลยแก้ด้วยการส่งไปไว้ที่อื่นเพื่อคำนวณแทน)
    x_start_coordinate_second , y_start_coordinate_second = best_path_near_Station[-1].x , best_path_near_Station[-1].y
    end_node_second = grid.node(best_path_near_Des[0].x, best_path_near_Des[0].y)

    # Check Mistake
    if start_node == end_node:
      print("🚧 ตำแหน่งเริ่มต้นและปลายทางเหมือนกัน!")

    start_node = grid.node(x_start_coordinate_second, y_start_coordinate_second)


    # เก็บคำสั่งนี้ไว้ใช้สำหรับกรณีต้องการเดินบนเส้นทแยงฮับ
    # finder = AStarFinder(diagonal_movement=DiagonalMovement.never)


    # print("---------------------")
    if best_path_near_Des is not None: # ลบตำแหน่งแรกที่ซ้ำออก เหมือนขั้นตอนก่อนหน้าทุกประการ
        best_path_near_Des.pop(0)

    # Check if best_path_near_Des is empty before accessing elements
    if not best_path_near_Des:
        return [best_path_near_Station, best_path_near_Des, (x_start_coordinate_second, y_start_coordinate_second), (x_start_coordinate_second, y_start_coordinate_second)] # Return the start coordinates if best_path_near_Des is empty

    # ถ้าทุกอย่างผ่าน จะส่งอันนี้ไปคำนวณ
    return [best_path_near_Station,best_path_near_Des, (x_start_coordinate_second, y_start_coordinate_second), (best_path_near_Des[0].x, best_path_near_Des[0].y)]

#ไว้หา station ที่ใกล้สุด
def find_nearest_station(person_position, station_locations):
    min_distance = float('inf')
    nearest_station = None
    for station_position in station_locations:
        distance = np.linalg.norm(np.array(person_position) - np.array(station_position))
        if distance < min_distance:
            min_distance = distance
            nearest_station = station_position
    return nearest_station

class Person(ap.Agent):
    counter = 1  # เริ่มนับที่ 100

    def setup(self):
        self.speed = 1
        self.path = None
        self.name = f"{Person.counter}"  # ใช้ counter เป็น ID
        Person.counter += 1  # เพิ่ม counter ทุกครั้งที่สร้างตัวแทนใหม่
        self.block_step = 0 # คิดไว้ว่า จะเอาไว้ใช้ว่าสรุปแล้ว Person คนนี้ใช้เวลาเดินทางไปกี่บล็อกบน  grid เพื่อเอาไว้เปรียบเทียบเป็นผลลัพ (ตอนนี้ยังไม่ได้ทำ)
        self.sta_to_des = 0  # บันทึกจำนวน grid node
        self.destination = (np.random.randint(0, self.p.size), np.random.randint(0, self.p.size)) # เป็น tuple (x,y) , random destinationของแต่ละPerson or agent

        self.arrived = False # Falseคือหมายความว่าagentยังไปไม่ถึงDes, TrueคือถึงDesแล้ว

# station_position
    def find_path(self):
        # หา station ที่ใกล้ที่สุด
        person_position = self.model.grid.positions[self]
        nearest_station = find_nearest_station(person_position, self.model.station_locations)
        self.nearest_station = nearest_station  # เก็บไว้ใน agent

        # หา station ที่ใกล้เดสสุด🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
        nearest_station_to_destination = find_nearest_station(self.destination, self.model.station_locations)
        self.nearest_station_to_destination = nearest_station_to_destination

        # สร้างแผนที่จาก is_road (ถนนสามารถผ่านได้คือ True)
        # grid_data = np.where(self.model.is_road, 1, 0)  # ใช้ 1 สำหรับถนนที่ผ่านได้ และ 0 สำหรับที่อื่น
        grid_data = np.ones((self.p.size, self.p.size))
        grid = Grid(matrix=grid_data)

        # ตำแหน่งปัจจุบันของ Person
        x, y = self.model.grid.positions[self]
        start = grid.node(x, y)

        # ตำแหน่ง Station
        station_x, station_y = nearest_station
        end = grid.node(station_x, station_y)

        # ใช้ A* หาเส้นทาง
        finder = AStarFinder()

        path, _ = finder.find_path(start, end, grid)
        # path [<GridNode(16:19 0x7b83a33891b0)>, <GridNode(16:18 0x7b83a338b730)> อธิบาย --> 16:19 คือ (x,y) ตัวด้านหลังเป็นแค่ memory address

        path.pop(0) # ทำไมถึง pop ออก เพราะ path มันจะนับตำแหน่งเริ่มต้นด้วย ทำให้ตอน step 1 มันจะไม่ขยับ แล้วไปขยับ step 2 แทน เลยต้องเอาตำแหน่งตัวมันเองออก
        # เก็บเส้นทางที่ได้

        # เพิ่ม block_step สำหรับระยะทางที่เดินไปยัง station
        # self.block_step += len(path)

        #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
        # เส้นทางจากสถานีไปยังสถานีใกล้ des
        if nearest_station != nearest_station_to_destination:
            start = grid.node(station_x, station_y)
            station_dest_x, station_dest_y = nearest_station_to_destination
            end = grid.node(station_dest_x, station_dest_y)
            path_between_stations, _ = finder.find_path(start, end, grid)
            path_between_stations.pop(0)
        else:
            path_between_stations = []

        # เริ่มเดินทางต่อไปยัง Destination
        start = grid.node(*nearest_station_to_destination)  #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
        des_x, des_y = self.destination
        end = grid.node(des_x, des_y)  #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
        path_to_Des, _ = finder.find_path(grid.node(station_x, station_y), grid.node(des_x, des_y), grid)
        path_to_Des.pop(0)

        path_journey = path + path_between_stations + path_to_Des #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
        self.path = path_journey

        # คำนวณระยะทาง🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
        self.block_step = len(self.path)
        self.sta_to_des = len(path_between_stations) + len(path)

    def move(self):
        if self.path and len(self.path) > 0:
            # ย้ายไปยังตำแหน่งต่อไปในเส้นทาง
            next_position = self.path.pop(0)  # ดึงจุดถัดไปจากเส้นทาง
            # x, y = self.model.grid.positions[self]
            # if (x == 5 and y == 9):
            #   print(self.name)

            self.model.grid.move_to(self, next_position)
        # if len(self.path) == 0:
        #   print("จบการเดินทาง" , self.name )

class Bicycle(ap.Agent):
    def setup(self):
        self.speed = 1

    def move(self):
        x, y = self.model.grid.positions[self]
        next_position = (x + self.speed) % self.model.p.size, y
        if self.model.is_road[next_position]:
            self.model.grid.move_to(self, next_position)

class TrafficModel(ap.Model):
    def setup(self):

        self.all_arrived = int(self.p.agents_Person) # ค่าตั้งต้นคือจำนวน agent ทั้งหมดซึ่งเป็นค่าตัวเลข int --> ถ้าเกิด agent ไปถึง Destination จะมาทำการลบค่าตรงนี้ -1 ซึ่งไว้ใช้เช็กว่า ทุก agent ถึง Destination ของตัวเองแล้วหรือยัง ถ้าถึงแล้วค่าตรงนี้จะเหลือ 0

        self.grid = ap.Grid(self, [self.p.size, self.p.size], track_empty=True)

        # Create Person
        self.agents_Person = ap.AgentList(self, self.p.agents_Person, Person)

        # Create Bicycle
        self.agents_Bicycle = ap.AgentList(self, self.p.agents_Bicycle, Bicycle)

        # สุ่มตำแหน่งคน แล้ว add บน Grid
        num_person_samples = self.p.agents_Person # จำนวนคนที่ต้องการสุ่ม
        positions_person = [(np.random.randint(0, self.p.size), np.random.randint(0, self.p.size)) for _ in range(num_person_samples)]
        self.grid.add_agents(self.agents_Person, positions_person)


        # Set Road Route
        line_road = self.p.size // 2
        road_positions = [(x, line_road) for x in range(self.p.size)] + [(line_road, y) for y in range(self.p.size)]

        # Set จุดจักรยาน ที่เป็นการสุ่มบนถนน Road_Position
        self.positions_bicycle = self.random.sample(road_positions, k=len(self.agents_Bicycle))
        print("start_positions", self.positions_bicycle) # เช่น [(19, 10)]
        self.grid.add_agents(self.agents_Bicycle, self.positions_bicycle)


        self.is_road = np.zeros((self.p.size, self.p.size), dtype=bool)
        self.is_road[:, line_road] = True # เลือกคอลัมน์ที่เป็นแนวตั้งทั้งหมด (:) สำหรับแถวที่เป็น road_y ซึ่งเป็นตำแหน่งของถนนที่คุณต้องการให้จักรยานเคลื่อนที่
        self.is_road[line_road, :] = True
        # จะได้ is_road ประมาณนี้ array([[False, False, False, False, True, False, False, False]]) กรณีsize 8*8
        self.is_road[13][5] = True
        self.is_road[14][5] = True

        self.is_station = np.zeros((self.p.size, self.p.size), dtype=bool)
        # ตำแหน่ง Fix Station Location 🚩🏳️‍🌈 --> ต้องไม่อยู่ทับถนน ถ้าอยู่ทับแล้วจะเกิด Error ชื่อ AttributeError: 'NoneType' object has no attribute 'pop'
        self.station_locations = [(3,9), (6,11), (9,6), (9,18), (11,2), (11,8), (11,12), (11,16), (14, 11), (17,9)]
        for each_station in self.station_locations:
            self.is_station[each_station] = True  # สถานี

        # station_position = self.station_locations[0]  # กำหนดให้ทุกคนไปยังสถานีแรก
        for person in self.agents_Person:
            # person.find_path(station_position)
            person.find_path() # ให้แต่ละ person หาสถานีที่ใกล้ที่สุด

    def step(self):
        for agent in self.agents_Bicycle:
            agent.move()

        for agent in self.agents_Person: # แก้ไขตรงนี้
            if (agent.arrived == True): # บรรทัดนี้มาเพื่อแก้ ปัญหาที่ว่าแม้agentจะไปDesแล้ว มันก็จะคำนวณซ้ำๆต่อไปทั้งที่ไม่มีความจำเป็นต้องคำนวณagentที่ถึงจุดหมายแล้ว จึงใช้ continueเพื่อข้ามไปตัวอื่นเลย
              continue
            agent_position = self.model.grid.positions[agent]
            # ตรวจสอบว่าตำแหน่งของ agent เข้า Station หรือยัง ถ้าเข้าแล้วจะให้คำนวณหาเส้นทางถนน พอคำนวณเสร็จ ได้ค่า Pathจนถึง Des มันจะเข้า eles แล้วแล้วจะไม่เข้าอันนี้อีก เพราะ if อันนี้จะถูกเข้าเงื่อนไขแค่ครั้งเดียวตอนถึงสถานีเท่านั้น
            if agent_position in self.station_locations:
              # print("enter goal" , agent_position , "name : ",agent.name) # enter goal (5, 9) 3
              station_position = (agent_position[0], agent_position[1])  # ตำแหน่ง station
              road_positions = [(x, self.p.size // 2) for x in range(self.p.size)] + [(self.p.size // 2, y) for y in range(self.p.size)]

              # ได้ค่า return กลับมาจากฟังชัน find_nearest_path
              nearest_path = find_nearest_path(station_position, road_positions, self.is_road, self.p.size , agent.destination) # หา Path ที่ใกล้ที่สุดที่ Person หลังจากมาถึง Station จะไปยังถนน [<GridNode(5:9 0x7a8933879f60)>, <GridNode(5:10 0x7a89338785b0)>]

              # เริ่มคำนวณหาเส้นทางต่อ
              finder = AStarFinder()
              grid_data_Obstacles = np.where(self.is_road, 1, 5) # ใช้ 1 สำหรับถนนที่ผ่านได้ และ 5 สำหรับที่อื่น คำถามคือทำไมกำหนด5? = เคยกำหนด 0 แล้ว แต่เหมือนค่า cost ระหว่าง 1 กับ 0 มันห่างกันนิดเดียว ทำให้มันดูไม่สนใจว่าจะเป็น 1 หรือ 0 มันเดินหมด เลยต้องกำหนดค่าcostให้ห่างกันหน่อย ซึ่งเลข5ก็แค่สุ่มขึ้นมาเพราะเลขสวยดีเฉยๆ ในอนาคตไม่รู้ว่าจะเปลี่ยนไหม แต่ถ้าเปลี่ยนก็ไม่กระทบไร
              grid = Grid(matrix=grid_data_Obstacles) # สร้างกริดที่มีสิ่งกีดขวางคือ เส้นทางที่ไม่ใช่ถนน ที่มันไม่อาจเดินได้ เพราะจะต้องปั่นจักรยานบนถนน Only!
              x, y = nearest_path[2] # (x , y) เป็นตำแหน่งของกริดถนนที่ใกล้กับ Station มากที่สุด
              des_x, des_y = nearest_path[3] # (des_x , des_y)  เป็นตำแหน่งของกริดถนนที่ใกล้กับ Des มากที่สุด
              start = grid.node(x, y)
              end = grid.node(des_x, des_y)
              road_path, runs = finder.find_path(start, end, grid)

              # ลบตำแหน่งแรกที่ซ้ำออก เหมือนขั้นตอนก่อนหน้าทุกประการ
              if road_path is not None:
                road_path.pop(0)
              best_path_road_near_Destination = nearest_path[1]
              print("Name : " , agent.name , "best_path_road_near_Destination " , best_path_road_near_Destination)
              #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙
              # if best_path_road_near_Destination is not None:
              #   best_path_road_near_Destination.pop(0)

              #แก้อันบนเป็นอันล่าง กัน IndexError: pop from empty list

              if best_path_road_near_Destination:
                best_path_road_near_Destination.pop(0)
                #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙


              best_path = nearest_path[0] + road_path + best_path_road_near_Destination # นำสามเส้นทางมาบวกกันได้ Best_path
              print(f"Best path {best_path} ของ {agent.name}")
              agent.path = best_path # สรุป Best_path คือ best_path_road_near_Station + road_path + best_path_road_near_Destination
              agent.move() # เริ่มเดิน
            else:
              # จากจุดตั้งต้นที่เกิด Move ไปหา Station จากนั้นพอมาถึง Stationจะเข้าเงื่อนไข if ด้านบนแล้วมีการกำหนด agent.path ใหม่ จากนั้นก็กลับมาเข้า else ตัวนี้จนจบการเดินทางถึง Des
              agent.move()
            if (len(agent.path) == 0):
              print(f"ว้าวๆๆ step ที่: {self.t} ของagentชื่อ: {agent.name}") # ถึงจุด Des พอดี
              self.all_arrived -= 1
              agent.arrived = True
              print("self.all_arrived ",self.all_arrived)

        # if self.all_arrived == 0:
        #   print(f"เข้าหมดทุกตัวแล้วฮะ 💵💵 จำนวน step การเดินทั้งหมดก็คือ คือ คือ --> {self.t}")
        #   self.stop()
        # 🛕🛕🛕🛕🛕🛕🛕
        if self.t == self.p.steps:
          self.stop()

    # สรุปผลการเดินทางของแต่ละคน
    def report(self, key=None, value=None): # Add key and value arguments กัน error เฉยๆ
        if key == 'seed':
            return # Exit the method

        for agent in self.agents_Person:
            start_y, start_x = agent.model.grid.positions[agent]
            station_y, station_x = agent.nearest_station
            dest_station_y, dest_station_x = agent.nearest_station_to_destination
            des_y, des_x = agent.destination
            # st.write(f"Person {agent.name}:"
            #   f"\n\tจุดเกิด ({start_x}, {start_y})"
            #   f"\n\tNearest station ({station_x}, {station_y})"
            #   f"\n\tDestination ({des_x}, {des_y})"
            #   f"\n\tStation ใกล้เดส: ({dest_station_x}, {dest_station_y})"
            #   f"\n\tระยะทางจาก station ไป destination {agent.sta_to_des} block steps"
            #   f"\n\tระยะทางทั้งหมด {agent.block_step} block steps")

    def update(self):
        return self.is_road, self.grid.positions, self.is_station, self.station_locations

    # def end(self):
    #     return self.all_arrived or self.t >= self.p.max_steps

def animation_plot(model, ax):
    road, positions, _, stations = model.update() # เป็นการเรียกให้โมเดลอัปเดตสถานะใหม่ในแต่ละ time-step โดยคืนค่าออกมาเป็นสองสิ่ง:
    # road: ข้อมูลเกี่ยวกับถนน เช่น แผนที่ถนนในรูปแบบของ array ที่มีค่าเป็น 0 หรือ 1 (binary) เพื่อแสดงส่วนของถนนและพื้นที่อื่น ๆ
    # positions: ตำแหน่งของจักรยานในโมเดลซึ่งเก็บในรูปแบบ dictionary ที่บอกตำแหน่ง (y, x) ของแต่ละจักรยาน --> การเข้าถึงค่าในอาเรย์สองมิติ โดยปกติจะเรียกผ่านการระบุ (row, column) ซึ่งเทียบได้กับ (y, x) นั่นเอง

    # Plot road
    ax.imshow(road, cmap='binary') # ใช้ ax.imshow() ในการวาดแผนที่ถนน (road) ลงบนแกน ax โดยใช้โหมดสี (colormap) แบบ binary เพื่อแสดงถนนเป็นโทนสีขาวดำ (0 เป็นสีหนึ่งและ 1 เป็นอีกสีหนึ่ง)

    # xticks คือเส้นขีดเล็ก ๆ ที่ปรากฏบนแกน x อะ
    ax.set_xticks(np.arange(-0.5, model.p.size, 1), minor=True)  # ตั้งค่าตำแหน่งเส้นกริดแนวนอน , ตั้งค่าตำแหน่งของ tick marks เพื่อให้แสดงที่ -0.5, 0.5, 1.5, ..., 19.5 ช่วยให้เส้นกริดอยู่กลางระหว่างช่องในกระดานหมากรุก เพื่อให้ผู้ใช้เห็นว่าแต่ละช่องแบ่งออกเป็นอย่างไร
    ax.set_yticks(np.arange(-0.5, model.p.size, 1), minor=True)  # ตั้งค่าตำแหน่งเส้นกริดแนวตั้ง

    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)  # วาดเส้นกริดสีดำ
    ax.tick_params(which='minor', size=0)  # ซ่อน tick-mark

    # แสดงตัวเลขตามขนาด (size)
    ax.set_xticks(np.arange(model.p.size))  # กำหนดตำแหน่งตัวเลขแกน X
    ax.set_yticks(np.arange(model.p.size))  # กำหนดตำแหน่งตัวเลขแกน Y
    ax.set_xticklabels(np.arange(model.p.size))  # กำหนดตัวเลขที่แสดงบนแกน X
    ax.set_yticklabels(np.arange(model.p.size))  # กำหนดตัวเลขที่แสดงบนแกน Y

    SIZE_CIRCLE_DRAW = 40 # ขนาดจุดที่จะวาด

    # Plot bicycles
    positions_bicycle = {k: v for k, v in model.grid.positions.items() if isinstance(k, Bicycle)}
    # model.grid.positions คือ dictionary ที่เก็บตำแหน่งของตัวแทน (agents) ในโมเดล โดยแต่ละตัวแทน (agent) จะถูกใช้เป็น key และตำแหน่งของมัน (เป็น tuple (x, y)) จะเป็น value
    # ==> ส่วน items() จะส่งกลับรายการของ key-value pairs ใน dictionary ซึ่งในที่นี้จะเป็นรายการของตัวแทน (agents) และตำแหน่งของพวกเขา.
    # k = แทนตัวแทน (agent) เช่น Bicycle หรือ Person , v = แทนตำแหน่งของตัวแทน (agent) ในรูปแบบ (x, y)
    if positions_bicycle: # ตรวจสอบว่ามีข้อมูลตำแหน่งของจักรยานหรือไม่ หากมีจักรยานใน simulation จะเข้าสู่ขั้นตอนต่อไป
        y, x = zip(*positions_bicycle.values()) # ตำแหน่งของจักรยานที่ได้จาก positions.values() จะถูกแยกออกเป็นพิกัด y และ x ด้วยการใช้ฟังก์ชัน zip() เพื่อเตรียมพิกัดสำหรับการ plot จุด
        ax.scatter(x, y, c='red', s=SIZE_CIRCLE_DRAW) # วาดตำแหน่งจักรยานลงบนแผนที่โดยใช้ ax.scatter() ซึ่งจะ plot จุดที่ตำแหน่ง (x, y) , ใช้สีแดง (c='red') เพื่อแทนจักรยาน และกำหนดขนาดของจุด (s=50)

    # Plot persons
    positions_person = {k: v for k, v in model.grid.positions.items() if isinstance(k, Person)}  # ดึงตำแหน่งของบุคคล
    # positions_person จะเป็น dictionary ใหม่ที่เก็บเฉพาะตำแหน่งของตัวแทนที่เป็นประเภท Person เช่น {Person1: (x1, y1), Person2: (x2, y2),...}
    if positions_person:  # ตรวจสอบว่ามีตำแหน่งของบุคคลหรือไม่
        y_person, x_person = zip(*positions_person.values())  # แยกพิกัด y และ x
        ax.scatter(x_person, y_person, c='blue', s=SIZE_CIRCLE_DRAW)  # วาดตำแหน่งบุคคลด้วยสีฟ้า

        # เพิ่มชื่อให้กับแต่ละคน
        for person, (x, y) in zip(positions_person.keys(), zip(x_person, y_person)):
          ax.text(x, y, f"P-{person.name}", fontsize=10, ha='right')
          # Plot Destination แต่ละ Person
          des_y, des_x = person.destination
          ax.scatter(des_x, des_y, c='grey', s=SIZE_CIRCLE_DRAW)
          ax.text(des_x, des_y, f"Des-{person.name}", fontsize=10, ha='right')


    # Plot Station (ที่ตอนนี้มีแค่ที่เดียว ถ้ามีหลายที่ จะต้องเปลี่ยน)
    y_station, x_station = zip(*stations)  # ใช้ zip เพื่อแยก
    ax.scatter(x_station, y_station, c='green', s=SIZE_CIRCLE_DRAW)  # วาดตำแหน่งบุคคลด้วยสีฟ้า


    ax.set_title(f"Traffic Simulation\n"
                  f"Time-step: {model.t}")
def run_model(max_steps):
    parameters = {
        'steps': max_steps,
        'agents_Bicycle': 1, # มีไว้เพื่อ test การเดินไปบนกริดทีละ step เฉยๆ แต่ในอนาคตอาจจะนำมาปรับปรุงเพิ่ม เพื่อทำการขนส่งจักรยานไปแต่ละ Station
        'agents_Person':10,
        'size': 20 # ตำแหน่งขนาดกว้างxสูงของกริด
    }
    fig, ax = plt.subplots(figsize=(5, 5)) #  ขนาดของรูป (figure) ถูกตั้งไว้ที่ _x_ นิ้ว , fig เป็นตัวแทนของรูปทั้งหมดที่สร้างขึ้น , ax เป็นแกน (axis) ที่จะใช้ในการวาดแผนภาพบนรูปนั้น
    model = TrafficModel(parameters)
    animation = ap.animate(model, fig, ax, animation_plot) # ฟังก์ชัน animate จากไลบรารี AgentPy จะสร้างแอนิเมชันสำหรับการจำลองโมเดล TrafficModel , animation_plot อาจจะเป็นฟังก์ชันที่กำหนดวิธีการแสดงผลการจำลองแต่ละเฟรม
    
    model.report()  # เรียกใช้ฟังก์ชันสรุป

    return st.components.v1.html(animation.to_jshtml(fps=10), height=600)

# Run the model and display the animation
run_model(20) # max สูงสุดคือ100 แต่มันจะหยุดก่อนหน้าได้ ถ้าเกิดทุก agent เข้า des ไปหมดแล้ว ,มันไม่สามารถกำหนด infinity ได้ เพราะมันบังคับให้ใส่ int เข้าไป

st.write("คิดถึงหมอจัง แงๆๆๆๆๆๆๆ")