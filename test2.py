import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

# 🏝️🍹⛱️🌞 🌊
import time
import random


from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

# Animation Simulation
from matplotlib.animation import FuncAnimation
import IPython

import streamlit as st

import ipywidgets as widgets




class Bicycle(ap.Agent):
# เก็บเอาไว้ก่อน เผื่อในอนาคตจะทำเรื่องรับส่งจักรยานไปแต่ละสถานี เป็นการกระจายจักรยานไปให้สถานีที่จักรยานมันหมด
    def setup(self):
        self.speed = 1
    def move(self):
        x, y = self.model.grid.positions[self]
        next_position = (x + self.speed) % self.model.p.size, y
        if self.model.is_road[next_position]:
            self.model.grid.move_to(self, next_position)




class Station(ap.Agent):
    def setup(self):
        self.amount_Bicycle = self.model.p.capicity_Station
        name_station = ["LatKra", "Asoke", "Siam", "Phaya"]
        self.name = random.choice(name_station)




class Person(ap.Agent):
    counter = 1  # เริ่มนับที่ 100

    def setup(self):
        # ค่า Default
        self.speed = 1
        self.path = None
        self.name = f"{Person.counter}"  # ใช้ counter เป็น ID
        Person.counter += 1  # เพิ่ม counter ทุกครั้งที่สร้างตัวแทนใหม่
        self.block_step = 0 # คิดไว้ว่า จะเอาไว้ใช้ว่าสรุปแล้ว Person คนนี้ใช้เวลาเดินทางไปกี่บล็อกบน  grid เพื่อเอาไว้เปรียบเทียบเป็นผลลัพ (ตอนนี้ยังไม่ได้ทำ)
        self.sta_to_des = 0  # บันทึกจำนวน grid node
        self.destination = (np.random.randint(0, self.p.size), np.random.randint(0, self.p.size)) # เป็น tuple (x,y) , random destinationของแต่ละPerson or agent
        self.arrived = False # Falseคือหมายความว่าagentยังไปไม่ถึงDes, TrueคือถึงDesแล้ว
        self.isBike_in_use = False # สถานะว่ากำลังยืมจักรยานอยู่หรือเปล่า ค่าเริ่มต้นคือยัง ถ้าไปถึงสถานีเช่าแล้วได้จักรยานมาถึงเปลี่ยนเป็น True

        # ค่าคำนวณตอน Setup
        self.station_rent_position = None

        # ค่าที่กำหนดที่อื่น
        # self.start : ตำแหน่ง x,y เริ่มต้น

        # ค่าที่คำนวณหลังจาก setup
        self.station_return_position = None






    def find_path_btw_road_and_to_Des(self , agents_Station, size_grid , is_road):
        """⛱ ฟังชันหาเส้นทางการเดินทางไปสถานีที่จะคืนจักรยาน โดยเส้นทางที่ได้ จะต้องอยู่บนถนนเท่านั้น"""
        person_position = self.model.grid.positions[self]

        grid_data = np.ones((size_grid, size_grid))
        grid = Grid(matrix=grid_data)

        # 1. หาก่อนว่า สถานีที่เราจะไปคืนคือสถานีไหน (เป็นสถานีที่ใกล้กับจุดหมายปลายทางของเรา)
        start_x, start_y = person_position
        start_node = grid.node(self.destination[0], self.destination[1])
        finder = AStarFinder()
        best_path_Station_Return_To_Des = None
        shortest_distance = float('inf')

        for station in agents_Station: # วนลูปเพื่อเทียบว่าจาก Station ไปหาถนนที่ใกล้ที่สุดจะมีเส้นทางยังไง
            if station.position != person_position: # ทำเพื่อจะไม่นับ สถานี ณ ตำแหน่งปัจจุบันที่เราอยู่
              des_x, des_y = station.position
              end_node = grid.node(des_x, des_y)
              path, runs = finder.find_path(start_node, end_node, grid)  # start_node  <GridNode(5:7 0x7adbde0ff7c0)>

              # ตรวจสอบความยาวเส้นทางและเลือกเส้นทางที่สั้นที่สุด
              if path and len(path) < shortest_distance: # ถ้า path ไม่เป็น None
                  shortest_distance = len(path)
                  best_path_Station_Return_To_Des = path

        # ได้ตำแหน่งของสถานีที่เราจะไปคืนจักรยานได้
        self.station_return_position = (best_path_Station_Return_To_Des[-1].x , best_path_Station_Return_To_Des[-1].y)
        best_path_Station_Return_To_Des.reverse() # ได้เส้นทางจาก สถานีที่จะคืนจักรยาน to จุดหมายปลายทาง

        # 2. หาเส้นทางจาก สถานีที่เรามายืมจักรยาน ไปยัง สถานีที่เราจะไปคืนจักรยาน
        grid_data_Obstacles = Grid(matrix=np.where(is_road, 1, 5))
        finder = AStarFinder()
        x, y = self.station_rent_position
        start = grid_data_Obstacles.node(x , y)
        end = grid_data_Obstacles.node(self.station_return_position[0], self.station_return_position[1])
        road_path, runs = finder.find_path(start, end, grid_data_Obstacles)

        if road_path is not None:
          road_path.pop(0)  # ลบindexแรกออก เพราะมันคือตำแหน่งที่เรายืนอยู่ ณ ปัจจุบัน ซึ่งเราเดินมาแล้ว

        if best_path_Station_Return_To_Des is not None:
          best_path_Station_Return_To_Des.pop(0) # ลบindexแรกออก เพราะถ้าไม่ลบตอนบวก list เข้ากับ road_path จะมีตำแหน่งที่ซ้ำกับindexสุดท้ายของ road_path

        self.path = road_path + best_path_Station_Return_To_Des




    # การกำหนดเส้นทางของ Person ตั้งแต่ตอนที่นางเกิด
    def find_path_to_station(self , agents_Station, size_grid):
        """⛏ เป็นฟังชันเริ่มต้นจะถูกใช้ตอน set up Person ตั้งแต่ต้น , ไว้หาเส้นทางจาก จุดเริ่มต้น --> station ที่ใกล้ที่สุด"""
        # ฟังชันนี้คือหาเส้นทางจาก จุดเริ่มต้น --> station ที่ใกล้ที่สุด เพื่อที่พอถึง station จะได้เข้าเงื่อนไขใน step แล้วจะคำนวณหา (เส้นทางระหว่างถนนเพื่อไปสถานีที่จะคืนจักรยาน)+(เส้นทางจากสถานีที่จะคืนไปยังDes)
        person_position = self.model.grid.positions[self]

        grid_data = np.ones((size_grid, size_grid))
        grid = Grid(matrix=grid_data)

        # ตำแหน่งเริ่มต้น
        start_x, start_y = person_position # เป็นตัวเลข (x, y) ของ station เพราะฟังชันนี้จะคำนวณเส้นทางหลังจากมาถึง Station ไป Destination
        start_node = grid.node(start_x, start_y)

        # คำนวณ A* สำหรับทุกจุดปลายทาง
        finder = AStarFinder()
        best_path_to_Station_Rent = None
        shortest_distance = float('inf') # ค่าเป็นอนันต์ (infinity)

        for station in agents_Station: # วนลูปเพื่อเทียบว่าจาก Station ไปหาถนนที่ใกล้ที่สุดจะมีเส้นทางยังไง
            if station.amount_Bicycle > 0 and station.position != person_position:
              des_x, des_y = station.position
              end_node = grid.node(des_x, des_y)
              path, runs = finder.find_path(start_node, end_node, grid)  # start_node  <GridNode(5:7 0x7adbde0ff7c0)>

              # ตรวจสอบความยาวเส้นทางและเลือกเส้นทางที่สั้นที่สุด
              if path and len(path) < shortest_distance: # ถ้า path ไม่เป็น None
                  shortest_distance = len(path)
                  best_path_to_Station_Rent = path

        if best_path_to_Station_Rent is not None:
            best_path_to_Station_Rent.pop(0)
        self.station_rent_position = (best_path_to_Station_Rent[-1].x , best_path_to_Station_Rent[-1].y)
        self.path = best_path_to_Station_Rent


    def move(self):
        if self.path and len(self.path) > 0:
            # ย้ายไปยังตำแหน่งต่อไปในเส้นทาง
            next_position = self.path.pop(0)  # ดึงจุดถัดไปจากเส้นทาง
            # x, y = self.model.grid.positions[self]
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
        for i, person in enumerate(self.agents_Person):
            person.start = positions_person[i]


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


        # ตำแหน่ง Fix Station Location 🚩🏳️‍🌈 --> ต้องไม่อยู่ทับถนน ถ้าอยู่ทับแล้วจะเกิด Error ชื่อ AttributeError: 'NoneType' object has no attribute 'pop'
        # self.station_locations = [(3,9), (6,11), (9,6), (9,18), (11,2), (11,8), (11,12), (11,16), (14, 11), (17,9)]
        self.station_locations = [(3,9), (6,11), (9,6), (9,18), (11,2), (11,8),]

        # 🚀🚀 สร้างตำแหน่ง station แล้ว add บน Grid
        num_station = len(self.station_locations)
        self.agents_Station = ap.AgentList(self, num_station, Station)
        self.grid.add_agents(self.agents_Station, self.station_locations)

        for i, station in enumerate(self.agents_Station):
            # station.setup(self.station_locations[i])
            station.position = self.station_locations[i]
            # print(f"i : {i}   station : {station.name} /// {station}  ---> {station.position}")

        # ให้แต่ละ person หาสถานีที่ใกล้ที่สุด
        for person in self.agents_Person:
            person.find_path_to_station(self.agents_Station, self.p.size)
            person.block_step += len(person.path)

    def step(self):
        for agent in self.agents_Person: # แก้ไขตรงนี้
            agent_position = self.model.grid.positions[agent]

            if (agent.arrived == True): # บรรทัดนี้มาเพื่อแก้ ปัญหาที่ว่าแม้agentจะไปDesแล้ว มันก็จะคำนวณซ้ำๆต่อไปทั้งที่ไม่มีความจำเป็นต้องคำนวณagentที่ถึงจุดหมายแล้ว จึงใช้ continueเพื่อข้ามไปตัวอื่นเลย
                continue

            # เมื่อเดินทางมาถึงสถานี
            elif (len(agent.path) == 1) and (agent.path[-1].x , agent.path[-1].y) == agent.station_rent_position:
              # คิด self.path ต่อ
              # 1. ก่อนจะเข้าสถานี หนึ่งก้าว (ถ้าหลัง agent.move ด้านล่างคือเข้าเรียบร้อย)
              # 2. เดินมาถึงสถานีเรียบร้อย
              agent.move()

              # 3. หลังจากมาถึงสถานี agent จะรู้ว่าสรุปแล้วมีจักรยานเหลือไหม
              for each_station in self.agents_Station:
                if each_station.position == agent.station_rent_position:
                    if each_station.amount_Bicycle > 0: # ลบจำนวนจักรยานออก
                      each_station.amount_Bicycle -= 1
                      print(f"{each_station.name} เหลือเท่าไหร่ : {each_station.amount_Bicycle} คนที่มายืมคือ : {agent.name}")
                      agent.isBike_in_use = True # ยืมจักรยานมาแล้วก็เปลี่ยนค่าเป็นกำลังใช้จักรยานอยู่
                      # 4. ค้นหาเส้นทางไปยัง Destination
                      agent.find_path_btw_road_and_to_Des(self.agents_Station, self.p.size, self.is_road)
                      agent.block_step += len(agent.path)

                    else: # กรณีจักรยานหมด จะคิดเส้นทางไปstationใกล้เคียงกับตำแหน่งที่เราอยู่ ณ ตอนนี้
                      agent.find_path_to_station(self.agents_Station, self.p.size) # ไม่ส่งตำแหน่ง ณ ปัจจุบันไป เพราะในฟังชัน มีการหาค่าที่agentอยู่ ณ ตอนนี้อยู่แล้ว
                      agent.block_step += len(agent.path)

            # 5. เงื่อนไข ถ้ามาถึงสถานีที่จะคืนจักรยาน จะบวกจำนวนจักรยานในสถานี้นั้นออก
            elif agent_position == agent.station_return_position:
              for each_station in self.agents_Station:
                if each_station.position == agent.station_return_position:
                    each_station.amount_Bicycle += 1
                    agent.isBike_in_use = False
                    agent.move()
            # จะเข้า else ถ้ายังมี self.path ให้คนเดิน, ส่วนพวกif elif ด้านบนคือ นางต้องคำนวณเส้นทางก่อน ถึงจะมี self.path มาให้เดินตาม
            else:
              agent.move()


            if (len(agent.path) == 0) and agent.destination == agent_position:
              self.all_arrived -= 1
              agent.arrived = True # บอกว่า agent ตัวนี้เดินทางเสร็จแล้ว

        if self.all_arrived == 0:
          print(f"เข้าหมดทุกตัวแล้วฮะ 💵💵 จำนวน step การเดินทั้งหมดก็คือ คือ คือ --> {self.t}")
          self.stop()
          self.report()

        if self.t == self.p.steps:
          self.stop()
          self.report()



    # สรุปผลการเดินทางของแต่ละคน
    def report(self, key=None, value=None): # Add key and value arguments กัน error เฉยๆ
        if key == 'seed':
            return # Exit the method

        for agent in self.agents_Person:
            start_y, start_x = agent.start
            station_y, station_x = agent.station_rent_position
            dest_station_y, dest_station_x = agent.station_return_position
            des_y, des_x = agent.destination
            print(f"Person {agent.name}:"
              f"\n\tจุดเกิด ({start_y}, {start_x})" # อย่าลืมสลับ x กับ y เพราะมันอยู่คนละแกนกันตอนที่วาดออก
              f"\n\tDestination ({des_y}, {des_x})"
              f"\n\tตำแหน่งสถานีที่ไปยืมจักรยาน ({station_y}, {station_x})"
              f"\n\tตำแหน่งสถานีที่ไปคืนจักรยาน: ({dest_station_y}, {dest_station_x})"
              f"\n\tระยะทางทั้งหมด {agent.block_step} block steps")


    def update(self):
        # update แต่ละ step เพื่อใช้ในฟังชัน animation_plot

        return self.is_road, self.station_locations, self.agents_Station , self.agents_Person

    # def end(self):
    #     return self.all_arrived or self.t >= self.p.max_steps






def animation_plot(model, ax):
    """
        Updates and plots the simulation state.

        Args:
            frame (int): The current frame number of the animation.
            model (TrafficModel): The simulation model instance.
            ax (matplotlib.axes.Axes): The axes object to plot on.
    """

    road, stations, agents_Station, agents_Person = model.update() # เป็นการเรียกให้โมเดลอัปเดตสถานะใหม่ในแต่ละ time-step โดยคืนค่าออกมาเป็นสองสิ่ง:
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

        # เพิ่มชื่อให้กับแต่ละคน
        for person, (x, y) in zip(positions_person.keys(), zip(x_person, y_person)):
          # print(f"person 🚼🚼 : {person.name} --> status {person.isBike_in_use}")

          # Plot Destination แต่ละ Person
          des_y, des_x = person.destination
          ax.scatter(des_x, des_y, c='grey', s=SIZE_CIRCLE_DRAW)
          ax.text(des_x, des_y, f"Des-{person.name}", fontsize=10, ha='right')




    # Plot Person
    for agent in agents_Person:
        ax.text(x, y, f"P-{agent.name}", fontsize=10, ha='right')
        agent_position = model.grid.positions[agent]
        # color_person = 'yellow' if person.isBike_in_use == True else 'blue'
        ax.scatter(agent_position[1], agent_position[0], c='yellow' if agent.isBike_in_use == True else 'blue', s=SIZE_CIRCLE_DRAW)  # วาดตำแหน่งบุคคลด้วยสีฟ้า


    # ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
    positions_station = {k: v for k, v in model.grid.positions.items() if isinstance(k, Station)}  # ดึงตำแหน่งสถานี
    if positions_station:  # ตรวจสอบว่ามีตำแหน่งของบุคคลหรือไม่
        y_station, x_station = zip(*positions_station.values())  # แยกพิกัด y และ x
        ax.scatter(x_station, y_station, c='green', s=SIZE_CIRCLE_DRAW)
        # เพิ่มชื่อให้กับแต่ละสถานี
        for station, (x, y) in zip(positions_station.keys(), zip(x_station, y_station)):
          ax.text(x, y, f"{station.name} {station.amount_Bicycle}", fontsize=10)


    ax.set_title(f"Traffic Simulation\n"
                  f"Time-step: {model.t}")

def run_model(max_steps):
    parameters = {
        'steps': max_steps,
        'agents_Bicycle':  1, # มีไว้เพื่อ test การเดินไปบนกริดทีละ step เฉยๆ แต่ในอนาคตอาจจะนำมาปรับปรุงเพิ่ม เพื่อทำการขนส่งจักรยานไปแต่ละ Station
        'agents_Person': 5,
        'size': 20, # ตำแหน่งขนาดกว้างxสูงของกริด
        'capicity_Station': 2, # ค่า capacity เริ่มต้น
    }
    fig, ax = plt.subplots(figsize=(7, 7)) #  ขนาดของรูป (figure) ถูกตั้งไว้ที่ _x_ นิ้ว , fig เป็นตัวแทนของรูปทั้งหมดที่สร้างขึ้น , ax เป็นแกน (axis) ที่จะใช้ในการวาดแผนภาพบนรูปนั้น
    start_time = time.time() # เริ่มจับเวลา
    model = TrafficModel(parameters)

    # สำหรับ ax1
    animation = ap.animate(model, fig, ax, animation_plot) # ฟังก์ชัน animate จากไลบรารี AgentPy จะสร้างแอนิเมชันสำหรับการจำลองโมเดล TrafficModel , animation_plot อาจจะเป็นฟังก์ชันที่กำหนดวิธีการแสดงผลการจำลองแต่ละเฟรม

    # animation = FuncAnimation(
    #     fig, animation_plot, fargs=(model, ax),
    #     interval=2, repeat=True
    # )



    end_time = time.time() # จับเวลาหลังจาก simulation เสร็จสิ้น
    # จับเวลา
    print(f"Simulation time: 🛕🛕🛕🛕 {((end_time - start_time) * 1000):.2f} ms")


    # return IPython.display.HTML(animation.to_jshtml())
    return st.components.v1.html(animation.to_jshtml(fps=10), height=800)


# Create an input box for max_steps
max_steps_input = st.number_input(
    value=40,
    min_value=1,
    max_value=100,
    label='Max Steps:',
    step=1
)

max_steps_input = st.slider(
    value=40,
    min_value=1,
    max_value=100,
    label='Max Steps:',
    step=1
)

# # Button to start the simulation
# start_button = widgets.Button(description="Run Simulation")

# # Function to run simulation when button is clicked
# def on_button_clicked(b):
#     display(run_model(max_steps_input.value))

# # Link button click to simulation function
# start_button.on_click(on_button_clicked)

# # Display the input box and button
# display(max_steps_input, start_button)

if st.button("Run Simulation"):
    run_model(max_steps_input)

# Run the model and display the animation
# run_model(40) # max สูงสุดคือ100 แต่มันจะหยุดก่อนหน้าได้ ถ้าเกิดทุก agent เข้า des ไปหมดแล้ว ,มันไม่สามารถกำหนด infinity ได้ เพราะมันบังคับให้ใส่ int เข้าไป