# # ฟังก์ชันสำหรับสร้างแอนิเมชันกราฟเส้น (line chart animation)
# def animate_traffic_simulation(model):
#     fig, ax = plt.subplots()

#     # ตั้งค่าแกนกราฟ
#     ax.set_xlim(0, model.p.size)
#     ax.set_ylim(0, model.p.size)
#     ax.set_title("Traffic Simulation Movement")
    
#     # สร้างเส้นกราฟสำหรับ agent แต่ละตัว
#     lines = [ax.plot([], [], label=f"Person {agent.name}")[0] for agent in model.agents_Person]

#     def init():
#         for line in lines:
#             line.set_data([], [])
#         return lines

#     def update(frame):
#         # สำหรับ agent แต่ละตัว ให้ update เส้นทาง
#         for i, agent in enumerate(model.agents_Person):
#             if agent.path:

#                 agent.move()

#                 # ตำแหน่งที่ agent เดินทางถึง
#                 x_data, y_data = zip(*[(node.x, node.y) for node in agent.path])  # ข้อมูลเส้นทางที่เดิน
#                 lines[i].set_data(x_data, y_data)  # อัปเดตข้อมูลของกราฟเส้น
#         return lines

#     ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, repeat=False)
#     plt.legend()
#     plt.show()

#     #🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙🧪🪙