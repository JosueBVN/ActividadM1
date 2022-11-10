from mesa import Agent, Model, time, space

from mesa.space import SingleGrid

from mesa.datacollection import DataCollector

from mesa.time import SimultaneousActivation
import numpy as np

class CleanRobot(Agent):
    
    def __init__(self,pos, model, agent_type):

        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type

    def step(self):
        similar = 0

        for neighbor in self.model.grid.neighbor_iter(self.pos):
            if neighbor.type == self.type:
                similar += 1

        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1
        
    class Robot(Model):
        def __init__(self, width=20, height=20, density=0.8, minority_pc=0.2, homophily=8):
        
            self.width = width
            self.height = height
            self.density = density
            self.minority_pc = minority_pc
            self.homophily = homophily

            self.schedule =time.RandomActivation(self)
            self.grid = space.SingleGrid(width, height, torus=True)

            self.happy = 0
            self.datacollector = DataCollector(
                {"happy": "happy"},  
                {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
            )

            for cell in self.grid.coord_iter():
                x = cell[1]
                y = cell[2]
                if self.random.random() < self.density:
                    if self.random.random() < self.minority_pc:
                        agent_type = 1
                    else:
                        agent_type = 0

                    agent = CleanRobot((x, y), self, agent_type)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)

            self.running = True
            self.datacollector.collect(self)

        def step(self):
            
            self.happy = 0  
            self.schedule.step()
            
            self.datacollector.collect(self)

            if self.happy == self.schedule.get_agent_count():
                self.running = False