#Environment

import numpy as np
from collections import deque
import random

class GridEnvironment:
    def __init__(self, N, demand_data, solar_data, wind_data, day_index, timestep_length):
        self.current_timestep = 0
        self.total_timesteps = int(24 / timestep_length) #Assuming episode is 24 hours
        self.timestep_length = timestep_length  # Length of each timestep in hours
       
        # Need to think about what start / stop time we do. 12am-12am? 4am-4am etc <-- Carla comment: recommend 12am-11:59pm
        self.N = N  # Number of EVs
        self.state_size = 3 + 1 +N # State Size, includes time, and SoC.... include P_Ev again?
        self.action_size = 201  # Action size from -1.0 to 1.0 in 0.01 steps

        self.demand_data = demand_data  
        self.solar_data = solar_data
        self.wind_data = wind_data
        self.day_index = day_index

        # Initialize with day 0 data (96 points = 24 hours of 15 min data)
        self.demand = demand_data[day_index,0]  
        self.solar = solar_data[day_index,0]
        self.wind = wind_data[day_index,0]

    
        self.P_EV = [0] * N  # Power status of each EV (non are connected to grid)
        # TODO Answer: If each episode is finite, how does the SoC status roll over to next episode and how does RL agent learn this?
        self.SoC = [100] * N  # SoC status of each EV (non are connected to grid), used by environment ONLY 
    

    def reset(self, day):
        self.current_timestep = 0
        self.demand = 0  
        self.solar = 0  
        self.wind = 0  
        self.P_EV = [0] * self.N  
        #CHANGE WHEN GOING THROUGH MORE THAN ONE DAY OF DATA
        return self.get_state()

    def decode_action(self, action_index):
        # Convert action index to proportion
        action = (action_index - 100) / 100.0  # Converts 0 to 200 into -1.0 to 1.0
        return [action]

    def battery_voltage(self, soc):
        min_voltage = 3.0  # Minimum voltage at 0% SoC
        max_voltage = 4.2  # Maximum voltage at 100% SoC
        soc_array = np.array(soc)
        return 4.2 * (soc_array / 100)

    def get_PEV(self, actions):
        timestep = (10 / 60)  # Convert 10 minutes to hours
        max_power = (11) / 3500  # Example scaling
        battery_capacity = (50) / 3500
        charge_efficiency = 0.90
        discharge_efficiency = 0.90
        min_soc = 20
        max_soc = 80

        self.P_EV = [0] * self.N  # Reset power for each EV

        # Filter EVs based on action and SoC constraints
        if actions[0] > 0:  # Charging
            eligible_evs = [i for i in range(self.N) if self.SoC[i] < max_soc]
        elif actions[0] < 0:  # Discharging
            eligible_evs = [i for i in range(self.N) if self.SoC[i] > min_soc]
        else:
            eligible_evs = []

        # Determine the number of EVs to affect based on the action percentage
        num_evs_affected = int(abs(actions[0]) * len(eligible_evs))
        selected_evs = random.sample(eligible_evs, min(num_evs_affected, len(eligible_evs)))

        voltages= self.battery_voltage(self.SoC)
        
        for i in selected_evs:
            if actions[0] > 0:  # Charging
                power = max_power * voltages[i] / 4.2   
                energy_added = power * timestep * charge_efficiency
                self.SoC[i] = self.SoC[i] + energy_added / battery_capacity * 100
                self.P_EV[i] = power # Charging ADDs to Demand (should be negative here)
            elif actions[0] < 0:  # Discharging
                power = max_power * voltages[i] /4.2  
                energy_used = power * timestep * discharge_efficiency #energy used will be negative
                self.SoC[i] = self.SoC[i] - energy_used / battery_capacity * 100
                self.P_EV[i] = -1*power  ## Discharging subtracts from Demand (should be positive here)
        return self.SoC, self.P_EV


    def step(self, action):
        #Apply action-> calculate reward -> update state of environment
        #Apply action, update P_EV states
        actions=self.decode_action(action)
        actions = np.array(actions)

        next_SoC, next_P_EV = self.get_PEV(actions) #Returns updated SoC & Power levels of each EV AFTER action
    
        #Calculate Reward based upon action within same timestep
        reward = self.calculate_reward(next_P_EV, actions) 
        
        #Move env forward one timestep
        self.current_timestep += 1
        done = self.current_timestep >= self.total_timesteps-1

        if not done:
            next_demand, next_solar, next_wind, next_SoC = self.get_state()
        else:
        # Handle the terminal state here, could be resetting or providing terminal values
        # Make sure to handle the case where you don't have a next state to provide
            next_demand, next_solar, next_wind, next_P_EV, next_SoC= 0, 0, 0, [0] * self.N, [0] * self.N
        
    
        return reward, done, next_demand, next_solar, next_wind , next_P_EV, next_SoC
   
#NEED TO FOCUS ON THE SEQUENCE OF, OBSERVE STATE, CALCULATE ACTION, CALCULATE REWARD etc

    def calculate_reward(self, next_P_EV, action):
        current_demand, current_solar, current_wind, current_SoC = self.get_state()

        #reward= np.abs(current_demand - (current_solar + current_wind + np.sum(next_P_EV)))

        reward= -1*np.abs(current_demand+ np.sum(next_P_EV) - (current_solar + current_wind))
    
        return reward

    def get_state(self):
        # Access the current timestep's data correctly
        current_demand = self.demand_data[self.day_index, self.current_timestep]
        current_solar = self.solar_data[self.day_index, self.current_timestep]
        current_wind = self.wind_data[self.day_index, self.current_timestep]
        current_SoC=self.SoC

        return current_demand, current_solar, current_wind, current_SoC

    def render(self):
        # Is this where we get our animations?!
        pass