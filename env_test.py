#Environment

import numpy as np

class GridEnvironment:
    def __init__(self, N, demand_data, solar_data, wind_data, day_index, timestep_length):
        self.current_timestep = 0
        self.total_timesteps = int(24 / timestep_length) #Assuming episode is 24 hours
        self.timestep_length = timestep_length  # Length of each timestep in hours

        # Need to think about what start / stop time we do. 12am-12am? 4am-4am etc <-- Carla comment: recommend 12am-11:59pm
        self.N = N  # Number of EVs
        self.state_size = 3 + N  # State Size
        self.action_size = 3**N # State Size

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
        self.SoC = [50] * N  # SoC status of each EV (non are connected to grid), used by environment ONLY 
    

    def reset(self, day):
        self.current_timestep = 0
        self.demand = 0  
        self.solar = 0  
        self.wind = 0  
        self.P_EV = [0] * self.N  
        #CHANGE WHEN GOING THROUGH MORE THAN ONE DAY OF DATA
        return self.get_state()

    def battery_voltage(self, soc):
        min_voltage = 3.0  # Minimum voltage at 0% SoC
        max_voltage = 4.2  # Maximum voltage at 100% SoC
        soc_array = np.array(soc)
        return min_voltage + (max_voltage - min_voltage) * (soc_array / 100)

    def get_PEV(self, soc, action):
        #MAX's CODE
        #ACTION IS A VECTOR OF 0s 1s, -1s
        #return power output of each EV (P_EV) & the SOC for the next state
        timestep = 0.25  # 15 minutes
        max_power = 11  # Maximum power in kW
        battery_capacity = 50  # Battery capacity in kWh
        charge_efficiency = 0.9
        discharge_efficiency = 0.95
        min_soc = 20
        max_soc = 80

        voltage = self.battery_voltage(soc)  # Calculate voltage for each SoC
        power = max_power * voltage / 4.2  # Calculate power for each SoC based on its voltage

        # Ensure new_soc is of a floating point type to accommodate fractional changes
        current_soc=np.array(soc, dtype=float)
        new_soc = np.copy(current_soc) # Cast to float to prevent UFuncTypeError
        action_np = np.array(action, dtype=float)
        powerEV = np.zeros_like(soc, dtype=float)  # Initialize powerEV array with zeros
     

    

        # Charging
        charge_indices = (action_np == 1) & (current_soc < max_soc)
        added_energy = np.minimum(power[charge_indices] * timestep, (max_soc - current_soc[charge_indices]) / 100 * battery_capacity) * charge_efficiency
        powerEV[charge_indices] = -(added_energy / timestep)  # Negative with respect to grid
        new_soc[charge_indices] += added_energy / battery_capacity * 100
        new_soc[charge_indices] = np.minimum(new_soc[charge_indices], max_soc)

        # Discharging
        discharge_indices = (action_np == -1) & (current_soc > min_soc)
        used_energy = np.minimum(power[discharge_indices] * timestep, (current_soc[discharge_indices] - min_soc) / 100 * battery_capacity) * discharge_efficiency
        powerEV[discharge_indices] = used_energy / timestep  # Positive with respect to grid
        new_soc[discharge_indices] -= used_energy / battery_capacity * 100
        new_soc[discharge_indices] = np.maximum(new_soc[discharge_indices], min_soc)

        # Idle
        idle_indices = (action_np == 0)
        powerEV[idle_indices] = 0  # No power exchange for idle

        return new_soc, powerEV

    def step(self, action):
        #Apply action-> calculate reward -> update state of environment
    
        #Apply action, update P_EV states
        self.SoC, self.P_EV = self.get_PEV(self.SoC, action) #Returns updated SoC & Power levels of each EV AFTER action
        next_P_EV=self.P_EV

        #Calculate Reward based upon action within same timestep
        reward = self.calculate_reward() 
        
        #Move env forward one timestep
        self.current_timestep += 1
        done = self.current_timestep >= self.total_timesteps
        next_demand, next_solar, next_wind = self.get_state()
        
        return reward, done, next_demand, next_solar, next_wind , next_P_EV
   
    def calculate_reward(self):
        # Calculate Reward
        reward= -np.abs(self.demand- (self.solar + self.wind + np.sum(self.P_EV)))

        return reward

    def get_state(self):
        # Access the current timestep's data correctly
        current_demand = self.demand_data[self.day_index, self.current_timestep]
        current_solar = self.solar_data[self.day_index, self.current_timestep]
        current_wind = self.wind_data[self.day_index, self.current_timestep]
        # Depending on your needs, return these values directly or along with other state information
        return current_demand, current_solar, current_wind




    def render(self):
        # Is this where we get our animations?!
        pass