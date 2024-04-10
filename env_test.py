#Environment
timestep_length= .25 # 15 minute increments

class GridEnvironment:

    def __init__(self, N, demand_data, solar_data, wind_data):

        # Need to think about what start / stop time we do. 12am-12am? 4am-4am etc <-- Carla comment: recommend 12am-11:59pm
        self.N = N  # Number of EVs

        # TODO BY CARLA
        # Initialize with day 0 data (96 points = 24 hours of 15 min data)
        self.demand = demand_data[0,:]  
        self.solar = solar_data[0,:]
        self.wind = wind_data[0,:]

        # TODO BY MAX
        self.P_EV = [0] * N  # Power status of each EV (non are connected to grid)

        self.timestep_length = timestep_length  # Length of each timestep in hours
        
        # Episode End
        self.current_timestep = 0
        self.total_timesteps = int(24 / self.timestep_length) #Assuming episode is 24 hours


    def reset(self, day):
        self.current_timestep = 0
        self.demand = 0  
        self.solar = 0  
        self.wind = 0  
        self.P_EV = [0] * self.N  
        #CHANGE WHEN GOING THROUGH MORE THAN ONE DAY OF DATA
        return self.get_state()

    def battery_voltage(soc):
        min_voltage = 3.0  # Minimum voltage at 0% SoC
        max_voltage = 4.2  # Maximum voltage at 100% SoC
        return min_voltage + (max_voltage - min_voltage) * (soc / 100)

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

        voltage = battery_voltage(soc)  # Calculate voltage for each SoC
        power = max_power * voltage / 4.2  # Calculate power for each SoC based on its voltage

        # Ensure new_soc is of a floating point type to accommodate fractional changes
        new_soc = np.copy(soc).astype(float)  # Cast to float to prevent UFuncTypeError
        powerEV = np.zeros_like(soc, dtype=float)  # Initialize powerEV array with zeros

        # Charging
        charge_indices = (action == 1) & (soc < max_soc)
        added_energy = np.minimum(power[charge_indices] * timestep, (max_soc - soc[charge_indices]) / 100 * battery_capacity) * charge_efficiency
        powerEV[charge_indices] = -(added_energy / timestep)  # Negative with respect to grid
        new_soc[charge_indices] += added_energy / battery_capacity * 100
        new_soc[charge_indices] = np.minimum(new_soc[charge_indices], max_soc)

        # Discharging
        discharge_indices = (action == -1) & (soc > min_soc)
        used_energy = np.minimum(power[discharge_indices] * timestep, (soc[discharge_indices] - min_soc) / 100 * battery_capacity) * discharge_efficiency
        powerEV[discharge_indices] = used_energy / timestep  # Positive with respect to grid
        new_soc[discharge_indices] -= used_energy / battery_capacity * 100
        new_soc[discharge_indices] = np.maximum(new_soc[discharge_indices], min_soc)

        # Idle
        idle_indices = (action == 0)
        powerEV[idle_indices] = 0  # No power exchange for idle

        return new_soc, powerEV

    def step(self, action):
        #Advances environment forward one step
        # Apply action, update the state of the environment,
        # calculate reward, and return new_state, reward, done, info
        self.demand = data[self.current_timestep,0] # demand entry indexed by timestep
        self.solar = data[self.current_timestep,1]# solar entry indexed by timestep
        self.wind = data[self.current_timestep,2] # wind entry indexed by timestep
        self.P_EV = [self.current_timestep] * self.N 

        #APPLY ACTION AND UPDATE PEV STATES
        self.P_EV = self.get_PEV() #      EXAMPLE

        #MOVE
        self.current_timestep += 1
        done = self.current_timestep >= self.total_timesteps


        reward = self.calculate_reward() 
        next_state = self.get_state() # DO WE NEED?????
        return next_state, reward, done, {}
   
    def calculate_reward(self):
        # Calculate Reward
        reward= -np.abs(self.demand- (self.solar + self.wind + np.sum(self.P_EV)))

        return reward

    def get_state(self):
        # Return the current state
        return [self.demand, self.solar, self.wind , self.P_EV]


    def render(self):
        # Is this where we get our animations?!
        pass