
class State:

    def __init__(self, left_state, right_state, top_state, bottom_state, x = None, y = None):
        self.policy = {'stay': 0, 'front': 1, 'left': 0, 'right': 0, 'back': 0}
        self.left_state = left_state
        self.right_state = right_state
        self.top_state = top_state
        self.bottom_state = bottom_state
        self.pos = {'x': x, 'y': y}
        self.reward = -0.4
        self.utility = [0.0, 'Stay']
        self.reward_set = False
        self.pomdp_utility = [0.0, 'Stay']
        
        
  
    def set_nearby_states(self, left_state, right_state, top_state, bottom_state):
        self.left_state = left_state
        self.right_state = right_state
        self.top_state = top_state
        self.bottom_state = bottom_state
        

    def insert(self, x, y, value):
        if x >= self.dimension['breadth'] or y >= self.dimension['length']:
            return False
        else:
            if y > 0:
                value.left_state = self.get_state(x, y-1)
            if y < self.dimension['length']-1:
                value.right_state = self.get_state(x, y+1)
            if x > 0:
                value.top_state = self.get_state(x-1, y)
            if x < self.dimension['breadth']-1:
                value.bottom_state = self.get_state(x+1, y)

            return True
        
        
class POMDP_PURSUIT_EVASION: 
     
#     transition_matrix = [];
#     states = []
#     discount_factor= 0.6
#     horizon = 7
# #     Rewards = [R1, R2, R3]
   
    
    
    def __init__(self, length, breadth):
        self.dimension = {'length': length, 'breadth': breadth}
        print(self.dimension)
        self.start = (2,2)
        self.destination = (4, 4); 
        self.states = [State(None, None, None, None) for i in range(0, length*breadth)]
        self.pursuer_location_time_step = [(0,4),(1,4),(2,4),(3,4),(3,3),(3,2),(4,2)]#,(4,2), (4,3), (5,3)]
        self.actions = ['Stay', 'Left', 'Right', 'Up', 'Down']
        self.gamma = 0.6
        self.horizon = 7
        self.chromeToAction = {'1': 'Stay', '2': 'Up', '3': 'Down', '4': 'Right', '5': 'Left',
                              '1.0': 'Stay', '2.0': 'Up', '3.0': 'Down', '4.0': 'Right', '5.0': 'Left'}
        
    def get_state(self, x, y):
        return self.states[self.dimension['length']*x+y]
    
    def assign_pos(self):
        for x in range(0, self.dimension['breadth']):
            for y in range(0, self.dimension['length']):
                state = self.get_state(x, y)
                state.pos['x'] = x
                state.pos['y'] = y
                State.insert( self,x, y, state)

            
#         % implement fitness 
#     def update_belief_state(belief, obs, action):
#         %implement here
    
    
    def find_transition_state(self, action, new_state, old_state):
        '''
        returns the probability of the transition model P(s'|s,a)
        :param action: possible actions ie Left, Right, Up, Bottom
        :param new_state:
        :param old_state:
        :return:
        '''
    
        if action == 'Left' and (new_state.right_state is old_state or (new_state is old_state and old_state.left_state is None)):
            prob = 1.0
        elif action == 'Right' and (new_state.left_state is old_state or (new_state is old_state and old_state.right_state is None)):
            prob = 1.0
        elif action == 'Up' and (new_state.bottom_state is old_state or (new_state is old_state and old_state.top_state is None)):
            prob = 1.0
        elif action == 'Down' and (new_state.top_state is old_state or (new_state is old_state and old_state.bottom_state is None)):
            prob = 1.0
        elif action =='Stay' and (new_state is old_state):
            prob = 1.0
        else:
            prob = 0.0
        return prob
    
    def compute_reward( self,ugvLoc, t):
        pursuerLoc = self.pursuer_location_time_step[t]
#         print('UGV ', ugvLoc , 'Pursuer: ',pursuerLoc)
        if pursuerLoc[0] == ugvLoc[0] and pursuerLoc[1] == ugvLoc[1]:
            reward = -1000
#             print('should land here')
        elif (pursuerLoc[0] != ugvLoc[0] or pursuerLoc[1] != ugvLoc[1]) and (self.destination[0] != ugvLoc[0] or  self.destination[1] != ugvLoc[1]):
            reward = -1
        elif self.destination[0] == ugvLoc[0] and  self.destination[1] == ugvLoc[1]: 
            reward = 100
        return reward
    
   
        
    def find_utility_state(self,action,state,t):
        UGV_LOC = [state.pos['x'], state.pos['y']]
        nearby_states = [state.left_state, state.right_state, state.top_state, state.bottom_state, state]
        refined_states = [x for x in nearby_states if x is not None]
        #sum([self.find_transition_state(i, x, state)*x.utility[0] for x in refined_states])
        utility= self.compute_reward(UGV_LOC, t)
        reward = sum([self.find_transition_state(action, x, state)*utility for x in refined_states])
        return reward
    
    def next_state(self,action,state):
        next_state = state;
        if action == self.actions[1] and state.left_state != None:
            next_state = state.left_state
        elif action == self.actions[2] and state.right_state != None:
            next_state = state.right_state
        elif action == self.actions[3] and state.top_state != None:
            next_state = state.top_state
        elif action == self.actions[4] and state.bottom_state != None:
            next_state = state.bottom_state
        return next_state;
    
    def value_fitness(self,policyNum):
#         print('-----------------------------------------------')
#         print(policyNum)
        policy = []
        for a in policyNum:
            policy.append(self.chromeToAction[str(int(a))])
#         print(policy)
        horizon = len(policy)
        #timestep =  horizon
        utility_set = []
        current_state = self.get_state(self.start[0],self.start[1])
        discounted_cumulative_reward = 0
        for timestep, action in enumerate(policy):
            utility= self.find_utility_state(action,current_state,timestep)
            discounted_reward = (self.gamma**timestep) * utility
            discounted_cumulative_reward =   discounted_reward + discounted_cumulative_reward
            current_state = self.next_state(action,current_state)
#             print('current_state: (', current_state.pos['x'],current_state.pos['y'], ') at timestep:  ',timestep)
        return discounted_cumulative_reward
            
        
    def expected_utility(a, s, timestep):
        
          return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])
            
            
            
            
            