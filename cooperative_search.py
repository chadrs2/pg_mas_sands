import numpy as np

class UAV():
    def __init__(self,id,mu0,Rs,eta_init):
        self.id = id
        self.mu = mu0
        self.Rs = Rs
        self.H_igt = None
        self.Q_igt = None
        self.eta_igt = eta_init

    def update_loc(self,delta_mu):
        self.mu += delta_mu

    def update_prob_map(self,Z,pc,pf,nrows,ncols):
        if self.H_igt is None:
            P_igt0 = np.ones((nrows,ncols)) * 0.5
            P_igt = P_igt0
            for coord, obs in Z.items():
                if obs == 0:
                    num = (1-pc)*P_igt0[coord[0],coord[1]]
                    den = (1-pc)*P_igt0[coord[0],coord[1]] + (1-pf)*(1-P_igt0[coord[0],coord[1]])
                    P_igt[coord[0],coord[1]] = num / den
                elif obs == 1:
                    num = pc*P_igt0[coord[0],coord[1]]
                    den = pc*P_igt0[coord[0],coord[1]] + pf*(1-P_igt0[coord[0],coord[1]])
                    P_igt[coord[0],coord[1]] = num / den
            self.H_igt = np.log(1 / P_igt - 1)
        else:
            for coord, obs in Z.items():
                if obs == 0:
                    self.H_igt[coord[0],coord[1]] += np.log((1-pf)/(1-pc))
                elif obs == 1:
                    self.H_igt[coord[0],coord[1]] += np.log(pf / pc)
        return self.H_igt
    
    def perform_observations(self,nrows,ncols):
        obsv = {}
        for g_row in range(nrows):
            for g_col in range(ncols):
                if self.is_cell_observable(g_col,g_row):
                    # Assuming no targets like paper
                    obsv[g_row,g_col] = 0 # x,y
        return obsv
    
    def is_cell_observable(self,x,y):
        dist = np.linalg.norm(np.array([x,y]) - self.mu, ord=2)
        return True if dist <= self.Rs else False

    def do_sensor_fusion(self,Hs,w_ijt):
        self.Q_igt = np.zeros_like(Hs[0])
        for j, H in enumerate(Hs):
            self.Q_igt += w_ijt[j] * H
    
    def update_uncertainty_map(self,kn):
        self.eta_igt = np.exp(-kn * self.Q_igt)

class CooperativeSearch():
    def __init__(self, eta_init):
        self.nrows = eta_init.shape[0]
        self.ncols = eta_init.shape[1]
        self.eta_igt = eta_init
        self.agents = None
        self.N = 0

    def add_agents(self,agents):
        if self.agents is None:
            self.agents = agents
            self.N = len(agents)
        else:
            for i in range(len(agents)):
                self.agents.append(agents[i])
            self.N += len(agents)

    def update_agent_loc(self,agent_id,delta_mu):
        self.agents[agent_id].update_loc(delta_mu)

    def get_constrained_actions(self,agent_id):
        cur_loc = self.agents[agent_id].mu
        C = []
        for x in range(-1,1+1):
            for y in range(-1,1+1):
                if (cur_loc[0] + x >= 0) and \
                    (cur_loc[1] + y >= 0) and \
                    (cur_loc[0] + x < self.ncols) and \
                    (cur_loc[1] + y < self.nrows):
                    if (x != 0 or y != 0) and self.eta_igt[cur_loc[1]+y, cur_loc[0]+x] > 0:
                        C.append((x,y))
        C.append((0,0))
        return C
    
    def sample_trial_action(self,agent_id):
        Ci = self.get_constrained_actions(agent_id)
        zi = 9 # max number of poss actions
        p = np.ones((len(Ci),)) / zi
        p[-1] = 1 - (len(Ci) - 1) / zi
        a_prime = np.random.choice(len(Ci), p=p)
        trial_action = np.array(Ci[a_prime],dtype=int)
        return trial_action
    
    def compute_coverage_performance(self):
        performance = 0
        for g_row in range(self.nrows):
            for g_col in range(self.ncols):
                min_dist, min_agent = self.closest_agent(g_col,g_row)
                performance += self.agents[min_agent].eta_igt[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
        return performance
    
    def compute_avg_uncertainty(self):
        avg_uncertainty = 0
        for n in range(len(self.agents)):
            avg_uncertainty += self.agents[n].eta_igt.sum()
        return avg_uncertainty / (len(self.agents)*self.nrows*self.ncols)
    
    def compute_curr_utility(self,agent_id,):
        performance_i = 0
        performance_not_i = 0
        for g_row in range(self.nrows):
            for g_col in range(self.ncols):
                min_dist, min_agent = self.closest_agent(g_col,g_row)
                performance_i += self.agents[min_agent].eta_igt[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
                min_dist, min_agent = self.closest_agent(g_col,g_row,agent_id)
                performance_not_i += self.agents[min_agent].eta_igt[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
        return performance_i - performance_not_i
    
    def compute_exp_utility(self,agent_id,trial_action):
        self.update_agent_loc(agent_id, trial_action)
        performance_i = 0
        performance_not_i = 0
        for g_row in range(self.nrows):
            for g_col in range(self.ncols):
                min_dist, min_agent = self.closest_agent(g_col,g_row)
                performance_i += self.agents[min_agent].eta_igt[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
                min_dist, min_agent = self.closest_agent(g_col,g_row,agent_id)
                performance_not_i += self.agents[min_agent].eta_igt[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
        self.update_agent_loc(agent_id, -trial_action)
        return performance_i - performance_not_i
    
    def closest_agent(self,x,y,not_id=None):
        min_dist = np.inf
        min_agent = None
        for n in range(self.N):
            if not_id == n:
                continue
            dist = np.linalg.norm(np.array([x,y]) - self.agents[n].mu, ord=2)
            if dist < min_dist:
                min_dist = dist
                min_agent = n
        return min_dist, min_agent

    def sensor_obsv_and_fusion(self,Rc,pc,pf,kn=1):
        neighbors = []
        Hs = []
        for n in range(self.N):
            # Perform sensor observations & individual map update
            Z_igt = self.agents[n].perform_observations(self.nrows,self.ncols)
            H_igt = self.agents[n].update_prob_map(Z_igt,pc,pf,self.nrows,self.ncols)
                        
            # Transmit updated probability map to neighbors based on Rc
            neighbor_idxs = self.get_neighbor_idxs(n, Rc)
            neighbors.append(neighbor_idxs)
            # for neighbor in neighbor_idxs:
                # self.agents[neighbor].add_neighbors_H(H_igt)
            Hs.append(H_igt)
        
        for i in range(self.N):
            w_ijt = np.zeros((self.N,))
            # Compute Weight-Matrix
            for j in range(self.N):
                if j in neighbors[i] and j != i:
                    w_ijt[j] = 1 / (1 + max(len(neighbors[i]),len(neighbors[j])))
            w_ijt[i] = 1 - w_ijt.sum()
        
            # Perform Information fusion
            self.agents[i].do_sensor_fusion(Hs,w_ijt)

            # Update unceratinty map (i.e. eta_igt)
            self.agents[i].update_uncertainty_map(kn)
    
    def get_neighbor_idxs(self, agent_id, Rc):
        neighbor_idxs = []
        agent_loc = self.agents[agent_id].mu
        for n in range(self.N):
            if np.linalg.norm(self.agents[n].mu - agent_loc, ord=2) <= Rc:
                neighbor_idxs.append(n)
        return neighbor_idxs
