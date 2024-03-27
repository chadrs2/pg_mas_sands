import matplotlib.pyplot as plt
import numpy as np


class UAV():
    def __init__(self,id,mu0,Rs):
        self.id = id
        self.mu = mu0
        self.Rs = Rs
        self.curr_a = None
        self.prev_a = mu0
        self.P_igt = None

    def update_loc(self,delta_mu):
        self.mu += delta_mu

    def update_prob_map(self,Z):
        if self.P_igt is None:
            self.P_igt = 0.5 * np.zeros_like(Z) 
    
    def perform_observations(self,nrows,ncols):
        obsv = np.zeros((nrows,ncols))
        for g_row in range(nrows):
            for g_col in range(ncols):
                if self.is_cell_observable(g_col,g_row):
                    obsv[g_row,g_col] = 1 # x,y
        return obsv
    
    def is_cell_observable(self,x,y):
        dist = np.linalg.norm(np.array([x,y]) - self.mu, ord=2)
        return True if dist <= self.Rs else False

class MultiUAVs():
    def __init__(self, eta_init):
        self.nrows = eta_init.shape[0]
        self.ncols = eta_init.shape[1]
        self.eta = eta_init
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
        # Unconstrained case. TODO: Add in Constrained case
        C = []
        for x in range(-1,1+1):
            for y in range(-1,1+1):
                if (cur_loc[0] + x >= 0) and \
                    (cur_loc[1] + y >= 0) and \
                    (cur_loc[0] + x < self.ncols) and \
                    (cur_loc[1] + y < self.nrows):
                    if (x != 0 or y != 0):
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
        # TODO: Constrained case:
        # trial_action = np.array(np.random.choice(Ci, p=np.ones_like(Ci)))
        return trial_action
    
    def compute_coverage_performance(self):
        performance = 0
        for g_row in range(self.nrows):
            for g_col in range(self.ncols):
                min_dist, min_agent = self.closest_agent(g_col,g_row)
                performance += self.eta[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
        return performance
    
    def compute_curr_utility(self,agent_id):
        performance_i = 0
        performance_not_i = 0
        for g_row in range(self.nrows):
            for g_col in range(self.ncols):
                min_dist, min_agent = self.closest_agent(g_col,g_row)
                performance_i += self.eta[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
                min_dist, min_agent = self.closest_agent(g_col,g_row,agent_id)
                performance_not_i += self.eta[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
        return performance_i - performance_not_i
    
    def compute_exp_utility(self,agent_id,trial_action):
        self.update_agent_loc(agent_id, trial_action)
        performance_i = 0
        performance_not_i = 0
        for g_row in range(self.nrows):
            for g_col in range(self.ncols):
                min_dist, min_agent = self.closest_agent(g_col,g_row)
                performance_i += self.eta[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
                min_dist, min_agent = self.closest_agent(g_col,g_row,agent_id)
                performance_not_i += self.eta[g_row,g_col] * np.exp(-min_dist) if min_dist <= self.agents[min_agent].Rs else 0.0
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

    def sensor_obsv_and_fusion(self):
        for n in range(self.N):
            Z_igt = self.agents[n].perform_observations()
            # TODO: Finish Sensor fusion implementation

class SimEnv():
    def __init__(self,nrows,ncols):
        self.nrows = nrows
        self.ncols = ncols

    def plot_env(self, agents, prior_map=None, t=0, display=True):
        fig = plt.figure(figsize=(8,8))
        plt.grid(True, linestyle='--', color='gray', alpha=0.5)
        plt.xticks(np.arange(0,self.ncols+1))
        plt.yticks(np.arange(0,self.nrows+1))
        plt.xlim(0,self.ncols)
        plt.ylim(0,self.nrows)
        plt.gca().set_aspect('equal',adjustable='box')

        # Plot Prior Map
        if prior_map is not None:
            plt.imshow(prior_map,cmap='spring')

        # Plot UAVs
        for n in range(len(agents)):
            assert(agents[n].id == n)
            circle = plt.Circle(
                agents[n].mu, 
                agents[n].Rs, 
                color='blue',
                alpha=0.5)
            plt.gca().add_patch(circle)
            plt.scatter(agents[n].mu[0],agents[n].mu[1],c='r',marker='*')
        
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Mission Space at {t} sec")
        if display:
            plt.show()
            return
        else:
            return fig