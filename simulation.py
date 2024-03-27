import matplotlib.pyplot as plt
import numpy as np

class SimEnv():
    def __init__(self,nrows,ncols,obstacles=None):
        self.nrows = nrows
        self.ncols = ncols
        self.obstacles = obstacles

    def plot_env(self, agents, prior_map=None, t=0, agent_id=0, display=True):
        fig = plt.figure(figsize=(8,8))
        plt.grid(True, linestyle='--', color='gray', alpha=0.5)
        plt.xticks(np.arange(0,self.ncols+1))
        plt.yticks(np.arange(0,self.nrows+1))
        plt.xlim(0,self.ncols)
        plt.ylim(0,self.nrows)
        plt.gca().set_aspect('equal',adjustable='box')

        # Plot Prior Map
        if prior_map is not None:
            plt.imshow(prior_map,cmap='viridis')#,vmin=0,vmax=1)
            plt.colorbar()

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
            plt.text(agents[n].mu[0],agents[n].mu[1]+1e-1,s=str(n),color='white')

        # Plot Obstacles
        if self.obstacles is not None:
            for o in self.obstacles:
                xy_start, xy_end = o
                rect = plt.Rectangle(
                    (xy_start[0], xy_start[1]), 
                    xy_end[0] - xy_start[0], 
                    xy_end[1] - xy_start[1], 
                    linewidth=1, edgecolor='k', facecolor='gray'
                )
                plt.gca().add_patch(rect)
        
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Mission Space at {t} sec from Agent {agent_id}")
        if display:
            plt.show()
            return
        else:
            return fig