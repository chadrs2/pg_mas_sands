from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation import SimEnv
from cooperative_search import UAV, CooperativeSearch


def straight_line_placement(N, min_x, max_x, min_y, max_y):
    x_values = np.linspace(min_x, max_x, N, dtype=int)
    y_values = np.full_like(x_values, max_y, dtype=int)  # Constant y value

    return x_values, y_values

def random_placement(N, min_x, max_x, min_y, max_y):
    x_values = np.random.uniform(min_x, max_x, N, dtype=int)
    y_values = np.random.uniform(min_y, max_y, N, dtype=int)

    return x_values, y_values

def generate_curved_line_image(nrows, ncols, 
                               circle_center, circle_radius, circle_width, 
                               decay_rate, hollow_radius):
    """
    Generates a 2D probability map with a high probability ridge along a circle.

    Args:
        (ncols, nrows): Tuple representing the size (width, height) of the map.
        circle_center: Tuple representing the (x, y) coordinates of the circle's center (bottom left is origin).
        circle_radius: Float representing the radius of the circle.
        circle_width: Float representing the width of the high probability ridge.
        decay_rate: Float representing the exponential decay rate outside the ridge.

    Returns:
        A 2D NumPy array representing the probability map.
    """
    x, y = np.ogrid[:ncols, :nrows]
    distance_from_center = np.sqrt(((x - circle_center[0])**2) + ((y - circle_center[1])**2))

    # Base probability within the circle
    base_prob = np.clip(1 - (distance_from_center - circle_radius) / circle_width, 0, 1)
    hollow_mask = distance_from_center <= hollow_radius
    base_prob[hollow_mask] = 0

    # Apply exponential decay outside the circle
    decay_mask = distance_from_center > circle_radius
    decayed_prob = base_prob[decay_mask] * np.exp(-decay_rate * (distance_from_center[decay_mask] - circle_radius))
    base_prob[decay_mask] = decayed_prob

    # Add probabilities everywhere
    base_prob += 1/(nrows*ncols)
    base_prob /= base_prob.sum()

    return base_prob

def main(args):
    np.random.seed(123)

    # Initialize agents and starting locations
    N = args.num_agents
    if args.heterogeneous_agents:
        min_x, max_x = (args.Rs+args.Rs//2), args.ncols - (args.Rs+args.Rs//2)
        min_y, max_y = (args.Rs+args.Rs//2), args.nrows - (args.Rs+args.Rs//2)
    else:
        min_x, max_x = args.Rs, args.ncols - args.Rs
        min_y, max_y = args.Rs, args.nrows - args.Rs
    if args.rand_mu0:
        xs, ys = random_placement(N,min_x,max_x,min_y,max_y)
    elif args.unif_mu0:
        x = np.linspace(min_x, max_x, int(np.sqrt(N)))
        y = np.linspace(min_y, max_y, int(np.sqrt(N)))
        xx, yy = np.meshgrid(x, y)
        xs = xx.ravel()
        ys = yy.ravel()
        N = int(np.sqrt(N)) * int(np.sqrt(N))
    else:
        xs, ys = straight_line_placement(N,min_x,max_x,min_y,max_y)
        
    if args.use_prior_map:
        print("Doing only cooperative motion")
        # prior_map = np.ones((args.nrows,args.ncols)) / (args.nrows*args.ncols)
        circle_center = (0, 0)  # Bottom left corner is (0, 0)
        circle_radius = args.nrows//2
        circle_width = args.nrows//4
        hollow_radius = circle_radius//2
        decay_rate = 1e-2
        prior_map = generate_curved_line_image(
            args.nrows, args.ncols, 
            circle_center, circle_radius, circle_width, decay_rate, hollow_radius
        )
    else:
        print("Doing both search and surveillance")
        prior_map = np.ones((args.nrows,args.ncols)) / (args.nrows*args.ncols)
    
    # Initialize Simulated Environment
    if args.add_obstacles:
        obstacles = [
            [(0,(3*args.nrows)//4), (args.ncols//2,(3*args.nrows)//4+2)],
            [(args.ncols//2,0), (args.ncols//2+2,args.nrows//4)],
        ]
        # Update prior map to have 0 probability @ obstacle locations
        for o in obstacles:
            xy_start, xy_end = o
            prior_map[xy_start[1]:xy_end[1], xy_start[0]:xy_end[0]] = 0.0
        # prior_map /= prior_map.sum()
        env = SimEnv(args.nrows, args.ncols, obstacles)
    else:
        env = SimEnv(args.nrows, args.ncols)
    
    # Initialize Agents
    agents = []
    for n in range(N):
        if args.heterogeneous_agents:
            Rs = args.Rs + np.random.randint(-args.Rs//2,args.Rs//2)
        else:
            Rs = args.Rs
        
        agents.append(
            UAV(n, np.array([xs[n],ys[n]]), Rs, prior_map)
        )
    coop_search = CooperativeSearch(prior_map)
    coop_search.add_agents(agents)
    env.plot_env(coop_search.agents, coop_search.eta_igt)

    # Run Cooperative Search with Multiple UAVs Algorithm (see Table 1)
    temp = args.temp
    progress_bar = tqdm(total=args.max_itr, desc=" ")
    overall_coverage_performance = []
    overall_coverage_performance.append(coop_search.compute_coverage_performance())
    itr = 0

    while (overall_coverage_performance[-1] < args.coverage_thresh) if args.use_prior_map \
            else (overall_coverage_performance[-1] > args.uncertainty_thresh):

        ## Optimal Coverage using binary log-linear learning
        for _ in range(N):
            vi = np.random.choice(N)
            trial_action = coop_search.sample_trial_action(vi)
            Ui_a_prev = coop_search.compute_curr_utility(vi)
            Ui_a_exp = coop_search.compute_exp_utility(vi,trial_action)
            p = np.array([
                np.exp(1/temp * Ui_a_prev) / ( np.exp(1/temp * Ui_a_prev) \
                                              + np.exp(1/temp * Ui_a_exp) ),
                np.exp(1/temp * Ui_a_exp) / ( np.exp(1/temp * Ui_a_prev) \
                                             + np.exp(1/temp * Ui_a_exp) )
            ])
            selected_idx = np.random.choice(2,p=p)
            if selected_idx == 0:
                selected_action = np.zeros((2,),dtype=int)
            else:
                selected_action = trial_action
            coop_search.update_agent_loc(vi, selected_action)
            
        if not args.use_prior_map:
            ## Sensor Observations and Information Fusion
            coop_search.sensor_obsv_and_fusion(Rc=args.Rs*4,pc=0.9,pf=0.3,kn=1)

        # Update description text with computed loss
        if args.use_prior_map:
            curr_perf = coop_search.compute_coverage_performance()
            progress_bar.set_description("Coverage Performance: {:.6f}".format(curr_perf))        
        else:
            curr_perf = coop_search.compute_avg_uncertainty()
            progress_bar.set_description("Average Uncertainty: {:.6f}".format(curr_perf))
        # Update progress bar
        progress_bar.update()
        overall_coverage_performance.append(curr_perf)
        itr += 1

        if itr >= args.max_itr:
            break
        if (args.num_epochs2plot > 0) and (itr % args.num_epochs2plot == 0):#% 25 == 0:
            if not args.use_prior_map:
                for n in range(N):
                    env.plot_env(coop_search.agents, coop_search.agents[n].eta_igt, 
                                 t=itr, agent_id=n, display=False)
                plt.show()
            else:
                env.plot_env(coop_search.agents, coop_search.eta_igt, t=itr)
    progress_bar.close()
    if not args.use_prior_map:
        for n in range(N):
            env.plot_env(coop_search.agents, coop_search.agents[n].eta_igt, 
                         t=itr, agent_id=n, display=False)
        plt.show()
    else:
        env.plot_env(coop_search.agents, coop_search.eta_igt, t=itr)
    
    plt.figure()
    plt.plot(overall_coverage_performance)
    plt.xlabel("Iterations")
    if args.use_prior_map:
        plt.ylabel("Overall performance of coverage")
    else:
        plt.ylabel("Average Uncertainty")
    plt.show()
    return


def setup_parser():
    parser = ArgumentParser()
    # Iteration Params
    parser.add_argument('--max_itr',default=100,type=int)

    # Environment Init
    parser.add_argument('--nrows',default=10,type=int,help="Number of rows of cells")
    parser.add_argument('--ncols',default=10,type=int,help="Number of columns of cells")
    parser.add_argument('--add_obstacles',action='store_true',help="Add rectangle obstacles")
    parser.add_argument('--num_epochs2plot',default=-1,type=int,help="Display plots every XX epochs")
    
    # Agent Init
    parser.add_argument('--use_prior_map',action='store_true',help="Use prior knowledge or not")
    parser.add_argument('--num_agents',default=5,type=int,help="Number of agents to simulate")
    parser.add_argument('--rand_mu0',action='store_true',help='Randomize starting agent positions')
    parser.add_argument('--unif_mu0',action='store_true',help='Uniformly spread starting agent positions')
    parser.add_argument('--Rs',default=5,type=float,help="Sensor range of agents")
    parser.add_argument('--heterogeneous_agents',action='store_true',help="Give agents random Rs's (i.e. Rs+rand(Rs/2))")
    parser.add_argument('--temp',default=0.2,type=float,help="Temperature parameter for utility calculations")

    # Convergence Params
    parser.add_argument('--coverage_thresh',default=0.9,help="Convergence criteria for Motion Coordination")
    parser.add_argument('--uncertainty_thresh',default=1e-4,help="Uncertainty convergence criteria for Cooperative Search")
    return parser

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
