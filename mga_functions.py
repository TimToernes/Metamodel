from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective
import numpy as np 
import time
import logging
import os 
#%% Helper functions

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'::
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list

#%%

def inital_solution(network,options):
    try :
        options['number_of_processes']
    except :
        options['number_of_processes'] = 1
    try :
        options['cpus']
    except :
        options['cpus'] = 2
    try :
        options['tmp_dir']
    except :
        options['tmp_dir'] = os.getcwd()
    # This function performs the initial optimization of the techno-economic PyPSA model
    print('starting initial solution')
    timer = time.time()
    logging.disable()
    # Solving network
    network.lopf(network.snapshots, 
                solver_name='gurobi',
                solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 2,
                                            #'NumericFocus' : 3,
                                            'method':2,
                                            'threads':options['cpus'],
                                            #'NumericFocus' : numeric_focus,
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                pyomo=False,
                keep_references=True,
                formulation='kirchhoff',
                solver_dir = options['tmp_dir']
                ),
    # initializing solutions class, to keep all network data
    network.old_objective = network.objective
    print('finished initial solution in {} sec'.format(time.time()-timer))
    return network

def mga_constraint(network,snapshots,options):
    scale = 1e-6
    # This function creates the MGA constraint 
    gen_capital_cost   = linexpr((scale*network.generators.capital_cost,get_var(network, 'Generator', 'p_nom'))).sum()
    gen_marginal_cost  = linexpr((scale*network.generators.marginal_cost,get_var(network, 'Generator', 'p'))).sum().sum()
    link_capital_cost  = linexpr((scale*network.links.capital_cost,get_var(network, 'Link', 'p_nom'))).sum()
    try :
        store_capital_cost = linexpr((scale*network.storage_units.capital_cost,get_var(network, 'StorageUnit', 'p_nom'))).sum()
        # total system cost
        cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,store_capital_cost,link_capital_cost]))
    except : 
        # total system cost
        cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,link_capital_cost]))
    # MGA slack
    if options['mga_slack_type'] == 'percent':
        slack = network.old_objective*options['mga_slack']+network.old_objective
    elif options['mga_slack_type'] == 'fixed':
        slack = options['baseline_cost']*options['mga_slack']+options['baseline_cost']

    define_constraints(network,cost_scaled,'<=',slack*scale,'GlobalConstraint','MGA_constraint')


def mga_objective(network,snapshots,direction,options):
    mga_variables = options['mga_variables']
    expr_list = []
    for i,variable in enumerate(mga_variables):
        if variable == 'transmission':
            expr_list.append(linexpr((direction[i],get_var(network,'Link','p_nom'))).sum())
        if variable == 'co2_emission':
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p').filter(network.generators.index[network.generators.type == 'ocgt']))).sum().sum())
        elif variable == 'H2' or variable == 'battery':
            expr_list.append(linexpr((direction[i],get_var(network,'StorageUnit','p_nom').filter(network.storage_units.index[network.storage_units.carrier == variable]))).sum())
        elif variable == 'wind' or variable == 'solar' or variable == 'ocgt': 
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p_nom').filter(network.generators.index[network.generators.type == variable]))).sum())
        else :
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p_nom').filter(network.generators.index[network.generators.index == variable]))))


    mga_obj = join_exprs(np.array(expr_list))
    #print(mga_obj)
    write_objective(network,mga_obj)

def extra_functionality(network,snapshots,direction,options):
    mga_constraint(network,snapshots,options)
    mga_objective(network,snapshots,direction,options)


def solve(network,options,direction):
    try :
        options['number_of_processes']
    except :
        options['number_of_processes'] = 1
    try :
        options['cpus']
    except :
        options['cpus'] = 2
    try :
        options['tmp_dir']
    except :
        options['tmp_dir'] = os.getcwd()
    try :
        network.old_objective
    except :
        network = inital_solution(network,options)


    stat = network.lopf(network.snapshots,
                            pyomo=False,
                            solver_name='gurobi',
                            solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 0,
                                            'ObjScale' : 1e6,
                                            'NumericFocus' : 3,
                                            'method':2,
                                            'threads':int(np.ceil(options['cpus']/options['number_of_processes'])),
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                            keep_references=True,
                            skip_objective=True,
                            formulation='kirchhoff',
                            solver_dir = options['tmp_dir'],
                            extra_functionality=lambda network,snapshots: extra_functionality(network,snapshots,direction,options))
    return network,stat

def filter_directions(directions,directions_searched):
    # Filter already searched directions out if the angle between the new vector and any 
    # previously sarched vector is less than 1e-2 radians
    obsolete_directions = []
    for direction,i in zip(directions,range(len(directions))):
        if any([abs(angle_between(dir_searched,direction))<1e-2  for dir_searched in directions_searched]) :
            obsolete_directions.append(i)
    directions = np.delete(directions,obsolete_directions,axis=0)

    if len(directions)>1000:
        directions = directions[0:1000]
    return directions

def run_mga(network,sol,options,eval_func):
    # This is the real MGA function
    MGA_convergence_tol = options['mga_convergence_tol']
    dim=len(options['mga_variables'])
    old_volume = 0 
    epsilon_log = [1]
    directions_searched = np.empty([0,dim])
    hull = None
    computations = 0

    while not all(np.array(epsilon_log[-2:])<MGA_convergence_tol) : # The last two itterations must satisfy convergence tollerence
        # Generate list of directions to search in for this batch
        if len(sol.gen_p)<=1 : # if only original solution exists, max/min directions are chosen
            directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
        else : # Otherwise search in directions normal to faces
            directions = np.array(hull.equations)[:,0:-1]
        # Filter directions for previously serched directions
        directions = filter_directions(directions,directions_searched)

        if len(directions)>0:
            # Start parallelpool of workers
            sol = eval_func(directions,network,options,sol)
        else :
            print('All directions searched')

        computations += len(directions)
        directions_searched = np.concatenate([directions_searched,directions],axis=0)

        # Saving data to avoid data loss
        sol.save_xlsx(options['data_file'])
        

        # Creating convex hull
        hull_points = sol.sum_vars[options['mga_variables']].values
        try :
            hull = ConvexHull(hull_points)#,qhull_options='Qs C-1e-32')#,qhull_options='A-0.99')
        except Exception as e: 
            print('did not manage to create hull first try')
            print(e)
            try :
                hull = ConvexHull(hull_points,qhull_options='Qx C-1e-32')
            except Exception as e:
                print('did not manage to create hull second try')
                print(e)


        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        epsilon_log.append(epsilon)
        print('####### EPSILON ###############')
        print(epsilon)
    print('performed {} computations'.format(computations))
    return sol