import pandas as pd
import numpy as np
from multiprocessing import  Queue

class solutions:

    # the solutions class contains all nececary data for all MGA solutions
    # The class also contains functions to append new solutions and to save the results

    def __init__(self,network):
        self.old_objective = network.objective
        self.sum_vars = self.calc_sum_vars(network)
        self.gen_p =    pd.DataFrame(data=[network.generators.p_nom_opt],index=[0])
        self.gen_E =    pd.DataFrame(data=[network.generators_t.p.sum()],index=[0])
        self.store_p =  pd.DataFrame(data=[network.storage_units.p_nom_opt],index=[0])
        self.store_E =  pd.DataFrame(data=[network.storage_units_t.p.sum()],index=[0])
        self.links =    pd.DataFrame(data=[network.links.p_nom_opt],index=[0])
        self.secondary_metrics = self.calc_secondary_metrics(network)
        self.objective = pd.DataFrame()

        self.df_list = {'gen_p':self.gen_p,
                        'gen_E':self.gen_E,
                        'store_E':self.store_E,
                        'store_p':self.store_p,
                        'links':self.links,
                        'sum_vars':self.sum_vars,
                        'secondary_metrics':self.secondary_metrics}

        try :
            co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
        except :
            co2_emission = 0 
        

    def append(self,network):
        # Append new data to all dataframes
        self.sum_vars = self.sum_vars.append(self.calc_sum_vars(network),ignore_index=True)
        self.gen_p =    self.gen_p.append([network.generators.p_nom_opt],ignore_index=True)
        self.links =    self.gen_p.append([network.links.p_nom_opt],ignore_index=True)
        self.gen_E =    self.gen_E.append([network.generators_t.p.sum()],ignore_index=True)
        self.secondary_metrics = self.secondary_metrics.append(self.calc_secondary_metrics(network),ignore_index=True)

    def calc_secondary_metrics(self,network):
        # Calculate secondary metrics
        gini = self.calc_gini(network)
        co2_emission = self.calc_co2_emission(network)
        system_cost = self.calc_system_cost(network)
        autoarky = self.calc_autoarky(network)
        return pd.DataFrame({'system_cost':system_cost,'co2_emission':co2_emission,'gini':gini,'autoarky':autoarky},index=[0])

    def calc_sum_vars(self,network):
        sum_data = dict(network.generators.p_nom_opt.groupby(network.generators.type).sum())
        sum_data['transmission'] = network.links.p_nom_opt.sum()
        sum_data['co2_emission'] = self.calc_co2_emission(network)
        sum_data.update(network.storage_units.p_nom_opt.groupby(network.storage_units.carrier).sum())
        sum_vars = pd.DataFrame(sum_data,index=[0])
        return sum_vars

    def put(self,network):
    # add new data to the solutions queue. This is used when new data is added from 
    # sub-process, when using multiprocessing 
        try :
            self.queue.qsize()
        except : 
            print('creating queue object')
            self.queue = Queue()

        part_result = solutions(network)
        self.queue.put(part_result,block=True,timeout=120)

    def init_queue(self):
        # Initialize results queue 
        try :
            self.queue.qsize()
        except : 
            self.queue = Queue()

    def merge(self):
        # Merge all solutions put into the solutions queue into the solutions dataframes
        merge_num = self.queue.qsize()
        while not self.queue.empty() :
            part_res = self.queue.get(60)
            self.gen_E = self.gen_E.append(part_res.gen_E,ignore_index=True)
            self.gen_p = self.gen_p.append(part_res.gen_p,ignore_index=True)
            self.store_E = self.store_E.append(part_res.store_E,ignore_index=True)
            self.store_p = self.store_p.append(part_res.store_p,ignore_index=True)
            self.links = self.links.append(part_res.links,ignore_index=True)
            self.sum_vars = self.sum_vars.append(part_res.sum_vars,ignore_index=True)
            self.secondary_metrics = self.secondary_metrics.append(part_res.secondary_metrics,ignore_index=True)
        print('merged {} solution'.format(merge_num))

    def save_xlsx(self,file='save.xlsx'):
        # Store all dataframes als excel file
        self.df_list = {'gen_p':self.gen_p,
                'gen_E':self.gen_E,
                'store_E':self.store_E,
                'store_p':self.store_p,
                'links':self.links,
                'sum_vars':self.sum_vars,
                'secondary_metrics':self.secondary_metrics}

        writer = pd.ExcelWriter(file)
        sheet_names =  ['gen_p','gen_E','links','sum_var','secondary_metrics']
        for i, df in enumerate(self.df_list):
            self.df_list[df].to_excel(writer,df)
        writer.save() 
        print('saved {}'.format(file))

    def calc_gini(self,network):
    # This function calculates the gini coefficient of a given PyPSA network. 
        bus_total_prod = network.generators_t.p.sum().groupby(network.generators.bus).sum()
        load_total= network.loads_t.p_set.sum()

        rel_demand = load_total/sum(load_total)
        rel_generation = bus_total_prod/sum(bus_total_prod)
        
        # Rearange demand and generation to be of increasing magnitude
        idy = np.argsort(rel_generation/rel_demand)
        rel_demand = rel_demand[idy]
        rel_generation = rel_generation[idy]

        # Calculate cumulative sum and add [0,0 as point
        rel_demand = np.cumsum(rel_demand)
        rel_demand = np.concatenate([[0],rel_demand])
        rel_generation = np.cumsum(rel_generation)
        rel_generation = np.concatenate([[0],rel_generation])

        lorenz_integral= 0
        for i in range(len(rel_demand)-1):
            lorenz_integral += (rel_demand[i+1]-rel_demand[i])*(rel_generation[i+1]-rel_generation[i])/2 + (rel_demand[i+1]-rel_demand[i])*rel_generation[i]
        
        gini = 1- 2*lorenz_integral
        return gini

    def calc_autoarky(self,network):
        # calculates the autoarky of a model solution 
        # autoarky is calculated as the mean self sufficiency (energy produced/energy consumed) of all countries in all hours
        mean_autoarky = []
        for snap in network.snapshots:
            hourly_load = network.loads_t.p_set.loc[snap]
            hourly_autoarky = network.generators_t.p.loc[snap].groupby(network.generators.bus).sum()/hourly_load
            hourly_autoarky_corected = hourly_autoarky.where(hourly_autoarky<1,1)
            mean_autoarky.append(np.mean(hourly_autoarky_corected))
        return np.mean(mean_autoarky)

    def calc_co2_emission(self,network):
            #CO2
        id_ocgt = network.generators.index[network.generators.type == 'ocgt']
        co2_emission = network.generators_t.p[id_ocgt].sum().sum()*network.carriers.co2_emissions['ocgt']/network.generators.efficiency.loc['AT ocgt']
        co2_emission
        return co2_emission

    def calc_system_cost(self,network):
        #Cost
        capital_cost = sum(network.generators.p_nom_opt*network.generators.capital_cost) + sum(network.links.p_nom_opt*network.links.capital_cost) + sum(network.storage_units.p_nom_opt * network.storage_units.capital_cost)
        marginal_cost = network.generators_t.p.groupby(network.generators.type,axis=1).sum().sum() * network.generators.marginal_cost.groupby(network.generators.type).mean()
        total_system_cost = marginal_cost.sum() + capital_cost
        return total_system_cost