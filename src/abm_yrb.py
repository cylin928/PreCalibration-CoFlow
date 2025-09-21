import os
import pickle
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from hydrocnhs.abm import Base, read_factor

user = "CL"#"cl2769"
wd = rf"C:\Users\{user}\OneDrive\Cornell\Proj_yrb_sa"
input_path = os.path.join(wd, r"inputs")


with open(os.path.join(input_path, "YRB_cali_abm_assigned.pickle"),
          "rb") as file:
    obv_assigned = pickle.load(file)

with open(os.path.join(input_path, "YRB_cali_abm_fixed_inputs.pickle"),
          "rb") as file:
    (init_obv_D_div, init_div_ref,
     div_Y_max, div_Y_min, ccurves, corr) = pickle.load(file)

class Base_just_for_read():
    """
    Agent_type class's available items:
    * name: agent's name.
    * config: agent's configuration dictionary the model file (.yaml).
    * start_date: datetime object.
    * data_length: length of the simulation.
    * data_collector: a container to store simulated data.
    * rn_gen: random number generator to ensure reproducibility (e.g., 
    * self.rn_gen.random()). Note that do NOT set a global random seed in 
    * this module! All type of random number should be created by "rn_gen."
    * dm: decision making object if assigned in the model file (.yaml).
    
     Decision-making class's available items:
    * start_date: datetime object.
    * data_length: length of the simulation.
    * abm: the ABM configuration dictionary from the model file (.yaml).
    * data_collector: a container to store simulated data.
    * rn_gen: random number generator to ensure reproducibility (e.g., 
    * self.rn_gen.random()). Note that do NOT set a global random seed in 
    * this module! All type of random number should be created by "rn_gen.
    """
    def __init__(self, **kwargs):
        for key in kwargs:  # Load back all the previous class attributions.
            setattr(self, key, kwargs[key])

class ResDam_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.config["Inputs"]
        self.assigned_actions = obv_assigned[self.name]
        print("Initialize reservoir agent: {}".format(self.name))

    def act(self, outlet):
        factor = read_factor(self.config, outlet)

        # Release (factor should be 1)
        if factor < 0:
            print("Something is not right in ResDam agent.")
        elif factor > 0:
            # Q["SCOO"][t] is the resevoir inflow
            res_t = self.assigned_actions[self.t]
            action = res_t
            return action

# Institutional dm class
class DivDM(Base):
    # Only called (make_dm) by first div agent
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        abm_config = self.abm_config
        
        self.has_run_in_date = None

        self.behavior_type = abm_config.get("BehaviorType")
        self.adaptive_type = abm_config.get("AdaptiveType")
        self.rng = pd.date_range(
            start=self.start_date, periods=self.data_length, freq="D"
            )
        self.ag_list = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
        self.ag_pars = {}
        self.dc.add_field("DivDM", {})
        records = self.dc.get_field("DivDM")
        self.div_D_req = {}
        for ag in self.ag_list:
            records[ag] = {
                "y": [],
                "x": [],
                "V": [],
                "Vavg": [],
                "Mu": [],
                "DivReqRef": [],
                "DivReq_Y": []
                }
            self.div_D_req[ag] = []


        # Scenario based inputs
        # flow_target [v1, v2]; v1 for y<1985, v2 for y>=1985
        self.flow_target = abm_config.get("FlowTarget")

        prec_11_8_db_path = abm_config.get("Database")
        prec_11_8_db = pd.read_csv(prec_11_8_db_path, index_col=0)
        self.prec_11_8_db = prec_11_8_db.to_numpy().flatten()
        self.start_year = self.start_date.year

        # Dynamic increment of min flow for learning agents
        self.increase_min_flow = 0

    def make_dm(self, Q, current_date, agents):
        """Output daily diversion requests for a year start from 3/1."""
        # Check if it has been run in the current date.
        # We make decisions for all five agents at once.
        if self.has_run_in_date == current_date:
            return None
        else:
            self.has_run_in_date = current_date
        
        # Here start the code for the decision making.
        current_year = current_date.year
        start_year = self.start_year
        bt = self.behavior_type
        at = self.adaptive_type
        rng = self.rng
        if current_year-1 < 1985:
            flow_target = self.flow_target[0]
        else:
            flow_target = self.flow_target[1]
        ag_list = self.ag_list
        records = self.dc.get_field("DivDM")

        #==================================================
        # Learning
        #==================================================
        if bt == "Learning":
            # calculate mean flow789 of last year (y)
            if current_year == start_year:
                # Initial value (No deviation from the flow target.)
                y = flow_target
            else:
                mask = [True if (i.month in [7,8,9]
                                 and (i.year == current_year-1)) \
                        else False for i in rng]
                y = np.mean(Q["G"][mask])

            increase_min_flow = []
            for ag in ag_list:
                records[ag]["y"].append(y)
                ag_par = agents[ag].config["Pars"]

                try:
                    div_req_ref = records[ag]["DivReqRef"][-1]
                except: # initial
                    print("Use initial DivReqRef.")
                    div_req_ref = init_div_ref[ag]

                # Update DivReqRef
                if y > flow_target + ag_par["L_U"]:
                    V = 1
                elif y < flow_target - ag_par["L_L"]:
                    V = -1
                else:
                    V = 0
                records[ag]["V"].append(V)

                # Mean value (strength) of the past "ten" years.
                # First few years, strengh is decreased on purpose.
                Vavg = np.sum(records[ag]["V"][-10:])/10
                records[ag]["Vavg"].append(Vavg)

                # Scale to 5 Lr_c in [0,1]
                div_req_ref = div_req_ref + Vavg*ag_par["Lr_c"]*5
                # Bound by Max and Min
                div_req_ref = min(div_Y_max[ag],
                                  max(div_req_ref,  div_Y_min[ag]))
                records[ag]["DivReqRef"].append(div_req_ref)

                # Calculate the increase_min_flow
                increase_min_flow.append(div_req_ref - init_div_ref[ag])
            self.increase_min_flow = sum(increase_min_flow)
        else:   # For non-learning ABM.
            for ag in ag_list:
                records[ag]["DivReqRef"].append(init_div_ref[ag])

        #==================================================
        # Adaptive & Emergency Operation (Drought year proration)
        #==================================================
        #--- Get feature: Annual daily mean BCPrec from 11 - 6
        x = self.prec_11_8_db[current_year - start_year]

        #--- Get Multinormal random noise
        if bt != "Static":
            rn = self.rn_gen.multivariate_normal(mean=[0]*5, cov=corr)

        #--- Get YDivReq
        for i, ag in enumerate(ag_list):
            ag_par = agents[ag].config["Pars"]
            div_req_ref = records[ag]["DivReqRef"][-1]

            # Emergency Operation (Drought year proration)
            if x <= 0.583:
                div_Y_req = div_req_ref * ag_par["ProratedRatio"]
                mu = div_Y_req
                if bt != "Static":
                    div_Y_req = mu + rn[i] * ag_par["Sig"]
            else:
                if bt == "Static":
                    div_Y_req = div_req_ref + ag_par["b"]
                    mu = div_Y_req
                elif at == "Linear":
                    mu = div_req_ref + ag_par["a"] * x + ag_par["b"]
                    div_Y_req = mu + rn[i] * ag_par["Sig"]
                elif at == "Quadratic":
                    mu = div_req_ref + ag_par["a"] * x**2 + ag_par["b"] * x \
                        + ag_par["c"]
                    div_Y_req = mu + rn[i] * ag_par["Sig"]

            # Hard constraint for MaxYDiv and MinYDiv
            div_Y_req = min(div_Y_max[ag],
                              max(div_Y_req,  div_Y_min[ag]))
            records[ag]["x"].append(x)
            try:
                records[ag]["Mu"].append(mu)
            except:
                print("QQQ")
            records[ag]["DivReq_Y"].append(div_Y_req)

        #==================================================
        # To Daily
        #==================================================

        #--- Map back to daily diversion (from Mar to Feb)
        def getMonthlyDiv(div_Y_req, a, b, LB, UB):
            if div_Y_req <= LB:
                div_M_req = b
            elif div_Y_req >= UB:
                div_M_req = a*(UB-LB) + b
            else:
                div_M_req = a*(div_Y_req-LB) + b
            return div_M_req

        # From Mar to Feb
        if (current_date + relativedelta(years=1)).is_leap_year :
            days_in_months = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29]
        else:
            days_in_months = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28]

        for ag in ag_list:
            div_Y_req = records[ag]["DivReq_Y"][-1]

            #--- To monthly.
            M_ratio = np.array([getMonthlyDiv(div_Y_req, *ccurves[ag][m-1]) \
                                for m in [3,4,5,6,7,8,9,10,11,12,1,2]])
            M_ratio = M_ratio / sum(M_ratio)
            div_M_req = div_Y_req * 12 * M_ratio

            #--- To daily. Uniformly assign those monthly average diversion to
            #    each day.
            div_D_req = []
            for m in range(12):
                div_D_req += [div_M_req[m]] * days_in_months[m]

            #--- Each agent will get div_D_req by calling self.div_D_req[ag]
            #    from DivDM class.
            self.div_D_req[ag] = div_D_req
        #==================================================

# Additional auxiliary class for redistribution Roze, Wapato, Sunnyside
class DivRedistribution_RWS():
    def __init__(self) -> None:
        self.has_run_in_date = None
        self.ag_list = ['Roza', 'Wapato', 'Sunnyside']
        # wr_ratio = proratable water right / total water right.
        self.wr_ratio = [1, 0.533851525, 0.352633532]
        pass
    def redistribute(self, dc, agents, outlet, pre_date, current_date, t, data_length):
        # If it has been run in the current date, return None.
        if self.has_run_in_date == current_date:
            return None
        else:
            self.has_run_in_date = current_date
        
        # Do the redistribution and save them in this object for retrieving
        ### Collect diversion requests of all agents in the group.
        div_reqs_t = []
        div_reqs_t_no_remain = []
        div_reqs_acc = []
        for ag in self.ag_list:
            
            records = dc.get_field(ag)
            # Get initial diversion requests
            if current_date.month != pre_date.month or t+1 == data_length:
                shortage_M = records["Shortage_D"][-1] / pre_date.days_in_month
                records["Shortage_M"].append(shortage_M)
                records["AccMDivReq"] = 0
                remain_Req = 0
            else:
                if records["Shortage_D"] == []:
                    remain_Req = 0
                else:
                    remain_Req = records["Shortage_D"][-1]
            
            agent = agents[ag]
            if current_date.month == 3 and current_date.day == 1:
                # run institutional dm (an object shared by all agents)
                agent.dm.make_dm(Q=dc.Q_routed, current_date=current_date, agents=agents)
                # actions is a list of daily diversion requests.
                agent.actions += agent.dm.div_D_req[ag]

            # Get action (request) for the current day
            div_dm = agent.actions[t]
            records["AccMDivReq"] += div_dm
            div_reqs_acc.append(records["AccMDivReq"])
            div_req_t = div_dm + remain_Req
            div_reqs_t.append(div_req_t)
            div_reqs_t_no_remain.append(div_dm)
            records["DivReq"].append(div_req_t)
            records["Qup"].append(dc.Q_routed[outlet][t])

        ### Calculate actual total diversion as a group.
        total_div_req_t = sum(div_reqs_t)
        min_flow = 3.53     # [cms]
        available_water_t = max(0, dc.Q_routed[outlet][t] - min_flow)
        if total_div_req_t > available_water_t:
            total_shortage_t = total_div_req_t - available_water_t
            total_div_t = available_water_t
        else:
            total_div_t = total_div_req_t
            total_shortage_t = 0

        ### Disaggregate group value into each agents in the group.
        p_reqs = [self.wr_ratio[i] * div_reqs_acc[i] for i in range(3)]
        total_p_reqs = sum(p_reqs)
        if total_p_reqs == 0:   # Avoid dividing zero.
            r_p_reqs = [0, 0, 0]
        else:
            r_p_reqs = [p_req / total_p_reqs for p_req in p_reqs]
        shortages_t = [total_shortage_t * r_p_req for r_p_req in r_p_reqs]

        ### Redistribute exceed shortages_t
        # Need to make sure shortage does not exceed monthly accumulated req.
        if shortages_t[0] > div_reqs_acc[0]:  # Roza
            redistritute = shortages_t[0] - div_reqs_acc[0]
            shortages_t[0] = div_reqs_acc[0]
            # To Wapato & Sunnyside
            total_rr_reqs = div_reqs_acc[1] + div_reqs_acc[2]
            shortages_t[1] += (div_reqs_acc[1]/total_rr_reqs * redistritute)
            shortages_t[2] += (div_reqs_acc[2]/total_rr_reqs * redistritute)
        if shortages_t[1] > div_reqs_acc[1]:  # Wapato
            redistritute = shortages_t[1] - div_reqs_acc[1]
            shortages_t[1] = div_reqs_acc[1]
            # To Sunnyside
            shortages_t[2] += redistritute
        if shortages_t[2] > div_reqs_acc[2]:  # Sunnyside
            if abs(shortages_t[2]-div_reqs_acc[2]) <= 10**(-5):
                shortages_t[2] = div_reqs_acc[2]
            else:
                print("Error! shortage distribution.")

        ### Record agents values
        q_down = dc.Q_routed[outlet][t] - total_div_t
        for i, ag in enumerate(self.ag_list):
            records = dc.get_field(ag)
            div_t = div_reqs_t[i] - shortages_t[i]
            records["Shortage_D"].append(shortages_t[i])
            records["Div"].append(div_t)
            records["Qdown"].append(q_down)
        return None
div_redistributor = DivRedistribution_RWS()

class IrrDiv_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.config["Inputs"]
        self.pars = self.config["Pars"]
        self.dc.add_field(self.name, {})
        records = self.dc.get_field(self.name)
        records["DivReq"] = []
        records["Div"] = []
        records["Shortage_D"] = []
        records["Qup"] = []
        records["Qdown"] = []
        records["ReFlow"] = []
        records["Shortage_M"] = []
        records["AccMDivReq"] = 0 # For redistribution
        self.pre_date = self.start_date
        self.data_length = self.data_length
        #self.assigned_actions = obv_assigned[name]
        self.actions = init_obv_D_div[self.name]
        print("Initialize irrigation diversion agent: {}".format(self.name))

        # Add div_redistributor if it is Roza, Wapato, Sunnyside
        # For proportionally redistribution of exceed shortage.
        if self.name in ["Roza", "Wapato", "Sunnyside"]:
            self.div_redistributor = div_redistributor  # Roza, Wapato, Sunnyside
        else:
            self.div_redistributor = None

        # Initialize DivDM (.yaml) if it is Tieton (C2 will be simulate first.)
        # self.dm

    def act(self, outlet):
        records = self.dc.get_field(self.name)
        data_length = self.data_length

        # Get factor
        factor = read_factor(self.config, outlet)

        # Compute actual diversion or return flow
        if factor < 0:  # Diversion
            if self.name in ["Tieton", "Kittitas"]:
                # We consider monthly shortage. Therefore the daily shortage will
                # be carried over to the next day until the end of the month.
                if self.current_date.month != self.pre_date.month or self.t+1 == data_length:
                    shortage_M = records["Shortage_D"][-1] / self.pre_date.days_in_month
                    records["Shortage_M"].append(shortage_M)
                    remain_Req = 0
                else:
                    if records["Shortage_D"] == []:
                        remain_Req = 0
                    else:
                        remain_Req = records["Shortage_D"][-1]
                
                # Make diversion request decision at March 1st.
                if self.current_date.month == 3 and self.current_date.day == 1:
                    # run institutional dm (an object shared by all agents)
                    self.dm.make_dm(Q=self.dc.Q_routed, current_date=self.current_date, agents=self.agents)
                    # actions is a list of daily diversion requests.
                    self.actions += self.dm.div_D_req[self.name]

                #div_req_t = self.assigned_actions[t] + remain_Req
                div_req_t = self.actions[self.t] + remain_Req
                min_flow = 3.53     # [cms]
                available_water_t = max(0, self.dc.Q_routed[outlet][self.t] - min_flow)
                if div_req_t > available_water_t:
                    shortage_t = div_req_t - available_water_t
                    div_t = available_water_t
                else:
                    div_t = div_req_t
                    shortage_t = 0

                records["Qup"].append(self.dc.Q_routed[outlet][self.t])
                records["DivReq"].append(div_req_t)
                records["Qdown"].append(self.dc.Q_routed[outlet][self.t] - div_t)
                records["Div"].append(div_t)
                records["Shortage_D"].append(shortage_t)

            # Has the redistribution component
            elif self.name in ["Roza", "Wapato", "Sunnyside"]:
                self.div_redistributor.redistribute(
                    dc=self.dc, agents=self.agents, outlet=outlet, 
                    pre_date=self.pre_date, current_date=self.current_date, 
                    t=self.t, data_length=self.data_length
                )
                # Get the assigned action from div_redistributor
                div_t = records["Div"][-1]

            action = factor * div_t
            self.pre_date = self.current_date
        
        ##### Return flow
        else:           
            div_t = records["Div"][self.t]
            action = factor * div_t
            records["ReFlow"] = action
        return action