from hydrocnhs.abm import Base, read_factor

class ResDam_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = self.config["Attributes"]["release"]
        print("Initialize reservoir agent: {}".format(self.name))

    def act(self, outlet):
        factor = read_factor(self.config, outlet)

        # Release (factor should be 1)
        if factor < 0:
            print("Something is not right in ResDam agent.")
        elif factor > 0:
            # Q["SCOO"][t] is the resevoir inflow
            res_t = self.release[self.t]
            action = res_t
            return action

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
            # if current_date.month == 3 and current_date.day == 1:
            #     # run institutional dm (an object shared by all agents)
            #     agent.dm.make_dm(Q=dc.Q_routed, current_date=current_date, agents=agents)
            #     # actions is a list of daily diversion requests.
            #     agent.actions += agent.dm.div_D_req[ag]

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
        self.actions = self.config["Attributes"]["diversion"]
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
