import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import hydrocnhs

__all__ = [
    "InputBuilder",
    "YRBModel",
    "DP",
    ]
def read_json(file_path):
    """
    Read data from a JSON file.
    
    Parameters
    ==========
    file_path : str
        Path to the input JSON file.

    Returns
    =======
    dict
        Data read from the JSON file.
    """
    import json
    with open(file_path, "r") as file:
        data = json.load(file)
    
    return data

def read_parquet(file_path, key_list=None):
    """
    Read data from a Parquet file or a collection of Parquet files.
    
    Parameters
    ==========
    file_path : str or Path
        Base path to the input Parquet file(s). If key_list is provided, expects files like 
        "<file_path>_<key>.parquet" for each key.
    key_list : list of str, optional
        List of keys to identify multiple files. If provided, reads multiple files into a dictionary.
    
    Returns
    =======
    pd.DataFrame or dict of pd.DataFrame
        Data read from the Parquet file(s). Returns a DataFrame for a single file,
        or a dictionary of DataFrames for multiple files.
    """
    import pandas as pd
    from pathlib import Path
    
    file_path = Path(file_path)

    if key_list is not None:
        data = {}
        for key in key_list:
            full_path = file_path.parent / f"{file_path.stem}_{key}.parquet"
            data[key] = pd.read_parquet(full_path)
        return data
    else:
        return pd.read_parquet(file_path)

class InputBuilder:
    def __init__(self, pn):
        self.pn = pn
        self.cfs_to_cms = 0.028316832

    def load_csv_files(self):
        pn = self.pn
        cfs_to_cms = self.cfs_to_cms

        # Load csv
        df_pr = pd.read_csv(pn.data.gridmet.get("gridmet_D_pr_cm.csv"), parse_dates=True, index_col=[0])
        df_pet = pd.read_csv(pn.data.gridmet.get("gridmet_D_pet_cm.csv"), parse_dates=True, index_col=[0])
        df_tmax = pd.read_csv(pn.data.gridmet.get("gridmet_D_tmax_degc.csv"), parse_dates=True, index_col=[0])
        df_tmin = pd.read_csv(pn.data.gridmet.get("gridmet_D_tmin_degc.csv"), parse_dates=True, index_col=[0])
        df_temp = (df_tmax + df_tmin) / 2

        df_flow = pd.read_csv(pn.data.hydromet.get("hydromet_reservoir_inflow_cfs.csv"), parse_dates=True, index_col=[0]) * cfs_to_cms
        df_flow = df_flow[["Upper", "BUM", "RIM"]]
        df_flow.columns = ["S1", "S2", "S3"]

        df_streamflow = pd.read_csv(pn.data.hydromet.get("hydromet_gauges_regulated_flow_cfs.csv"), parse_dates=True, index_col=[0]) * cfs_to_cms
        df_streamflow = df_streamflow[['NACW', 'YUMW', 'UMTW', 'PARW']]
        df_streamflow.columns = ['C2', 'YUMW', 'C1', 'G']
        df_div = pd.read_csv(pn.data.hydromet.get("hydromet_canal_diversion_cfs.csv"), parse_dates=True, index_col=[0]) * cfs_to_cms
        df_div = df_div[['KTCW', 'ROZW', 'RSCW', 'SNCW', 'TIEW']]
        df_div.columns = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
        df_release = pd.read_csv(pn.data.hydromet.get("hydromet_reservoir_release_cfs.csv"), parse_dates=True, index_col=[0]) * cfs_to_cms
        df_release = df_release[["Upper", "BUM", "RIM"]]
        df_release.columns = ["R1", "R2", "R3"]

        df_minor_div = pd.read_csv(pn.data.get("minor_div_D_cms.csv"), parse_dates=True, index_col=[0])

        #ccurves
        ccurves = {}
        for ag in ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']:
            path = pn.data.ccurves.get(f"MRatio_{ag}.csv")
            ccurves[ag] = pd.read_csv(path, index_col=0).to_numpy().tolist()

        self.df_pr = df_pr
        self.df_pet = df_pet
        self.df_temp = df_temp
        self.df_flow = df_flow
        self.df_streamflow = df_streamflow
        self.df_div = df_div
        self.df_release = df_release
        self.ccurves = ccurves
        self.df_minor_div = df_minor_div

    def to_hdf5(self, dict_of_dfs, filename):
        # Save everything to HDF5 ().h5 file)
        with pd.HDFStore(filename) as store:
            for key, df in dict_of_dfs.items():
                store[key] = df

    def to_json(self, my_dict, filename):
        with open(filename, "w") as f:
            json.dump(my_dict, f)

    def load_hdf5(self, filename, to_dict=True):
        # Load everything from HDF5 file
        with pd.HDFStore(filename) as store:
            dict_of_dfs = {key.strip('/'): store[key] for key in store.keys()}
            if to_dict:
                for key, df in dict_of_dfs.items():
                    if key in ['temp', 'prec', 'pet', 'releases', 'diversions']:
                        dict_of_dfs[key] = df.to_dict(orient='list')
            return dict_of_dfs

    def load_json(self, filename):
        with open(filename, "r") as f:
            my_dict = json.load(f)
        return my_dict

    def create_inputs(self, start="1979/1/1", end="2023/12/31", agt_list=['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']):
        df_pr = self.df_pr
        df_pet = self.df_pet
        df_temp = self.df_temp
        df_flow = self.df_flow
        df_streamflow = self.df_streamflow
        df_div = self.df_div
        df_release = self.df_release
        df_minor_div = self.df_minor_div

        pn = self.pn

        # Create inputs for S calibration
        for o in ["S1", "S2", "S3"]:
            temp = df_temp[[o]][start:end]
            prec = df_pr[[o]][start:end]
            pet = df_pet[[o]][start:end]
            obv_flow = df_flow[[o]][start:end].resample("MS").mean()

            inputs = {'temp': temp, 'prec': prec, 'pet': pet, 'obv_flow': obv_flow}
            self.to_hdf5(inputs, pn.inputs.get() / f"inputs_yrb_{o}_1979_2023.h5")

        # Create inputs for CCG calibration
        outlets = ["C1", "C2", "G"]
        temp = df_temp[outlets][start:end]
        prec = df_pr[outlets][start:end]
        pet = df_pet[outlets][start:end]
        obv_flow = df_streamflow[outlets][start:end].resample("MS").mean()
        obv_div = df_div[agt_list][start:end].resample("YS").mean()
        inputs = {'temp': temp, 'prec': prec, 'pet': pet, 'obv_flow': obv_flow, 'obv_div': obv_div}
        self.to_hdf5(inputs, pn.inputs.get() / "inputs_yrb_CCG_1979_2023.h5")

        # For sa, when minor divs is available
        minor_divs_M = df_minor_div.resample("MS").mean()[start:end].round(4) # Daily
        inputs = {'temp': temp, 'prec': prec, 'pet': pet, 'obv_flow': obv_flow, 'obv_div': obv_div, 'minor_divs_M': minor_divs_M}
        self.to_hdf5(inputs, pn.inputs.get() / "inputs_yrb_CCG_1979_2023_with_minor_divs.h5")

        # Create inputs for coupledABM calibration
        outlets = ["C1", "C2", "G"]
        temp = df_temp[outlets][start:end]
        prec = df_pr[outlets][start:end]
        pet = df_pet[outlets][start:end]
        obv_flow = df_streamflow[outlets][start:end].resample("MS").mean()
        obv_div = df_div[agt_list][start:end].resample("YS").mean()
        inputs = {'temp': temp, 'prec': prec, 'pet': pet, 'obv_flow': obv_flow, 'obv_div': obv_div}
        self.to_hdf5(inputs, pn.inputs.get() / "inputs_yrb_coupledABM_1979_2023.h5")

        # ABM inputs
        releases = df_release[start:end].round(4)
        diversions = df_div[start:end].round(4)
        inputs = {'releases': releases, 'diversions': diversions}
        self.to_hdf5(inputs, pn.inputs.get() / "inputs_yrb_CCG_1979_2023_abm.h5")

        # ABM data for coupledABM
        df_filtered = df_pr.loc[df_pr.index.month.isin([11, 12, 1, 2, 3, 4, 5, 6]), ["S1", "S2", "S3"]].copy()
        ## Adjust the year: Nov-Dec should be associated with the *next* year
        df_filtered['water_year'] = df_filtered.index.year
        df_filtered.loc[df_filtered.index.month.isin([11, 12]), 'water_year'] += 1
        ## Sum over each water year
        df_nov_to_jun_sum = df_filtered.groupby('water_year').sum()
        ## Set water year as index (optional: rename index for clarity)
        df_nov_to_jun_sum.index.name = 'year'
        df_nov_to_jun_sum = df_nov_to_jun_sum.sum(axis=1).to_frame("pr_nov_to_jun")
        df_nov_to_jun_sum.loc[1979, "pr_nov_to_jun"] += 6.34 # cm 1978 S1, S2, S3 Nov, Dec from BC data from EF paper
        df_nov_to_jun_sum = df_nov_to_jun_sum.loc[int(start[:4]):int(end[:4]),:].round(2)

        # Initial Diversion req ref (1978)
        init_div_ref = df_div.loc["1978-01-01":"1978-12-31"].mean().to_dict()
        df_div_Y = df_div.resample("YS").mean()[start:end].round(2)
        df_div_Y_max = (df_div_Y.max()*1.2).to_dict()
        df_div_Y_min = (df_div_Y.min()*0.8).to_dict()
        corr = df_div_Y.corr()
        corr = corr.loc[agt_list, agt_list]
        ccurves = self.ccurves

        minor_divs = df_minor_div[start:end].round(4)

        inputs = {
            'releases': releases.to_dict(orient="list"),
            'diversions': diversions.to_dict(orient="list"),
            "pr_nov_to_jun_sum": df_nov_to_jun_sum.loc[:, "pr_nov_to_jun"].values.tolist(),
            "init_div_ref": init_div_ref,
            "div_Y_max": df_div_Y_max,
            "div_Y_min": df_div_Y_min,
            "corr": corr.values.tolist(),
            "ccurves": ccurves,
            "minor_divs": minor_divs["G"].values.tolist(),
            }
        self.to_json(inputs, pn.inputs.get() / "inputs_yrb_coupledABM_1979_2023_abm.json")

class YRBModel:
    def __init__(self, pn):
        self.pn = pn

    def make_model_template_S(self, start="1979/1/1", end="2023/12/31", runoff_model="GWLF"):
        pn = self.pn

        # Create model template for S calibration
        outlet_list = ["S1", "S2", "S3"]
        area_list = [83014.25, 11601.47, 28016.2]
        lat_list = [47.416, 46.814, 46.622]

        for o, a, l in zip(outlet_list, area_list, lat_list):
            mb = hydrocnhs.ModelBuilder(wd="")
            mb.set_water_system(start_date=start, end_date=end)
            mb.set_rainfall_runoff(
                outlet_list=[o],
                area_list=[a],
                lat_list=[l],
                runoff_model=runoff_model
                )
            mb.set_routing_outlet(
                routing_outlet=o,
                upstream_outlet_list=[]
                )

            mb.write_model_to_yaml(
                str(pn.model.join(f"cali_yrb_{o}_{runoff_model}.yaml"))
            )
        print("Model templates for S calibration created.")

    def make_model_template_CCG_hydro_only(
        self, start="1979/1/1", end="2023/12/31",
        inputs_yrb_CCG_abm="inputs_yrb_CCG_1979_2023_abm.h5",
        runoff_model="GWLF"
        ):
        """Create a model template for CCG hydrological model calibration."""
        pn = self.pn
        ib = InputBuilder(pn)
        inputs = ib.load_hdf5(filename=pn.inputs.get(inputs_yrb_CCG_abm), to_dict=True)
        releases = inputs["releases"]
        diversions = inputs["diversions"]

        mb = hydrocnhs.ModelBuilder(wd="")
        mb.set_water_system(start_date=start, end_date=end)
        mb.set_rainfall_runoff(
            outlet_list=["S1", "S2", "S3"],
            area_list=[83014.25, 11601.47, 28016.2],
            lat_list=[47.416, 46.814, 46.622],
            runoff_model=runoff_model,
            activate=False
            )
        mb.set_rainfall_runoff(
            outlet_list=["C1", "C2", "G"],
            area_list=[328818.7, 203799.79, 291203.8],
            lat_list=[47.145, 46.839, 46.682],
            runoff_model=runoff_model
            )
        mb.set_routing_outlet(
            routing_outlet="G",
            upstream_outlet_list=["C1", "C2"],
            flow_length_list=[59404.82, 48847.74]
            )
        mb.set_routing_outlet(
            routing_outlet="C1",
            upstream_outlet_list=["R1"],
            flow_length_list=[100364.29],
            instream_objects=["R1"]
            )
        mb.set_routing_outlet(
            routing_outlet="C2",
            upstream_outlet_list=["R2", "R3"],
            flow_length_list=[70713.85, 36293.57],
            instream_objects=["R2", "R3"]
            )
        for s in ["S1", "S2", "S3"]:
            mb.set_routing_outlet(routing_outlet=s, upstream_outlet_list=[], activate=False)

        mb.set_ABM(
            abm_module_folder_path=pn.src.get_str(),
            abm_module_name="abm_yrb_with_res_div_as_inputs_for_ccg_cali.py"
            )

        # Add additional info
        mb.model["WaterSystem"]["ABM"]["div_dict"] = "should assigned this in the simulation setup from inputs"

        for i, r in enumerate(["R1", "R2", "R3"]):
            mb.add_agent(
                agt_type_class="ResDam_AgType",
                agt_name=r,
                api="DamAPI",
                priority=0,
                link_dict={f"S{i+1}": -1, r: 1},
                par_dict={},
                attr_dict={
                    "release": releases[r],
                }
                )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Kittitas",
            api="RiverDivAPI",
            priority=1,
            link_dict={"C1": ["DivFactor", 0, "Minus"]}, #-0.5736
            dm_class=None,#"DivDM",
            par_dict={
                "DivFactor": [-99]
                },
            attr_dict={
                "diversion": diversions["Kittitas"]
                }
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Tieton",
            api="RiverDivAPI",
            priority=1,
            link_dict={"C2": -1, "G": ["ReturnFactor", 0, "Plus"]},
            dm_class=None,#"DivDM",
            par_dict={
                "ReturnFactor": [-99]
                },
            attr_dict={
                "diversion": diversions["Tieton"]
                }
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Roza",
            api="RiverDivAPI",
            priority=1,
            link_dict={"G": -1},
            dm_class=None,#"DivDM",
            par_dict={},
            attr_dict={
                "diversion": diversions["Roza"]
                }
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Wapato",
            api="RiverDivAPI",
            priority=1,
            link_dict={"G": -1},
            dm_class=None,#"DivDM",
            par_dict={},
            attr_dict={
                "diversion": diversions["Wapato"]
                }
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Sunnyside",
            api="RiverDivAPI",
            priority=1,
            link_dict={"G": -1},
            dm_class=None,#"DivDM",
            par_dict={},
            attr_dict={
                "diversion": diversions["Sunnyside"]
                }
            )

        mb.print_model()
        mb.write_model_to_yaml(
            str(pn.model.join(f"cali_yrb_CCG_{runoff_model}.yaml"))
        )
        print("Model templates for CCG calibration created.")

    @staticmethod
    def CCG_hydro_only_func(
        pn,
        inputs,
        model_file="cali_yrb_CCG_GWLF.yaml",
        #avg_minor_divs=None,
        minor_divs_M=None,
        disable=True,
        get_model=False,
        get_model_dict=False,
        mode="cali",
        baseline_M=None,
        baseline_Y=None,
        ):
        """Return evaluation function for Borg. nvar=45, nobjs=2, nconstr=0."""

        temp = inputs["temp"]
        prec = inputs["prec"]
        pet = inputs["pet"]
        obv_flow = inputs["obv_flow"]
        obv_div = inputs["obv_div"] # For calculating metrics only

        model_path = pn.model.get(model_file)
        wd = pn.get_str()
        abm_path = pn.src.get_str()

        # Load model
        model_dict_template = hydrocnhs.load_model(model_path)
        # We don't need S discharge to reservoir. We directly assign release amount.
        l = model_dict_template["WaterSystem"]["DataLength"]
        assigned_Q = {sub: [0]*l for sub in ["S1","S2","S3"]}

        model_dict_template["Path"]["WD"] = wd
        model_dict_template["Path"]["Modules"] = abm_path

        def return_model_dict(*params):
            #assert len(params) ==
            model_dict = deepcopy(model_dict_template)

            outlets = ["C1", "C2", "G"]

            # Total 27 rainfall runoff parameters
            for i, outlet in enumerate(outlets):
                model_dict["RainfallRunoff"][outlet]["Pars"]["CN2"] = params[0+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["IS"] = params[1+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Res"] = params[2+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Sep"] = params[3+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Alpha"] = params[4+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Beta"] = params[5+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Ur"] = params[6+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Df"] = params[7+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Kc"] = params[8+9*i]

            # Total 16 routing parameters
            model_dict["Routing"]["G"]["C1"]["Pars"]["Velo"] = params[27]
            model_dict["Routing"]["G"]["C1"]["Pars"]["Diff"] = params[28]
            model_dict["Routing"]["G"]["C2"]["Pars"]["Velo"] = params[29]
            model_dict["Routing"]["G"]["C2"]["Pars"]["Diff"] = params[30]
            model_dict["Routing"]["G"]["G"]["Pars"]["GShape"] = params[31]
            model_dict["Routing"]["G"]["G"]["Pars"]["GScale"] = params[32]

            model_dict["Routing"]["C1"]["R1"]["Pars"]["Velo"] = params[33]
            model_dict["Routing"]["C1"]["R1"]["Pars"]["Diff"] = params[34]
            model_dict["Routing"]["C1"]["C1"]["Pars"]["GShape"] = params[35]
            model_dict["Routing"]["C1"]["C1"]["Pars"]["GScale"] = params[36]

            model_dict["Routing"]["C2"]["R2"]["Pars"]["Velo"] = params[37]
            model_dict["Routing"]["C2"]["R2"]["Pars"]["Diff"] = params[38]
            model_dict["Routing"]["C2"]["R3"]["Pars"]["Velo"] = params[39]
            model_dict["Routing"]["C2"]["R3"]["Pars"]["Diff"] = params[40]
            model_dict["Routing"]["C2"]["C2"]["Pars"]["GShape"] = params[41]
            model_dict["Routing"]["C2"]["C2"]["Pars"]["GScale"] = params[42]

            # Total 2 return flow factors
            model_dict["ABM"]["IrrDiv_AgType"]["Kittitas"]["Pars"]["DivFactor"] = [params[43]]
            model_dict["ABM"]["IrrDiv_AgType"]["Tieton"]["Pars"]["ReturnFactor"] = [params[44]]
            return model_dict

        def return_model(*params):
            model_dict = return_model_dict(*params)
            model = hydrocnhs.Model(model_dict, "cali_CCG_hydro_only")
            model.paral_setting = {'verbose': 0, 'cores_pet': 1, 'cores_formUH': 1, 'cores_runoff': 1}
            return model

        def return_metrics(model, start_date, end_date, obv_flow, obv_div, avg_minor_divs=None, minor_divs_M=None, baseline_M=None, baseline_Y=None):
            Q = model.dc.get_field("Q_routed")
            sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
            sim_Q_M = sim_Q_D.resample("MS").mean()
            pd_date_M_index = sim_Q_M.index

            sim_Q_M = sim_Q_M[start_date:end_date]
            obv_flow = obv_flow[start_date:end_date]

            # Create a DataFrame from avg_minor_divs with the same index as sim_Q_M
            if avg_minor_divs is not None:
                avg_minor_divs_df = pd.DataFrame(index=sim_Q_M.index)
                avg_minor_divs_df['G'] = [avg_minor_divs[month - 1] for month in sim_Q_M.index.month]

                # Subtract the corresponding values from avg_minor_divs for each month
                sim_Q_M["G"] = sim_Q_M["G"].subtract(avg_minor_divs_df['G'], axis=0)
            if minor_divs_M is not None:
                sim_Q_M["G"] -= minor_divs_M["G"]

            # Calculate objective values
            indicator = hydrocnhs.Indicator()
            kge_c1 = indicator.get_kge(x_obv=obv_flow["C1"], y_sim=sim_Q_M["C1"], r_na=True)
            kge_c2 = indicator.get_kge(x_obv=obv_flow["C2"], y_sim=sim_Q_M["C2"], r_na=True)
            kge_g = indicator.get_kge(x_obv=obv_flow["G"], y_sim=sim_Q_M["G"], r_na=True)

            mask = [True if i.month in [7,8,9] else False for i in sim_Q_M.index]
            Q789 = sim_Q_M[mask].resample("YS").mean()["G"]
            kge_g_789 = indicator.get_kge(x_obv=obv_flow[mask].resample("YS").mean()["G"], y_sim=Q789, r_na=True)

            #minor_divs = sum(avg_minor_divs)/235 # Normalize the sum of avg_minor_divs using the value in Lin & Yang (2022)

            ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
            shortage_M = pd.DataFrame(index=pd_date_M_index)
            div_D = pd.DataFrame(index=model.pd_date_index)
            for ag in ag_list:
                shortage_M[ag] = model.dc.get_field(ag)["Shortage_M"]
                div_D[ag] = model.dc.get_field(ag)["Div"]
            shortage_M = shortage_M[start_date:end_date]
            mean_Y_shortage = shortage_M.groupby(shortage_M.index.year).mean().mean().sum()

            obv_div = obv_div[start_date:end_date]
            div_Y = div_D.resample("YS").mean()[start_date:end_date]
            kge_Kittitas = indicator.get_kge(x_obv=obv_div["Kittitas"], y_sim=div_Y["Kittitas"], r_na=True)
            kge_Tieton = indicator.get_kge(x_obv=obv_div["Tieton"], y_sim=div_Y["Tieton"], r_na=True)
            kge_Roza = indicator.get_kge(x_obv=obv_div["Roza"], y_sim=div_Y["Roza"], r_na=True)
            kge_Wapato = indicator.get_kge(x_obv=obv_div["Wapato"], y_sim=div_Y["Wapato"], r_na=True)
            kge_Sunnyside = indicator.get_kge(x_obv=obv_div["Sunnyside"], y_sim=div_Y["Sunnyside"], r_na=True)

            # Compare with baseline
            b_kge_g = b_kge_c1 = b_kge_c2 = b_kge_g_789 = None
            b_kge_Kittitas = b_kge_Tieton = b_kge_Roza = b_kge_Wapato = b_kge_Sunnyside = None
            if baseline_M is not None:
                b_kge_g = indicator.get_kge(x_obv=baseline_M["G"], y_sim=sim_Q_M["G"], r_na=True)
                b_kge_c1 = indicator.get_kge(x_obv=baseline_M["C1"], y_sim=sim_Q_M["C1"], r_na=True)
                b_kge_c2 = indicator.get_kge(x_obv=baseline_M["C2"], y_sim=sim_Q_M["C2"], r_na=True)
            if baseline_Y is not None:
                b_kge_g_789 = indicator.get_kge(x_obv=baseline_Y["Q789"], y_sim=Q789, r_na=True)
                b_kge_Kittitas = indicator.get_kge(x_obv=baseline_Y["Kittitas"], y_sim=div_Y["Kittitas"], r_na=True)
                b_kge_Tieton = indicator.get_kge(x_obv=baseline_Y["Tieton"], y_sim=div_Y["Tieton"], r_na=True)
                b_kge_Roza = indicator.get_kge(x_obv=baseline_Y["Roza"], y_sim=div_Y["Roza"], r_na=True)
                b_kge_Wapato = indicator.get_kge(x_obv=baseline_Y["Wapato"], y_sim=div_Y["Wapato"], r_na=True)
                b_kge_Sunnyside = indicator.get_kge(x_obv=baseline_Y["Sunnyside"], y_sim=div_Y["Sunnyside"], r_na=True)


            #def nrmse(y_sim, x_obv):
            #    """Normalized root mean squared error."""
            #    y_sim = np.array(y_sim)
            #    x_obv = np.array(x_obv)
            #    return np.sqrt(np.mean((y_sim - x_obv) ** 2)) / (np.max(x_obv) - np.min(x_obv))

            #nrmse_Kittitas = nrmse(x_obv=obv_div["Kittitas"], y_sim=div_Y["Kittitas"])
            #nrmse_Tieton = nrmse(x_obv=obv_div["Tieton"], y_sim=div_Y["Tieton"])
            #nrmse_Roza = nrmse(x_obv=obv_div["Roza"], y_sim=div_Y["Roza"])
            #nrmse_Wapato = nrmse(x_obv=obv_div["Wapato"], y_sim=div_Y["Wapato"])
            #nrmse_Sunnyside = nrmse(x_obv=obv_div["Sunnyside"], y_sim=div_Y["Sunnyside"])
            #nrmse_g_789 = nrmse(x_obv=obv_flow.loc[mask, :]["G"], y_sim=sim_Q_M.loc[mask, :]["G"])


            met_dict = {
                "KGE_C1": kge_c1, "KGE_C2": kge_c2, "KGE_G": kge_g, "KGE_G(789)": kge_g_789, "avg_div_deficit": mean_Y_shortage,
                "KGE_Div": np.mean([kge_Kittitas, kge_Tieton, kge_Roza, kge_Wapato, kge_Sunnyside]),
                'KGE_Kittitas': kge_Kittitas, 'KGE_Tieton': kge_Tieton, 'KGE_Roza': kge_Roza, 'KGE_Wapato': kge_Wapato, 'KGE_Sunnyside': kge_Sunnyside,
                "b_KGE_C1": b_kge_c1, "b_KGE_C2": b_kge_c2, "b_KGE_G": b_kge_g, "b_KGE_G(789)": b_kge_g_789,
                "b_KGE_Div": np.mean([b_kge_Kittitas, b_kge_Tieton, b_kge_Roza, b_kge_Wapato, b_kge_Sunnyside]),
                "b_KGE_Kittitas": b_kge_Kittitas, "b_KGE_Tieton": b_kge_Tieton, "b_KGE_Roza": b_kge_Roza, "b_KGE_Wapato": b_kge_Wapato, "b_KGE_Sunnyside": b_kge_Sunnyside,
                #'nrmse_Kittitas': nrmse_Kittitas, 'nrmse_Tieton': nrmse_Tieton, 'nrmse_Roza': nrmse_Roza, 'nrmse_Wapato': nrmse_Wapato, 'nrmse_Sunnyside': nrmse_Sunnyside,
                #'nrmse_g_789': nrmse_g_789
                }
            return met_dict

        def return_ts(model, start_date, end_date, avg_minor_divs=None, minor_divs_M=None):
            Q = model.dc.get_field("Q_routed")
            sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
            sim_Q_M = sim_Q_D.resample("MS").mean()
            pd_date_M_index = sim_Q_M.index

            sim_Q_M = sim_Q_M[start_date:end_date]

            # Create a DataFrame from avg_minor_divs with the same index as sim_Q_M
            if avg_minor_divs is not None:
                avg_minor_divs_df = pd.DataFrame(index=sim_Q_M.index)
                avg_minor_divs_df['G'] = [avg_minor_divs[month - 1] for month in sim_Q_M.index.month]

                # Subtract the corresponding values from avg_minor_divs for each month
                sim_Q_M["G"] = sim_Q_M["G"].subtract(avg_minor_divs_df['G'], axis=0)
            if minor_divs_M is not None:
                sim_Q_M["G"] -= minor_divs_M["G"]
            agt_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
            shortage_M = pd.DataFrame(index=pd_date_M_index)
            div_D = pd.DataFrame(index=model.pd_date_index)
            for ag in agt_list:
                shortage_M[ag] = model.dc.get_field(ag)["Shortage_M"]
                div_D[ag] = model.dc.get_field(ag)["Div"]

            shortage_M = shortage_M[start_date:end_date]
            div_M = div_D.resample("MS").mean()[start_date:end_date]
            shortage_M.columns = [f"{ag}_shortage" for ag in shortage_M.columns]
            df_M = pd.concat([sim_Q_M, shortage_M, div_M], axis=1)
            df_M = df_M[[
                'C1', 'C2', 'G', 'Kittitas_shortage', 'Tieton_shortage', 'Roza_shortage',
                'Wapato_shortage', 'Sunnyside_shortage', 'Kittitas', 'Tieton', 'Roza',
                'Wapato', 'Sunnyside'
                ]]
            df_M.index.name = 'date'
            return df_M

        def run_yrb_CCG_hydro_only_for_cali(*params):
            avg_minor_divs = [0]*12
            avg_minor_divs[7-1] = params[45]
            avg_minor_divs[8-1] = params[46]
            avg_minor_divs[9-1] = params[47]
            model = return_model(*params)
            try:
                _ = model.run(temp, prec, pet, assigned_Q, disable=disable)

                # Calibration period with one year warm-up
                start_date = '1980-01-01'
                end_date = '2023-12-31'

                # We set minor_divs to none but in the future we can iterate on this and recalibrate with minor_divs
                met_dict = return_metrics(model, start_date, end_date, obv_flow, obv_div, avg_minor_divs, minor_divs_M=None, baseline_M=baseline_M, baseline_Y=baseline_Y)
                kge_c1 = met_dict["KGE_C1"]
                kge_c2 = met_dict["KGE_C2"]
                kge_g = met_dict["KGE_G"]
                mean_Y_shortage = met_dict["avg_div_deficit"]
                nrmse_g_789 = met_dict["nrmse_g_789"]

                objs = [sum([-kge_c1, -kge_c2, -kge_g])/3, nrmse_g_789, -(1 - mean_Y_shortage)]
            except Exception as e:
                print(e)
                objs = [999] * 3 #nobjs
                #constrs = [0]
            return (objs, )#constrs

        def run_yrb_CCG_hydro_only_for_sa(*params):
            try:
                model = return_model(*params)
                _ = model.run(temp, prec, pet, assigned_Q, disable=disable)

                # Calibration period with one year warm-up
                start_date = '1980-01-01'
                end_date = '2023-12-31'

                # Use the identified monthly minor_divs_M
                met_dict = return_metrics(model, start_date, end_date, obv_flow, obv_div, avg_minor_divs=None,
                                          minor_divs_M=minor_divs_M, baseline_M=baseline_M, baseline_Y=baseline_Y)
                df_M = return_ts(model, start_date, end_date, avg_minor_divs=None, minor_divs_M=minor_divs_M)
            except Exception as e:
                print(e)
                met_dict, df_M = None, None

            return met_dict, df_M

        def return_model_yrb_CCG_hydro_only(*params):
            model = return_model(*params)
            return model, (temp, prec, pet, obv_flow, minor_divs_M, assigned_Q)

        if get_model:
            # return model, (temp, prec, pet, obv_flow, minor_divs_M, assigned_Q)
            return return_model_yrb_CCG_hydro_only
        elif get_model_dict:
            # return model_dict
            return return_model_dict
        elif mode == "cali":
            #return (objs, )
            return run_yrb_CCG_hydro_only_for_cali
        elif mode == "sa":
            # return met_dict, df_M  or  None, None if error occurs
            return run_yrb_CCG_hydro_only_for_sa
        else:
            raise ValueError("mode should be either 'cali' or 'sa'.")

    def make_model_template_coupledABM(
        self, start="1979/1/1", end="2023/12/31",
        inputs_yrb_coupledABM="inputs_yrb_coupledABM_1979_2023_abm.json",
        runoff_model="GWLF"
        ):
        """Create a model template for CCG hydrological model calibration."""
        pn = self.pn
        ib = InputBuilder(pn)
        inputs = ib.load_json(filename=pn.inputs.get(inputs_yrb_coupledABM))

        releases = inputs["releases"]
        pr_nov_to_jun_sum = inputs["pr_nov_to_jun_sum"]
        init_div_ref = inputs["init_div_ref"]
        div_Y_max = inputs["div_Y_max"]
        div_Y_min = inputs["div_Y_min"]
        corr = inputs["corr"]
        ccurves = inputs["ccurves"]
        diversions = inputs["diversions"] # for initial diversion
        minor_divs = inputs["minor_divs"] # for minor diversion

        mb = hydrocnhs.ModelBuilder(wd="")
        mb.set_water_system(start_date=start, end_date=end)
        mb.set_rainfall_runoff(
            outlet_list=["S1", "S2", "S3"],
            area_list=[83014.25, 11601.47, 28016.2],
            lat_list=[47.416, 46.814, 46.622],
            runoff_model=runoff_model,
            activate=False
            )
        mb.set_rainfall_runoff(
            outlet_list=["C1", "C2", "G"],
            area_list=[328818.7, 203799.79, 291203.8],
            lat_list=[47.145, 46.839, 46.682],
            runoff_model=runoff_model
            )
        mb.set_routing_outlet(
            routing_outlet="G",
            upstream_outlet_list=["C1", "C2"],
            flow_length_list=[59404.82, 48847.74]
            )
        mb.set_routing_outlet(
            routing_outlet="C1",
            upstream_outlet_list=["R1"],
            flow_length_list=[100364.29],
            instream_objects=["R1"]
            )
        mb.set_routing_outlet(
            routing_outlet="C2",
            upstream_outlet_list=["R2", "R3"],
            flow_length_list=[70713.85, 36293.57],
            instream_objects=["R2", "R3"]
            )
        for s in ["S1", "S2", "S3"]:
            mb.set_routing_outlet(routing_outlet=s, upstream_outlet_list=[], activate=False)

        mb.set_ABM(
            abm_module_folder_path=pn.src.get_str(),
            abm_module_name="abm_yrb_with_res_as_inputs_for_coupledABM_cali.py"
            )
        mb.model["WaterSystem"]["ABM"]["FlowTargetCoeff"] = (0.0992, 10) # use values <= Q50 from 1985-2024 to fit the linear regression
        mb.model["WaterSystem"]["ABM"]["pr_nov_to_jun_sum"] = pr_nov_to_jun_sum
        mb.model["WaterSystem"]["ABM"]["Corr"] = corr # correlation matrix follow agt_list order
        mb.model["WaterSystem"]["ABM"]["minor_divs"] = minor_divs # minor diversion for G
        # Add additional info
        #mb.model["WaterSystem"]["ABM"]["div_dict"] = "should assigned this in the simulation setup from inputs"

        for i, r in enumerate(["R1", "R2", "R3"]):
            mb.add_agent(
                agt_type_class="ResDam_AgType",
                agt_name=r,
                api="DamAPI",
                priority=0,
                link_dict={f"S{i+1}": -1, r: 1},
                par_dict={},
                attr_dict={
                    "release": releases[r],
                }
                )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Kittitas",
            api="RiverDivAPI",
            priority=1,
            link_dict={"C1": ["DivFactor", 0, "Minus"]}, #-0.5736
            dm_class="DivDM",
            par_dict={
                "DivFactor": [-99],
                "L_U": -99,
                "L_L": -99,
                "Lr_c": -99,
                "a": -99,
                "b": -99,
                "Sig": -99,
                "ProratedRatio": -99,
                },
            attr_dict={
                "init_div": diversions["Kittitas"][0:31+28],
                "init_div_ref": init_div_ref["Kittitas"],
                "div_Y_max": div_Y_max["Kittitas"],
                "div_Y_min": div_Y_min["Kittitas"],
                "ccurve": ccurves["Kittitas"],
                },
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Tieton",
            api="RiverDivAPI",
            priority=1,
            link_dict={"C2": -1, "G": ["ReturnFactor", 0, "Plus"]},
            dm_class="DivDM",
            par_dict={
                "ReturnFactor": [-99],
                "L_U": -99,
                "L_L": -99,
                "Lr_c": -99,
                "a": -99,
                "b": -99,
                "Sig": -99,
                "ProratedRatio": -99,
                },
            attr_dict={
                "init_div": diversions["Tieton"][0:31+28],
                "init_div_ref": init_div_ref["Tieton"],
                "div_Y_max": div_Y_max["Tieton"],
                "div_Y_min": div_Y_min["Tieton"],
                "ccurve": ccurves["Tieton"],
                },
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Roza",
            api="RiverDivAPI",
            priority=1,
            link_dict={"G": -1},
            dm_class="DivDM",
            par_dict={
                "L_U": -99,
                "L_L": -99,
                "Lr_c": -99,
                "a": -99,
                "b": -99,
                "Sig": -99,
                "ProratedRatio": -99,
                },
            attr_dict={
                "init_div": diversions["Roza"][0:31+28],
                "init_div_ref": init_div_ref["Roza"],
                "div_Y_max": div_Y_max["Roza"],
                "div_Y_min": div_Y_min["Roza"],
                "ccurve": ccurves["Roza"],
                },
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Wapato",
            api="RiverDivAPI",
            priority=1,
            link_dict={"G": -1},
            dm_class="DivDM",
            par_dict={
                "L_U": -99,
                "L_L": -99,
                "Lr_c": -99,
                "a": -99,
                "b": -99,
                "Sig": -99,
                "ProratedRatio": -99,
                },
            attr_dict={
                "init_div": diversions["Wapato"][0:31+28],
                "init_div_ref": init_div_ref["Wapato"],
                "div_Y_max": div_Y_max["Wapato"],
                "div_Y_min": div_Y_min["Wapato"],
                "ccurve": ccurves["Wapato"],
                },
            )
        mb.add_agent(
            agt_type_class="IrrDiv_AgType",
            agt_name="Sunnyside",
            api="RiverDivAPI",
            priority=1,
            link_dict={"G": -1},
            dm_class="DivDM",
            par_dict={
                "L_U": -99,
                "L_L": -99,
                "Lr_c": -99,
                "a": -99,
                "b": -99,
                "Sig": -99,
                "ProratedRatio": -99,
                },
            attr_dict={
                "init_div": diversions["Sunnyside"][0:31+28],
                "init_div_ref": init_div_ref["Sunnyside"],
                "div_Y_max": div_Y_max["Sunnyside"],
                "div_Y_min": div_Y_min["Sunnyside"],
                "ccurve": ccurves["Sunnyside"],
                },
            )

        mb.add_institution(
            institution="agt_div_dm",
            instit_dm_class="DivDM",
            agent_list=["Kittitas", "Tieton", "Roza", "Wapato", "Sunnyside"]
        )

        mb.print_model()
        mb.write_model_to_yaml(
            str(pn.model.join(f"cali_yrb_coupledABM_{runoff_model}.yaml"))
        )
        print("Model templates for coupledABM created.")

    @staticmethod
    def coupledABM_func(
        pn,
        inputs,
        model_file="cali_yrb_coupledABM_GWLF.yaml",
        avg_minor_divs=None,
        disable=True,
        get_model=False,
        get_model_dict=False,
        mode="cali",
        seeds=[None],
        prefix_params=[],
        baseline_M=None,
        baseline_Y=None
        ):
        """Return evaluation function for Borg. nvar=, nobjs=2, nconstr=0."""

        if len(prefix_params) != 0:
            assert len(prefix_params) == 45, "prefix_params should be 45 parameters."

        temp = inputs["temp"]
        prec = inputs["prec"]
        pet = inputs["pet"]
        obv_flow = inputs["obv_flow"]
        obv_div = inputs["obv_div"]

        model_path = pn.model.get(model_file)
        wd = pn.get_str()
        abm_path = pn.src.get_str()

        # Load model
        model_dict_template = hydrocnhs.load_model(model_path)
        # We don't need S discharge to reservoir. We directly assign release amount.
        l = model_dict_template["WaterSystem"]["DataLength"]
        assigned_Q = {sub: [0]*l for sub in ["S1","S2","S3"]}

        model_dict_template["Path"]["WD"] = wd
        model_dict_template["Path"]["Modules"] = abm_path

        def return_model_dict_linear(*params):
            if len(prefix_params) != 0:
                assert len(params) == 35, "Params should be 35 parameters." # for exp2
            params = list(prefix_params) + list(params)

            model_dict = deepcopy(model_dict_template)
            outlets = ["C1", "C2", "G"]
            # Total 27 rainfall runoff parameters
            for i, outlet in enumerate(outlets):
                model_dict["RainfallRunoff"][outlet]["Pars"]["CN2"] = params[0+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["IS"] = params[1+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Res"] = params[2+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Sep"] = params[3+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Alpha"] = params[4+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Beta"] = params[5+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Ur"] = params[6+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Df"] = params[7+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Kc"] = params[8+9*i]

            # Total 16 routing parameters
            model_dict["Routing"]["G"]["C1"]["Pars"]["Velo"] = params[27]
            model_dict["Routing"]["G"]["C1"]["Pars"]["Diff"] = params[28]
            model_dict["Routing"]["G"]["C2"]["Pars"]["Velo"] = params[29]
            model_dict["Routing"]["G"]["C2"]["Pars"]["Diff"] = params[30]
            model_dict["Routing"]["G"]["G"]["Pars"]["GShape"] = params[31]
            model_dict["Routing"]["G"]["G"]["Pars"]["GScale"] = params[32]

            model_dict["Routing"]["C1"]["R1"]["Pars"]["Velo"] = params[33]
            model_dict["Routing"]["C1"]["R1"]["Pars"]["Diff"] = params[34]
            model_dict["Routing"]["C1"]["C1"]["Pars"]["GShape"] = params[35]
            model_dict["Routing"]["C1"]["C1"]["Pars"]["GScale"] = params[36]

            model_dict["Routing"]["C2"]["R2"]["Pars"]["Velo"] = params[37]
            model_dict["Routing"]["C2"]["R2"]["Pars"]["Diff"] = params[38]
            model_dict["Routing"]["C2"]["R3"]["Pars"]["Velo"] = params[39]
            model_dict["Routing"]["C2"]["R3"]["Pars"]["Diff"] = params[40]
            model_dict["Routing"]["C2"]["C2"]["Pars"]["GShape"] = params[41]
            model_dict["Routing"]["C2"]["C2"]["Pars"]["GScale"] = params[42]

            # Total 2 return flow factors
            model_dict["ABM"]["IrrDiv_AgType"]["Kittitas"]["Pars"]["DivFactor"] = [params[43]]
            model_dict["ABM"]["IrrDiv_AgType"]["Tieton"]["Pars"]["ReturnFactor"] = [params[44]]

            agt_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
            l = 7 # number of parameters for each agent type
            for i, ag in enumerate(agt_list):
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["L_U"] = params[45+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["L_L"] = params[46+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["Lr_c"] = params[47+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["Sig"] = params[48+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["a"] = params[49+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["b"] = params[50+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["ProratedRatio"] = params[51+i*l]
            return model_dict

        def return_model_dict(*params):
            if len(prefix_params) != 0:
                assert len(params) == 40, "Params should be 40 parameters." # for exp2
            params = list(prefix_params) + list(params)

            model_dict = deepcopy(model_dict_template)
            outlets = ["C1", "C2", "G"]
            # Total 27 rainfall runoff parameters
            for i, outlet in enumerate(outlets):
                model_dict["RainfallRunoff"][outlet]["Pars"]["CN2"] = params[0+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["IS"] = params[1+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Res"] = params[2+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Sep"] = params[3+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Alpha"] = params[4+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Beta"] = params[5+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Ur"] = params[6+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Df"] = params[7+9*i]
                model_dict["RainfallRunoff"][outlet]["Pars"]["Kc"] = params[8+9*i]

            # Total 16 routing parameters
            model_dict["Routing"]["G"]["C1"]["Pars"]["Velo"] = params[27]
            model_dict["Routing"]["G"]["C1"]["Pars"]["Diff"] = params[28]
            model_dict["Routing"]["G"]["C2"]["Pars"]["Velo"] = params[29]
            model_dict["Routing"]["G"]["C2"]["Pars"]["Diff"] = params[30]
            model_dict["Routing"]["G"]["G"]["Pars"]["GShape"] = params[31]
            model_dict["Routing"]["G"]["G"]["Pars"]["GScale"] = params[32]

            model_dict["Routing"]["C1"]["R1"]["Pars"]["Velo"] = params[33]
            model_dict["Routing"]["C1"]["R1"]["Pars"]["Diff"] = params[34]
            model_dict["Routing"]["C1"]["C1"]["Pars"]["GShape"] = params[35]
            model_dict["Routing"]["C1"]["C1"]["Pars"]["GScale"] = params[36]

            model_dict["Routing"]["C2"]["R2"]["Pars"]["Velo"] = params[37]
            model_dict["Routing"]["C2"]["R2"]["Pars"]["Diff"] = params[38]
            model_dict["Routing"]["C2"]["R3"]["Pars"]["Velo"] = params[39]
            model_dict["Routing"]["C2"]["R3"]["Pars"]["Diff"] = params[40]
            model_dict["Routing"]["C2"]["C2"]["Pars"]["GShape"] = params[41]
            model_dict["Routing"]["C2"]["C2"]["Pars"]["GScale"] = params[42]

            # Total 2 return flow factors
            model_dict["ABM"]["IrrDiv_AgType"]["Kittitas"]["Pars"]["DivFactor"] = [params[43]]
            model_dict["ABM"]["IrrDiv_AgType"]["Tieton"]["Pars"]["ReturnFactor"] = [params[44]]

            agt_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
            l = 8 # number of parameters for each agent type
            for i, ag in enumerate(agt_list):
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["L_U"] = params[45+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["L_L"] = params[46+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["Lr_c"] = params[47+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["Sig"] = params[48+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["a"] = params[49+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["b"] = params[50+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["c"] = params[51+i*l]
                model_dict["ABM"]["IrrDiv_AgType"][ag]["Pars"]["ProratedRatio"] = params[52+i*l]
            return model_dict

        def return_model(*params, seed=None):
            model_dict = return_model_dict(*params)
            if seed is not None:
                rn_gen = hydrocnhs.create_rn_gen(seed)
                model = hydrocnhs.Model(model_dict, "cali_coupledABM", rn_gen=rn_gen)
            else:
                model = hydrocnhs.Model(model_dict, "cali_coupledABM")
            model.paral_setting = {'verbose': 0, 'cores_pet': 1, 'cores_formUH': 1, 'cores_runoff': 1}
            return model

        def return_metrics(model, start_date, end_date, obv_flow, obv_div, avg_minor_divs=None, baseline_M=None, baseline_Y=None):
            Q = model.dc.get_field("Q_routed")
            sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
            sim_Q_M = sim_Q_D.resample("MS").mean()
            pd_date_M_index = sim_Q_M.index

            sim_Q_M = sim_Q_M[start_date:end_date]
            obv_flow = obv_flow[start_date:end_date]

            # Create a DataFrame from avg_minor_divs with the same index as sim_Q_M
            if avg_minor_divs is not None:
                avg_minor_divs_df = pd.DataFrame(index=sim_Q_M.index)
                avg_minor_divs_df['G'] = [avg_minor_divs[month - 1] for month in sim_Q_M.index.month]

                # Subtract the corresponding values from avg_minor_divs for each month
                sim_Q_M["G"] = sim_Q_M["G"].subtract(avg_minor_divs_df['G'], axis=0)

            # Calculate objective values
            indicator = hydrocnhs.Indicator()
            kge_c1 = indicator.get_kge(x_obv=obv_flow["C1"], y_sim=sim_Q_M["C1"], r_na=True)
            kge_c2 = indicator.get_kge(x_obv=obv_flow["C2"], y_sim=sim_Q_M["C2"], r_na=True)
            kge_g = indicator.get_kge(x_obv=obv_flow["G"], y_sim=sim_Q_M["G"], r_na=True)

            mask = [True if i.month in [7,8,9] else False for i in sim_Q_M.index]
            Q789 = sim_Q_M[mask].resample("YS").mean()["G"]
            kge_g_789 = indicator.get_kge(x_obv=obv_flow[mask].resample("YS").mean()["G"], y_sim=Q789, r_na=True)

            #minor_divs = sum(avg_minor_divs)/235 # Normalize the sum of avg_minor_divs using the value in Lin & Yang (2022)

            ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
            shortage_M = pd.DataFrame(index=pd_date_M_index)
            div_D = pd.DataFrame(index=model.pd_date_index)
            for ag in ag_list:
                shortage_M[ag] = model.dc.get_field(ag)["Shortage_M"]
                div_D[ag] = model.dc.get_field(ag)["Div"]
            shortage_M = shortage_M[start_date:end_date]
            mean_Y_shortage = shortage_M.groupby(shortage_M.index.year).mean().mean().sum()

            obv_div = obv_div[start_date:end_date]
            div_Y = div_D.resample("YS").mean()[start_date:end_date]
            kge_Kittitas = indicator.get_kge(x_obv=obv_div["Kittitas"], y_sim=div_Y["Kittitas"], r_na=True)
            kge_Tieton = indicator.get_kge(x_obv=obv_div["Tieton"], y_sim=div_Y["Tieton"], r_na=True)
            kge_Roza = indicator.get_kge(x_obv=obv_div["Roza"], y_sim=div_Y["Roza"], r_na=True)
            kge_Wapato = indicator.get_kge(x_obv=obv_div["Wapato"], y_sim=div_Y["Wapato"], r_na=True)
            kge_Sunnyside = indicator.get_kge(x_obv=obv_div["Sunnyside"], y_sim=div_Y["Sunnyside"], r_na=True)

            # Compare with baseline
            b_kge_g = b_kge_c1 = b_kge_c2 = b_kge_g_789 = None
            b_kge_Kittitas = b_kge_Tieton = b_kge_Roza = b_kge_Wapato = b_kge_Sunnyside = None
            if baseline_M is not None:
                b_kge_g = indicator.get_kge(x_obv=baseline_M["G"], y_sim=sim_Q_M["G"], r_na=True)
                b_kge_c1 = indicator.get_kge(x_obv=baseline_M["C1"], y_sim=sim_Q_M["C1"], r_na=True)
                b_kge_c2 = indicator.get_kge(x_obv=baseline_M["C2"], y_sim=sim_Q_M["C2"], r_na=True)
            if baseline_Y is not None:
                b_kge_g_789 = indicator.get_kge(x_obv=baseline_Y["Q789"], y_sim=Q789, r_na=True)
                b_kge_Kittitas = indicator.get_kge(x_obv=baseline_Y["Kittitas"], y_sim=div_Y["Kittitas"], r_na=True)
                b_kge_Tieton = indicator.get_kge(x_obv=baseline_Y["Tieton"], y_sim=div_Y["Tieton"], r_na=True)
                b_kge_Roza = indicator.get_kge(x_obv=baseline_Y["Roza"], y_sim=div_Y["Roza"], r_na=True)
                b_kge_Wapato = indicator.get_kge(x_obv=baseline_Y["Wapato"], y_sim=div_Y["Wapato"], r_na=True)
                b_kge_Sunnyside = indicator.get_kge(x_obv=baseline_Y["Sunnyside"], y_sim=div_Y["Sunnyside"], r_na=True)


            #def nrmse(y_sim, x_obv):
            #    """Normalized root mean squared error."""
            #    y_sim = np.array(y_sim)
            #    x_obv = np.array(x_obv)
            #    return np.sqrt(np.mean((y_sim - x_obv) ** 2)) / (np.max(x_obv) - np.min(x_obv))

            #nrmse_Kittitas = nrmse(x_obv=obv_div["Kittitas"], y_sim=div_Y["Kittitas"])
            #nrmse_Tieton = nrmse(x_obv=obv_div["Tieton"], y_sim=div_Y["Tieton"])
            #nrmse_Roza = nrmse(x_obv=obv_div["Roza"], y_sim=div_Y["Roza"])
            #nrmse_Wapato = nrmse(x_obv=obv_div["Wapato"], y_sim=div_Y["Wapato"])
            #nrmse_Sunnyside = nrmse(x_obv=obv_div["Sunnyside"], y_sim=div_Y["Sunnyside"])
            #nrmse_g_789 = nrmse(x_obv=obv_flow.loc[mask, :]["G"], y_sim=sim_Q_M.loc[mask, :]["G"])


            met_dict = {
                "KGE_C1": kge_c1, "KGE_C2": kge_c2, "KGE_G": kge_g, "KGE_G(789)": kge_g_789, "avg_div_deficit": mean_Y_shortage,
                "KGE_Div": np.mean([kge_Kittitas, kge_Tieton, kge_Roza, kge_Wapato, kge_Sunnyside]),
                'KGE_Kittitas': kge_Kittitas, 'KGE_Tieton': kge_Tieton, 'KGE_Roza': kge_Roza, 'KGE_Wapato': kge_Wapato, 'KGE_Sunnyside': kge_Sunnyside,
                "b_KGE_C1": b_kge_c1, "b_KGE_C2": b_kge_c2, "b_KGE_G": b_kge_g, "b_KGE_G(789)": b_kge_g_789,
                "b_KGE_Div": np.mean([b_kge_Kittitas, b_kge_Tieton, b_kge_Roza, b_kge_Wapato, b_kge_Sunnyside]),
                "b_KGE_Kittitas": b_kge_Kittitas, "b_KGE_Tieton": b_kge_Tieton, "b_KGE_Roza": b_kge_Roza, "b_KGE_Wapato": b_kge_Wapato, "b_KGE_Sunnyside": b_kge_Sunnyside,
                #'nrmse_Kittitas': nrmse_Kittitas, 'nrmse_Tieton': nrmse_Tieton, 'nrmse_Roza': nrmse_Roza, 'nrmse_Wapato': nrmse_Wapato, 'nrmse_Sunnyside': nrmse_Sunnyside,
                #'nrmse_g_789': nrmse_g_789
                }

            return met_dict

        def return_ts(model, start_date, end_date, avg_minor_divs):
            Q = model.dc.get_field("Q_routed")
            sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
            sim_Q_M = sim_Q_D.resample("MS").mean()
            pd_date_M_index = sim_Q_M.index

            sim_Q_M = sim_Q_M[start_date:end_date]

            # Create a DataFrame from avg_minor_divs with the same index as sim_Q_M
            #avg_minor_divs_df = pd.DataFrame(index=sim_Q_M.index)
            #vg_minor_divs_df['avg_minor_divs'] = [avg_minor_divs[month - 1] for month in sim_Q_M.index.month]

            # Subtract the corresponding values from avg_minor_divs for each month
            #sim_Q_M["G"] = sim_Q_M["G"].subtract(avg_minor_divs_df['avg_minor_divs'], axis=0)
            agt_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
            shortage_M = pd.DataFrame(index=pd_date_M_index)
            div_D = pd.DataFrame(index=model.pd_date_index)
            for ag in agt_list:
                shortage_M[ag] = model.dc.get_field(ag)["Shortage_M"]
                div_D[ag] = model.dc.get_field(ag)["Div"]

            shortage_M = shortage_M[start_date:end_date]
            div_M = div_D.resample("MS").mean()[start_date:end_date]
            shortage_M.columns = [f"{ag}_shortage" for ag in shortage_M.columns]
            df_M = pd.concat([sim_Q_M, shortage_M, div_M], axis=1)
            df_M = df_M[[
                'C1', 'C2', 'G', 'Kittitas_shortage', 'Tieton_shortage', 'Roza_shortage',
                'Wapato_shortage', 'Sunnyside_shortage', 'Kittitas', 'Tieton', 'Roza',
                'Wapato', 'Sunnyside'
                ]]
            df_M.index.name = 'date'
            return df_M

        def run_yrb_coupledABM_for_cali(*params):
            try:
                objs_list = []
                for seed in seeds:
                    model = return_model(*params, seed=seed)

                    _ = model.run(temp, prec, pet, assigned_Q, disable=disable)

                    # Calibration period with one year warm-up
                    start_date = '1980-01-01'
                    end_date = '2023-12-31'

                    met_dict = return_metrics(model, start_date, end_date, obv_flow, obv_div, avg_minor_divs, baseline_M=baseline_M, baseline_Y=baseline_Y)
                    kge_c1 = met_dict["KGE_C1"]
                    kge_c2 = met_dict["KGE_C2"]
                    kge_g = met_dict["KGE_G"]
                    kge_g789 = met_dict["KGE_G(789)"]
                    mean_Y_shortage = met_dict["avg_div_deficit"]
                    kge_Kittitas = met_dict["KGE_Kittitas"]
                    kge_Tieton = met_dict["KGE_Tieton"]
                    kge_Roza = met_dict["KGE_Roza"]
                    kge_Wapato = met_dict["KGE_Wapato"]
                    kge_Sunnyside = met_dict["KGE_Sunnyside"]
                    nrmse_Kittitas = met_dict["nrmse_Kittitas"]
                    nrmse_Tieton = met_dict["nrmse_Tieton"]
                    nrmse_Roza = met_dict["nrmse_Roza"]
                    nrmse_Wapato = met_dict["nrmse_Wapato"]
                    nrmse_Sunnyside = met_dict["nrmse_Sunnyside"]
                    nrmse_g_789 = met_dict["nrmse_g_789"]

                    #mean_kge = sum([-kge_c1, -kge_c2, -kge_g, -kge_Kittitas, -kge_Tieton, -kge_Roza, -kge_Wapato, -kge_Sunnyside])/8
                    #mean_kge = sum([-kge_g789, -kge_Kittitas, -kge_Tieton, -kge_Roza, -kge_Wapato, -kge_Sunnyside, -kge_g789])/7
                    #mean_nrmse = sum([nrmse_Kittitas, nrmse_Tieton, nrmse_Roza, nrmse_Wapato, nrmse_Sunnyside, nrmse_g_789])/6
                    #objs = [mean_kge, mean_nrmse, -(1 - mean_Y_shortage)]

                    mean_kge = sum([-kge_g789, -kge_Kittitas, -kge_Tieton, -kge_Roza, -kge_Wapato, -kge_Sunnyside])/6
                    #mean_nrmse = sum([nrmse_Kittitas, nrmse_Tieton, nrmse_Roza, nrmse_Wapato, nrmse_Sunnyside])/5
                    mean_nrmse_sqr = sum([
                        (nrmse_Kittitas+1)**2,
                        (nrmse_Tieton+1)**2,
                        (nrmse_Roza+1)**2,
                        (nrmse_Wapato+1)**2,
                        (nrmse_Sunnyside+1)**2])/5
                    objs = [nrmse_g_789, mean_nrmse_sqr, mean_kge]

                    objs_list.append(objs)
                objs = np.mean(objs_list, axis=0).flatten()

                # Check for NaNs
                if np.isnan(objs).any():
                    raise ValueError("objs_list contains NaN values.")

                # Check for None values
                if np.any([x is None for x in objs]):
                    raise ValueError("objs_list contains None values.")

            except Exception as e:
                print(e)
                objs = [999] * 3 #nobjs
                    #constrs = [0]
            return (objs, )#constrs

        def run_yrb_coupledABM_for_sa(*params):
            try:
                #met_dict_sums = defaultdict(list)
                met_dict_all = {}
                df_M_all = {}
                for seed in seeds:
                    model = return_model(*params, seed=seed)
                    _ = model.run(temp, prec, pet, assigned_Q, disable=disable)

                    # Calibration period with one year warm-up
                    start_date = '1980-01-01'
                    end_date = '2023-12-31'

                    met_dict = return_metrics(model, start_date, end_date, obv_flow, obv_div, avg_minor_divs, baseline_M=baseline_M, baseline_Y=baseline_Y)
                    met_dict_all[seed] = met_dict
                    df_M = return_ts(model, start_date, end_date, avg_minor_divs)
                    df_M_all[seed] = df_M
                    # Collect all values for each key
                    #for d in met_dict:
                    #    for key, value in d.items():
                    #        met_dict_sums[key].append(value)
                # Calculate the mean for each key
                #met_dict = {key: np.mean(values) for key, values in met_dict_sums.items()}
            except Exception as e:
                print(e)
                #met_dict = None
                met_dict_all = {}
                df_M_all = {}
            return met_dict_all, df_M_all

        def return_model_yrb_coupledABM(*params):
            # Take the first seed for the model
            model = return_model(*params, seed=seeds[0])
            return model, (temp, prec, pet, obv_flow, obv_div, assigned_Q)

        if get_model:
            # return model, (temp, prec, pet, obv_flow, obv_div, assigned_Q)
            return return_model_yrb_coupledABM
        elif get_model_dict:
            # return model_dict
            return return_model_dict
        elif mode == "cali":
            #return (objs, )
            return run_yrb_coupledABM_for_cali
        elif mode == "sa":
            # return met_dict_all, df_M_all
            return run_yrb_coupledABM_for_sa
        else:
            raise ValueError("mode should be either 'cali' or 'sa'.")

class DP:
    def __init__(self):
        pass

    @staticmethod
    def cal_UH_IG(shape, scale, show=False):
        T_IG = 12  # [day] Base time for within subbasin UH
        UH_IG = np.zeros(T_IG)
        if scale <= 0.0001:
            scale = 0.0001  # Since we cannot divide zero.
        for i in range(T_IG):
            # x-axis is in hr unit. We calculate in daily time step.
            UH_IG[i] = gamma.cdf(24 * (i + 1), a=shape, loc=0, scale=scale) - gamma.cdf(
                24 * i, a=shape, loc=0, scale=scale
            )

        if show:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(UH_IG, label=f"shape={shape}, scale={scale}")
            ax.set_xlabel("Days")
            ax.set_ylabel("ratio")
            ax.legend()
            plt.show()
        return UH_IG

    @staticmethod
    def cal_UH_RR(Velo, Diff, flow_len, show=False):
        # River routing
        T_RR = 96  # [day] Base time for river routing UH
        dT_sec = 3600  # [sec] Time step in second for solving
        ##Saint-Venant equation. This will affect Tmax_hr.
        Tmax_hr = T_RR * 24  # [hr] Base time of river routing UH in hour
        ##because dT_sec is for an hour
        Tgr_hr = 48 * 50  # [hr] Base time for Green function values

        # ----- Derive Daily River Impulse Response Function (Green's function) ----
        UH_RR = np.zeros(T_RR)
        if flow_len == 0:
            # No time delay for river routing when the outlet is gauged outlet.
            UH_RR[0] = 1
        else:
            # Calculate h(x, t)
            t = 0
            UH_RRm = np.zeros(Tgr_hr)  # h() in hour
            for k in range(Tgr_hr):
                # Since Velo is m/s, we use the t in sec with 1 hr as time step.
                t = t + dT_sec
                pot = ((Velo * t - flow_len) ** 2) / (4 * Diff * t)
                if pot <= 69:  # e^(-69) = E-30 a cut-off threshold
                    H = flow_len / (2 * t * (np.pi * t * Diff) ** 0.5) * np.exp(-pot)
                else:
                    H = 0
                UH_RRm[k] = H

            if sum(UH_RRm) == 0:
                UH_RRm[0] = 1.0
            else:
                UH_RRm = UH_RRm / sum(UH_RRm)  # Force UH_RRm to sum to 1

            # Much quicker!!  Think about S-hydrograph process.
            # And remember we should have [0] in UH_RRm[0].
            # Therefore, we use i+1 and 23 to form S-hydrolograph.
            FR = np.zeros((Tmax_hr + 23, Tmax_hr - 1))
            for i in range(Tmax_hr - 1):
                FR[:, i] = np.pad(UH_RRm, (i + 1, 23), "constant", constant_values=(0, 0))[
                    : Tmax_hr + 23
                ]
            FR = np.sum(FR, axis=1) / 24
            # Lag 24 hrs
            FR = (
                FR[:Tmax_hr]
                - np.pad(FR, (24, 0), "constant", constant_values=(0, 0))[:Tmax_hr]
            )

            # Aggregate to daily UH
            for t in range(T_RR):
                UH_RR[t] = sum(FR[(24 * (t + 1) - 24) : (24 * (t + 1) - 1)])

        if show:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(UH_RR, label=f"Velo={Velo}, Diff={Diff}, flow_len={flow_len}")
            ax.set_xlabel("Days")
            ax.set_ylabel("ratio")
            ax.legend()
            plt.show()

        return UH_RR

    @staticmethod
    def cal_UH_direct(shape, scale, Velo, Diff, flow_len, show=False):
        UH_IG = DP.cal_UH_IG(shape, scale, show=False)
        UH_RR = DP.cal_UH_RR(Velo, Diff, flow_len, show=False)

        T_IG = 12  # [day] Base time for within subbasin UH
        T_RR = 96  # [day] Base time for river routing UH
        UH_direct = np.zeros(T_IG + T_RR - 1)  # Convolute total time step [day]
        for k in range(0, T_IG):
            for u in range(0, T_RR):
                UH_direct[k + u] = UH_direct[k + u] + UH_IG[k] * UH_RR[u]
        UH_direct = UH_direct / sum(UH_direct)
        # Trim zero from back. So when we run routing, we don't need to run whole
        # array.
        UH_direct = np.trim_zeros(UH_direct, "b")

        if show:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(UH_direct, label="Direct UH")
            ax.set_xlabel("Days")
            ax.set_ylabel("ratio")
            ax.legend()
            plt.show()
        return UH_direct

    @staticmethod
    def plot_CCG_routing(routing):
        dp = DP()
        s = "C1"
        routing_subs = routing[s]
        dp.cal_UH_IG(shape=routing_subs[s]["Pars"]["GShape"], scale=routing_subs[s]["Pars"]["GScale"], show=True)
        us = "R1"
        dp.cal_UH_RR(Velo=routing_subs[us]["Pars"]["Velo"], Diff=routing_subs[us]["Pars"]["Diff"],
                    flow_len=routing_subs[us]["Inputs"]["FlowLength"], show=True)

        s = "C2"
        routing_subs = routing[s]
        dp.cal_UH_IG(shape=routing_subs[s]["Pars"]["GShape"], scale=routing_subs[s]["Pars"]["GScale"], show=True)
        us = "R2"
        dp.cal_UH_RR(Velo=routing_subs[us]["Pars"]["Velo"], Diff=routing_subs[us]["Pars"]["Diff"],
                    flow_len=routing_subs[us]["Inputs"]["FlowLength"], show=True)
        us = "R3"
        dp.cal_UH_RR(Velo=routing_subs[us]["Pars"]["Velo"], Diff=routing_subs[us]["Pars"]["Diff"],
                    flow_len=routing_subs[us]["Inputs"]["FlowLength"], show=True)

        s = "G"
        routing_subs = routing[s]
        dp.cal_UH_IG(shape=routing_subs[s]["Pars"]["GShape"], scale=routing_subs[s]["Pars"]["GScale"], show=True)
        us = "C1"
        dp.cal_UH_RR(Velo=routing_subs[us]["Pars"]["Velo"], Diff=routing_subs[us]["Pars"]["Diff"],
                    flow_len=routing_subs[us]["Inputs"]["FlowLength"], show=True)
        us = "C2"
        dp.cal_UH_RR(Velo=routing_subs[us]["Pars"]["Velo"], Diff=routing_subs[us]["Pars"]["Diff"],
                    flow_len=routing_subs[us]["Inputs"]["FlowLength"], show=True)

class FigHelper:
    def __init__(self):
        pass

    def add_conservation_policy(self, ax, obv_789, yshift=0):
        policy_start_year = 1985
        # Make sure obv_789 is a pandas Series with a DateTime index
        obv_789_filtered = obv_789['1985':'2024']  # filter years
        q50 = obv_789_filtered.quantile(0.5)       # 50% quantile
        low_vals = obv_789_filtered[obv_789_filtered <= q50]  # only values <= 50% quantile

        # Prepare data for regression
        # Convert time index to ordinal (number representation of dates)
        X = low_vals.index.year.values.reshape(-1, 1) - policy_start_year  # years since 1985
        y = low_vals.values

        # Fit linear regression
        reg = LinearRegression().fit(X, y)
        a = reg.coef_[0]
        b = reg.intercept_

        # Predict over the full time range for visualization
        X_full = obv_789_filtered.index.year.values.reshape(-1, 1) - policy_start_year # years since 1985
        y_pred = reg.predict(X_full)

        ax.plot(obv_789_filtered.index, y_pred + yshift, label=f"Policy: Q={a:.4f}x + {b:.2f}", c="r")

        y_1985 = b + yshift # y value at 1985 (x=0)
        ax.hlines(y=y_1985, xmin=obv_789.index[0], xmax=obv_789_filtered.index[0], color="darkred", label=f"Pre-policy: {y_1985:.2f}")

import operator
operators = {
        "<": operator.lt,  # Failure if x < threshold
        ">": operator.gt,   # Failure if x > threshold
        "<=": operator.le,  # Failure if x <= threshold
        ">=": operator.ge,   # Failure if x >= threshold
        "==": operator.eq,  # Failure if x == threshold
    }

# Management metrics
def reliability(x, threshold, failure_operator="<"):
    r"""
    Calculate reliability: Proportion of time steps where the system is in a satisfactory state.

    Parameters
    ----------
    x : array-like
        Time series data of the system state.
    threshold : float
        Threshold value to determine satisfactory states.
    failure_operator : str, optional
        Operator to determine satisfactory states. Default is "<".

    Returns
    -------
    float
        Reliability value in the range [0, 1].

    Notes
    -----
    The reliability is calculated as:

    .. math::

        \text{Reliability} = \frac{\text{Number of satisfactory states}}{\text{Total number of states}}

    where:
        - Satisfactory states are determined based on the `failure_operator` and `threshold`.
        - Total number of states excludes NaN values.
    """
    x = np.asarray(x)  # Convert to numpy array for easier manipulation

    # Validate the failure_operator
    if failure_operator not in operators:
        raise ValueError(f"Invalid failure_operator '{failure_operator}'. Valid options are {list(operators.keys())}.")

    # Count satisfactory states using the operator
    failures = np.sum(operators[failure_operator](x, threshold))
    total = np.count_nonzero(~np.isnan(x))  # Eliminate NaN values
    satisfactory = total - failures  # Satisfactory states are total states minus failures
    reliability = satisfactory / total
    
    return reliability
    


def resiliency(x, threshold, failure_operator="<", cap=float('inf')):
    """
    Calculate resiliency: Inverse of the average failure duration.
    Resiliency is in the range (0, cap]. Higher values indicate better resiliency.

    Parameters
    ----------
    x : array-like
        Time series data of the system state.
    threshold : float
        Threshold value to determine satisfactory states.
    failure_operator : str, optional
        Operator to determine satisfactory states. Default is "<".
    cap : float, optional
        Maximum value for resiliency. Default is infinity.
        
    Returns
    -------
    float
        Resiliency value in the range (0, cap].

    Notes
    -----
    Resiliency is calculated as the inverse of the average failure duration.
    """
    x = np.asarray(x)  # Convert to numpy array for easier manipulation
    
    if failure_operator not in operators:
        raise ValueError(f"Invalid failure_operator '{failure_operator}'. Use '<' or '>'.")

    r"""
    # Identify failure states
    failures = operators[failure_operator](x, threshold)

    # Calculate failure durations
    failure_durations = []
    duration = 0
    for f in failures:
        if f:
            duration += 1  # Increase duration for failures
        else:
            if duration > 0:
                failure_durations.append(duration)  # Record completed failure duration
                duration = 0  # Reset for next failure
    if duration > 0:  # If the last failure period is at the end
        failure_durations.append(duration)

    # Calculate resiliency
    if failure_durations:
        avg_failure_duration = np.mean(failure_durations)  # Average failure duration
        resiliency = 1 / avg_failure_duration  # Inverse of the average duration
    else:  # No failures
        resiliency = cap

    return resiliency
    """
    # Faster version
    fails = operators[failure_operator](x, threshold)
    padded = np.pad(fails.astype(int), (1, 1))
    diff = np.diff(padded)
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0]
    durations = end_idx - start_idx
    if len(durations) == 0:
        return cap
    else:
        return 1 / np.mean(durations)
    
def vulnerability(x, threshold, failure_operator="<"):
    """
    Calculate vulnerability: Expected severity of failures.
    Vulnerability is in the range [0, inf], the lower the better.

    Parameters
    ----------
    x : array-like
        Time series data of the system state.
    threshold : float
        Threshold value to determine satisfactory states.
    failure_operator : str, optional
        Operator to determine satisfactory states. Default is "<".

    Returns
    -------
    float
        Vulnerability value in the range [0, inf].

    Notes
    -----
    Vulnerability is calculated as the average of the maximum violations
    for each contiguous failure event.
    """
    x = np.asarray(x)  # Convert to numpy array for easier manipulation

    if failure_operator not in operators:
        raise ValueError(f"Invalid failure_operator '{failure_operator}'. Use '<' or '>'.")

    # Identify failure states
    failures = operators[failure_operator](x, threshold)

    # Identify contiguous failure events
    failure_events = []
    current_event = []

    for i, failure in enumerate(failures):
        if failure:
            current_event.append(x[i])
        else:
            if current_event:
                failure_events.append(current_event)
                current_event = []

    if current_event:
        failure_events.append(current_event)

    # Calculate the maximum violation for each event
    max_violations = [
        max(abs(value - threshold) for value in event)
        for event in failure_events
    ]

    # Calculate the average of the maximum violations
    if max_violations:
        vulnerability = np.mean(max_violations)
    else:
        vulnerability = 0  # No failures, so vulnerability is zero

    return vulnerability

# =============================================================================
# Borg post processing
# =============================================================================

class BorgPostProcessing():
    def __init__(self, pn, exp_folder):
        self.pn = pn
        self.exp_folder = exp_folder

    def read_metric_file(self, file_path):
        with open(file_path, 'r') as file:
            # Read the first line to get the column names and replace '#' with ''
            columns = [col for col in file.readline().strip().split() if "#" not in col]

            # Read the rest of the file into a DataFrame
            df = pd.read_csv(file, sep='\\s+', names=columns)

        return df

    def read_metrics(self, metric="Hypervolume"):
        pn = self.pn
        folder = self.exp_folder

        files = [i for i in os.listdir(pn.outputs.get(folder, "metrics")) if ".metric" in i]

        df = pd.DataFrame()
        for file in files:
            file_path = pn.outputs.get(folder, "metrics", file)
            try:
                seed = [int(i.split(".")[0][4:]) for i in file.split("_") if "seed" in i][0]
            except:
                seed = "--"
            df[seed] = self.read_metric_file(file_path)[metric]

        return df

    def read_ref(self, filename=None):
        if filename is None:
            filename = self.pn.outputs.get(self.exp_folder, "borg.ref")
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract number of variables and objectives
        num_vars = int([line for line in lines if line.startswith('# NumberOfVariables=')][0].split('=')[1])
        num_objs = int([line for line in lines if line.startswith('# NumberOfObjectives=')][0].split('=')[1])

        # Construct header names
        headers = [f'var{i+1}' for i in range(num_vars)] + [f'obj{i+1}' for i in range(num_objs)]

        # Extract data lines (non-comment lines)
        data_lines = [line.strip() for line in lines if not line.strip().startswith('#') and line.strip()]

        # Convert to float values
        data = [list(map(float, line.split())) for line in data_lines]

        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        return df

    def plot_convergence(self, df=None, freq=1, xlabel_freq=10):
        if df is None:
            df = self.read_metrics()
        fig, ax = plt.subplots()
        df.plot(ax=ax, legend=False)
        ax.set_xlabel("NFE")
        ax.set_ylabel("Hypervolume")
        xticks = df.index[::xlabel_freq]
        ax.set_xticks(xticks)  # Set tick positions
        ax.set_xticklabels(np.array(xticks+1) * freq)  # Set tick labels as xticks * freq
        ax.legend(ncols=5, frameon=False, title="Seed")
        plt.show()

    def plot_parallel_axes(self, df, hue='Reliability', orders=None, browser=True, no_ctrl_objs=None, obv_tr_objs=None):
        if browser:
            import plotly.io as pio
            pio.renderers.default = "browser"

        # Normalize reference values to match the Parcoords scale (0-1 range)
        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        # Compute the range for each dimension
        if orders is None:
            orders = list(df.columns)

        axes_config = {
            "Reliability": dict(label="Reliability", range=[1, 0]),  # Reverse scale
            "Resilience": dict(label="Resilience", range=[1, 0]),  # Reverse scale
            "Vulnerability": dict(label="Vulnerability"),  # min/max from df
            "AccumulatedDD\n(t0=21.44)": dict(label="AccumulatedDD\n(t0=21.44)"),
            "Thermal bank usage\n(%)": dict(label="Thermal bank usage\n(%)"),
            "nrmse_daily": dict(label="nrmse_daily", range=[0, 1]),
            "nrmse_annual": dict(label="nrmse_annual", range=[0, 1]),
            "inconsistency_ratio_NoTr|Tr": dict(label="inconsistency_ratio_NoTr|Tr", range=[0, 1]),
            "inconsistency_ratio_Tr|NoTr": dict(label="inconsistency_ratio_Tr|NoTr", range=[0, 1])
        }

        # Build options dictionary only for columns that exist in df
        options = {}
        for key, config in axes_config.items():
            if key in df.columns:
                values = df[key]
                # Automatically set range if not defined in config
                if "range" not in config:
                    config["range"] = [values.min(), values.max()]
                config["values"] = values
                options[key] = config

        dimensions = [options[i] for i in orders]

        # Create Parallel Coordinates Figure
        fig = go.Figure()

        fig.add_trace(
            go.Parcoords(
                line=dict(color=df[hue],
                        colorscale='Tealrose',
                        showscale=True,
                        colorbar=dict(
                                title=hue,  # Set color bar label
                                len=0.7
                            )
                        ),
                dimensions=dimensions
            )
        )

        if obv_tr_objs is not None:
            normalized_obv_tr_objs = [
                normalize(obv_tr_objs[dim["label"]], dim["range"][0], dim["range"][1]) for dim in dimensions
            ]
            # Add Overlay Line
            fig.add_trace(go.Scatter(
                x=[dim["label"] for dim in dimensions],
                y=normalized_obv_tr_objs,
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                name='With obv tr',
            ))

            # Force Y-axis limits from 0 to 1
            fig.update_layout(
                xaxis=dict(
                    showticklabels=False,  # Hide x-axis labels
                    showgrid=False,  # Remove x-grid
                    zeroline=False,  # Remove x-axis zero line
                    range=[0, len(orders)-1]
                ),
                yaxis=dict(
                    showticklabels=False,  # Hide y-axis labels
                    showgrid=False,  # Remove y-grid
                    zeroline=False,  # Remove y-axis zero line
                    range=[0, 1]  # Keep y in 0-1 range
                ),
            )

        if no_ctrl_objs is not None:
            normalized_no_ctrl_objs = [
                normalize(no_ctrl_objs[dim["label"]], dim["range"][0], dim["range"][1]) for dim in dimensions
            ]
            # Add Overlay Line
            fig.add_trace(go.Scatter(
                x=[dim["label"] for dim in dimensions],
                y=normalized_no_ctrl_objs,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                name='No tr',
            ))

            # Force Y-axis limits from 0 to 1
            fig.update_layout(
                xaxis=dict(
                    showticklabels=False,  # Hide x-axis labels
                    showgrid=False,  # Remove x-grid
                    zeroline=False,  # Remove x-axis zero line
                    range=[0, len(orders)-1]
                ),
                yaxis=dict(
                    showticklabels=False,  # Hide y-axis labels
                    showgrid=False,  # Remove y-grid
                    zeroline=False,  # Remove y-axis zero line
                    range=[0, 1]  # Keep y in 0-1 range
                ),
            )

        fig.show()