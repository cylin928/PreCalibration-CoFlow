import os
import pandas as pd
import pathnavigator

# Set working directory
if os.name == 'nt':  # 'nt' means Windows
    root_dir = rf"C:\Users\{os.getlogin()}\Documents\GitHub\PreCalibration-CoFlow"
else:
    root_dir = os.path.expanduser("~/Github/PreCalibration-CoFlow")
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.utils import reliability, resiliency, vulnerability

# Function to compute violation and shortage metrics
def get_df_vio_shortage(df_ts, df_met, kge_threshold=None):
    df_ts = df_ts.copy()
    df_ts = df_ts[['Violation', 'Shortage', 'global_index']]
    df_met = df_met.copy()

    def compute_metrics_per_group(df_ts, col='Violation'):
        if col == 'Violation':
            failure_operator = "<"
        elif col == 'Shortage':
            failure_operator = ">"

        # Define vectorized function to pass into agg
        def rel(x):
            return reliability(x=x, threshold=0, failure_operator=failure_operator)

        def res(x):
            return resiliency(x=x, threshold=0, failure_operator=failure_operator, cap=float('inf'))

        def vul(x):
            return vulnerability(x=x, threshold=0, failure_operator=failure_operator)

        df_metrics = df_ts.groupby("global_index")[col].agg(
            Reliability=rel,
            Resiliency=res,
            Vulnerability=vul
        ).reset_index()

        return df_metrics

    #df = pd.read_csv(metrics_csv_path, index_col=[0])
    #df_met = df_met[df_met["KGE_G(789)"]>=kge_threshold]
    #df_met = df_met[df_met["KGE_Div"]>=kge_threshold]

    selected_index = list(df_met.index)
    if len(df_ts["global_index"].unique()) != len(selected_index):
        df_ts_selected = df_ts[df_ts['global_index'].isin(selected_index)]
    else:
        df_ts_selected = df_ts#[df_ts['global_index'].isin(selected_index)]

    # Apply the function to each group by global_index
    df_vio = compute_metrics_per_group(df_ts_selected, col='Violation')
    df_vio["KGE_G(789)"] = df_met["KGE_G(789)"].values
    df_vio = df_vio[['KGE_G(789)', 'Reliability', 'Resiliency', 'Vulnerability']]

    df_shortage = compute_metrics_per_group(df_ts_selected, col='Shortage')
    df_shortage["KGE_Div"] = df_met["KGE_Div"].values
    df_shortage = df_shortage[['KGE_Div', 'Reliability', 'Resiliency', 'Vulnerability']]
    return df_vio, df_shortage

#%% Take long time (avoid)
df_met_ensemble = pd.read_parquet(pn.get(r"component_wise_precalibration_hyabm\output", "component_wise_precalibration_hyabm_135347_selected_metrics.parquet"))
df_ts_ensemble = pd.read_parquet(pn.get(r"component_wise_precalibration_hyabm\output", "component_wise_precalibration_hyabm_135347_ts_selected.parquet"))

df_vio_ensemble, df_shortage_ensemble = get_df_vio_shortage(df_ts_ensemble, df_met_ensemble)

df_vio_ensemble.to_parquet(pn.get(r"component_wise_precalibration_hyabm\output") / "df_vio_ensemble.parquet")
df_shortage_ensemble.to_parquet(pn.get(r"component_wise_precalibration_hyabm\output") / "df_shortage_ensemble.parquet")

#%%
df_met_whole = pd.read_parquet(pn.get(r"whole_system_precalibration\output", "precali_coupledABM_run_135667_selected_metrics.parquet"))
df_ts_whole = pd.read_parquet(pn.get(r"whole_system_precalibration\output", "precali_coupledABM_run_135667_ts_selected.parquet"))

# selected_index = list(df_met_whole.index)
# if len(df_ts_whole["global_index"].unique()) != len(selected_index):
#     df_ts_selected = df_ts_whole[df_ts_whole['global_index'].isin(selected_index)]
# df_ts_selected.to_parquet(pn.get(r"whole_system_precalibration\output", "precali_coupledABM_run_135667_ts_selected.parquet"))

df_vio_whole, df_shortage_whole = get_df_vio_shortage(df_ts_whole, df_met_whole)

df_vio_whole.to_parquet(pn.get(r"whole_system_precalibration\output") / "df_vio_whole.parquet")
df_shortage_whole.to_parquet(pn.get(r"whole_system_precalibration\output") / "df_shortage_whole.parquet")

#%%
df_vio_ensemble.hist()
df_vio_whole.hist()
#%%
import matplotlib.pyplot as plt
import numpy as np

# List of metrics to compare
metrics = ["KGE_G(789)", "Reliability", "Resiliency", "Vulnerability"]

# Create a figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, col in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    
    # Ensemble CDF
    sorted_ens = np.sort(df_vio_ensemble[col])
    cdf_ens = np.arange(1, len(sorted_ens)+1) / len(sorted_ens)
    ax.plot(sorted_ens, cdf_ens, label="Ensemble", lw=2)
    
    # Whole-system CDF
    sorted_whole = np.sort(df_vio_whole[col])
    cdf_whole = np.arange(1, len(sorted_whole)+1) / len(sorted_whole)
    ax.plot(sorted_whole, cdf_whole, label="Whole", lw=2)
    
    ax.set_title(col)
    ax.legend()

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt

# List of metrics to compare
metrics = ["KGE_G(789)", "Reliability", "Resiliency", "Vulnerability"]

# Create a figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, col in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    ax.hist(df_vio_ensemble[col], alpha=0.6, bins=15, label="Ensemble")
    ax.hist(df_vio_whole[col], alpha=0.6, bins=15, label="Whole")
    ax.set_title(col)
    ax.legend()

plt.tight_layout()
plt.show()
