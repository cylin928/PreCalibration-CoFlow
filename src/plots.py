import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import FigHelper
from cmap import Colormap
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec

def plot_hydro_sensitivity_index(Si_dict, y="KGE_G(789)"):
    df = pd.DataFrame()
    df["si"] = Si_dict[y]["delta"]
    names = Si_dict[y]["names"]
    df["parameter"] = [i.split("|")[0] for i in names]
    df["node"] = [i.split("|")[1] for i in names]
    df = df.replace("DivFactor", "DivF")
    df = df.replace("ReturnFactor", "ReF")

    # Custom order
    param_order = ["CN2", "IS", "Res", "Sep", "Alpha", "Beta", "Ur", "Df", "Kc", 'DivF', 'ReF']
    node_order = ['C1', 'C2', 'G', 'Kittitas', 'Tieton']

    # Pivot table
    heatmap_data = df.pivot(index="parameter", columns="node", values="si")
    heatmap_data = heatmap_data.loc[param_order, node_order]

    # Normalize the data to [0, 1]
    heatmap_data = (heatmap_data - heatmap_data.min().min()) / (heatmap_data.max().max() - heatmap_data.min().min())
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis_r", ax=ax)
    if y == "KGE_G(789)":
        plt.title("Normalized Sensitivity Index on Summer Flow")
    elif y == "KGE_Div":
        plt.title("Normalized Sensitivity Index on Diversion")
    plt.xlabel("Node")
    plt.ylabel("Parameter\n\n")

    # Get y-tick positions
    yticks = ax.get_yticks()
    plt.tight_layout()
    plt.show()


def plot_abm_sensitivity_index(Si_dict, y):
    df = pd.DataFrame()
    df["si"] = Si_dict[y]["delta"]
    names = Si_dict[y]["names"]
    df["parameter"] = [i.split("|")[0] for i in names]
    df["agent"] = [i.split("|")[1] for i in names]
    df = df.replace("ProratedRatio", "PR")

    # Custom order
    param_order = ['Lr_c', 'L_U', 'L_L', 'a', 'b', 'Sig', 'PR']
    agent_order = ['Kittitas', 'Tieton', 'Roza', 'Sunnyside', 'Wapato']

    # Pivot table
    heatmap_data = df.pivot(index="parameter", columns="agent", values="si")
    heatmap_data = heatmap_data.loc[param_order, agent_order]
    
    # Normalize the data to [0, 1]
    heatmap_data = (heatmap_data - heatmap_data.min().min()) / (heatmap_data.max().max() - heatmap_data.min().min())

    names = {"KGE_G(789)": "KGE of Summer Flow", "KGE_Div": "KGE of Diversion"}
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis_r", ax=ax)
    if y == "KGE_G(789)":
        plt.title("Normalized Sensitivity Index on Summer Flow")
    elif y == "KGE_Div":
        plt.title("Normalized Sensitivity Index on Diversion")
    plt.xlabel("Agent")
    plt.ylabel("Parameter\n\n")

    # Get y-tick positions
    yticks = ax.get_yticks()

    # Define groupings (start index, end index, label)
    group_labels = [
        (0, 2, "Learning"),
        (3, 4, "Adaptive"),
        (5, 5, "Social"),
        (6, 6, "Drought"),
    ]

    # Add brackets and labels
    for start, end, label in group_labels:
        y0 = yticks[start]
        y1 = yticks[end]
        x = -0.4  # adjust as needed to the left
        ax.plot([x, x], [y0, y1], color='black', lw=1, clip_on=False)
        ax.plot([x, x + 0.05], [y0, y0], color='black', lw=1, clip_on=False)
        ax.plot([x, x + 0.05], [y1, y1], color='black', lw=1, clip_on=False)
        ax.text(x - 0.1, (y0 + y1)/2, label, va='center', ha='center', fontsize=9, rotation=90)

    plt.tight_layout()
    plt.show()


# def add_flow_div_ts_with_uc(axes, df_ts, colors, label=""):    
#     variables = ['G789', 'Div']  
#     names = {"G789": "Summer flow\n(cms)", "Div": "Diversion\n(cms)"}
    
#     # Plot each variable
#     for ax, col, color in zip(axes, variables, colors):
#         grouped = df_ts.groupby('Year')[col]
#         mean = grouped.mean()
#         std = grouped.std()
    
#         ax.plot(mean.index, mean, color=color)
#         ax.fill_between(mean.index, mean - 2*std, mean + 2*std, 
#                         alpha=0.2, color=color, label=label)
    
#         ax.set_ylabel(names[col])
#         ax.set_xlim([1980, 2023])
#         #ax.legend(loc='upper right')

def add_flow_div_ts_with_uc(axes, df_ts, colors, label="", fill=True, linestyle='-'):
    variables = ['G789', 'Div']  
    names = {"G789": "Summer flow\n(cms)", "Div": "Diversion\n(cms)"}
    
    for ax, col, color in zip(axes, variables, colors):
        grouped = df_ts.groupby('Year')[col]
        mean = grouped.mean()
        std = grouped.std()
        
        # Plot mean line
        ax.plot(mean.index, mean, color=color, linestyle='-', label=label)

        # Plot uncertainty
        lower = mean - 2 * std
        upper = mean + 2 * std

        if fill:
            ax.fill_between(mean.index, lower, upper, alpha=0.2, color=color, label=None)
        else:
            ax.plot(mean.index, lower, color=color, linestyle=linestyle, linewidth=0.8)
            ax.plot(mean.index, upper, color=color, linestyle=linestyle, linewidth=0.8)

        ax.set_ylabel(names[col])
        ax.set_xlim([1980, 2023])


def plot_flow_div_ts_with_uc(df_ts, df_met, kge_thresholds=[0.1, 0.3]):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 4), sharex=True)
    cm_br = Colormap('colorbrewer:Blues_6')
    cm_gr = Colormap('colorbrewer:BuGn_6')
    #kge_thresholds = [0.1, 0.3]
    for i, kge_threshold in enumerate(kge_thresholds):
        if len(kge_thresholds) == 1:
            colors = [cm_br(4), cm_gr(4)]
        else:
            colors = [cm_br(i*2+2), cm_gr(i*2+2)]
        #kge_threshold = 0
        df_met_ = df_met.copy()
        df_met_ = df_met_[df_met_["KGE_G(789)"] >= kge_threshold]
        try:
            df_met_ = df_met_[df_met_["KGE_Div"] >= kge_threshold]
        except:
            print("KGE_Div filter is not used.")

        selected_index = list(df_met_.index)
        print(len(selected_index))
        df_ts_selected = df_ts[df_ts['global_index'].isin(selected_index)]
        add_flow_div_ts_with_uc(
            axes=axes, df_ts=df_ts_selected, colors=colors,
            label=f"KGE ≥ {kge_threshold} [{len(selected_index)}]")

    axes[0].legend(frameon=False, fontsize=6)
    axes[1].legend(frameon=False, fontsize=6)
    # Common X label and title
    axes[1].set_xlabel("Year")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_flow_div_ts_with_uc_with_hydro_overlap(df_ts, df_met, df_ts_narrow, df_met_narrow, kge_thresholds=[0.1, 0.3]):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 4), sharex=True)
    cm_br = Colormap('colorbrewer:Blues_6')
    cm_gr = Colormap('colorbrewer:BuGn_6')

    for i, kge_threshold in enumerate(kge_thresholds):
        if len(kge_thresholds) == 1:
            colors = [cm_br(4), cm_gr(4)]
        else:
            colors = [cm_br(i*2+2), cm_gr(i*2+2)]
        line_colors = ['#08306B', '#00441B']  # Dark blue, dark green for dashed lines

        # === Wide bands ===
        df_met_ = df_met.copy()
        df_met_ = df_met_[df_met_["KGE_G(789)"] >= kge_threshold]
        try:
            #df_met_ = df_met_[df_met_["KGE_Div"] >= kge_threshold]
            pass
        except KeyError:
            print("KGE_Div filter is not used.")

        selected_index = list(df_met_.index)
        print(f"WIDE BAND - KGE ≥ {kge_threshold}: {len(selected_index)} samples")
        df_ts_selected = df_ts[df_ts['global_index'].isin(selected_index)]

        add_flow_div_ts_with_uc(
            axes=axes, df_ts=df_ts_selected, colors=colors,
            label=f"KGE ≥ {kge_threshold} [{len(selected_index)}]", fill=True)

        # === Narrow bands (dashed boundaries) ===
        df_met_n_ = df_met_narrow.copy()
        df_met_n_ = df_met_n_[df_met_n_["KGE_G(789)"] >= kge_threshold]
        try:
            df_met_n_ = df_met_n_[df_met_n_["KGE_Div"] >= kge_threshold]
        except KeyError:
            print("KGE_Div filter is not used for narrow band.")

        selected_index_n = list(df_met_n_.index)
        print(f"NARROW BAND - KGE ≥ {kge_threshold}: {len(selected_index_n)} samples")
        df_ts_selected_n = df_ts_narrow[df_ts_narrow['global_index'].isin(selected_index_n)]

        add_flow_div_ts_with_uc(
            axes=axes, df_ts=df_ts_selected_n, colors=line_colors,
            label=f"KGE ≥ {kge_threshold} [{len(selected_index_n)}]", fill=False, linestyle='--')

    axes[0].legend(frameon=False, fontsize=6)
    axes[1].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel("Year")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_sd_difference_boxplot(df_ts, df_met, df_ts_narrow, df_met_narrow, kge_threshold=0.3):
    variables = ['G789', 'Div']
    variable_names = {'G789': 'Summer\nFlow', 'Div': 'Diversion'}

    # === Filter wide band data ===
    df_met_ = df_met[df_met["KGE_G(789)"] >= kge_threshold]
    try:
        df_met_ = df_met_[df_met_["KGE_Div"] >= kge_threshold]
    except KeyError:
        print("KGE_Div filter not used in wide band.")
    df_ts_sel = df_ts[df_ts['global_index'].isin(df_met_.index)]

    # === Filter narrow band data ===
    df_met_n_ = df_met_narrow[df_met_narrow["KGE_G(789)"] >= kge_threshold]
    try:
        df_met_n_ = df_met_n_[df_met_n_["KGE_Div"] >= kge_threshold]
    except KeyError:
        print("KGE_Div filter not used in narrow band.")
    df_ts_sel_n = df_ts_narrow[df_ts_narrow['global_index'].isin(df_met_n_.index)]

    # === Calculate per-year SDs and differences ===
    sd_diffs = {}
    for var in variables:
        sd_wide = df_ts_sel.groupby('Year')[var].std()
        sd_narrow = df_ts_sel_n.groupby('Year')[var].std()

        # Align by year (intersection)
        common_years = sd_wide.index.intersection(sd_narrow.index)
        diff = (sd_narrow[common_years] - sd_wide[common_years]).dropna()
        sd_diffs[var] = diff.values

    # === Plot boxplot ===
    fig, ax = plt.subplots(figsize=(2.5, 3.5))
    bp = ax.boxplot([sd_diffs['G789'], sd_diffs['Div']],
               labels=[variable_names['G789'], variable_names['Div']],
               patch_artist=True,
               boxprops=dict(facecolor='lightgray'),
               medianprops=dict(color='black'))
    
    # Apply custom colors
    box_colors = ['#6baed6', '#74c476'] 
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(f"Differences in standard deviation\n(single best prior - ensemble priors)\n(cms), [KGE ≥ {kge_threshold}]")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    fig.tight_layout()
    plt.show()



def plot_error_vs_management_metrics(df, dfs=None, key='KGE_G(789)'):
    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(nrows=3, ncols=2, figure=fig)
    
    if key == 'KGE_G(789)':
        name = "Summer flow"
        name_p = "flow target violation"
    elif key == 'KGE_Div':
        name = "Diversion"
        name_p = "diversion shortage"
    
    # Left column: one large plot
    ax_left = fig.add_subplot(gs[:, 0])
    data = df[key].dropna()
    ax_left.hist(data, bins=50, density=True, alpha=0.6, color='lightcoral', edgecolor='black')
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200)
    ax_left.plot(x_vals, kde(x_vals), color='darkred', lw=2)
    ax_left.set_title(name)
    ax_left.set_xlabel(name+" (cms)")
    ax_left.set_ylabel("Density")
    if dfs is not None:
        for i, df_ in enumerate(dfs):
            data = df_[key].dropna()
            #ax_left.hist(data, bins=50, density=True, alpha=0.6, color='lightcoral', edgecolor='black')
            kde = gaussian_kde(data)
            #x_vals = np.linspace(data.min(), data.max(), 200)
            ax_left.plot(x_vals, kde(x_vals), color='darkred', lw=2, ls="--")
    
    # Right column: three stacked plots
    cols = ['Reliability', 'Resiliency', 'Vulnerability']
    names = [f'Reliability\n({name_p} frequency)', f'Resiliency\n({name_p} duration)', f'Vulnerability\n(max {name_p})']
    for i, col in enumerate(cols):
        ax = fig.add_subplot(gs[i, 1])
        data = df[col].dropna()
        
        ax.hist(data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_vals, kde(x_vals), color='darkblue', lw=2)
    
        ax.set_title(names[i])
        #ax.set_ylabel("Density")
        if i == 2:
            ax.set_xlabel("Values")
            
        if dfs is not None:
            for i, df_ in enumerate(dfs):
                data = df_[col].dropna()
                #ax.hist(data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
                kde = gaussian_kde(data)
                #x_vals = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_vals, kde(x_vals), color='darkblue', lw=2, ls="--")
    
    plt.tight_layout()
    plt.show()

def plot_sd_difference_boxplot_management_metrics(df_ts, df_met, df_ts_narrow, df_met_narrow, kge_threshold=0.3):
    metrics = ['Reliability', 'Resiliency', 'Vulnerability']
    metric_names = {
        'Reliability': 'Reliability\n(frequency)',
        'Resiliency': 'Resiliency\n(duration)',
        'Vulnerability': 'Vulnerability\n(severity)'
    }

    # === Filter wide band data ===
    df_met_ = df_met[df_met["KGE_G(789)"] >= kge_threshold]
    try:
        df_met_ = df_met_[df_met_["KGE_Div"] >= kge_threshold]
    except KeyError:
        print("KGE_Div filter not used in wide band.")
    df_ts_sel = df_ts[df_ts['global_index'].isin(df_met_.index)]

    # === Filter narrow band data ===
    df_met_n_ = df_met_narrow[df_met_narrow["KGE_G(789)"] >= kge_threshold]
    try:
        df_met_n_ = df_met_n_[df_met_n_["KGE_Div"] >= kge_threshold]
    except KeyError:
        print("KGE_Div filter not used in narrow band.")
    df_ts_sel_n = df_ts_narrow[df_ts_narrow['global_index'].isin(df_met_n_.index)]

    # === Calculate per-year SDs and differences ===
    sd_diffs = {}
    for metric in metrics:
        sd_wide = df_ts_sel.groupby('Year')[metric].std()
        sd_narrow = df_ts_sel_n.groupby('Year')[metric].std()

        # Align by year (intersection)
        common_years = sd_wide.index.intersection(sd_narrow.index)
        diff = (sd_narrow[common_years] - sd_wide[common_years]).dropna()
        sd_diffs[metric] = diff.values

    # === Plot boxplot ===
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    bp = ax.boxplot([sd_diffs[m] for m in metrics],
                    labels=[metric_names[m] for m in metrics],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightgray'),
                    medianprops=dict(color='black'))

    # Apply custom colors
    box_colors = ['#9ecae1', '#a1d99b', '#fcae91']  # Blue, green, coral
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel(f"Differences in standard deviation\n(narrow - wide) [KGE ≥ {kge_threshold}]")
    fig.tight_layout()
    plt.show()



def C1C2G_monthly_ts_with_KGE(sim_Q_M, obv_flow=None, met_dict=None):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 5), sharex=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    for i, s in enumerate(["C1", "C2", "G"]):
        ax = axes[i]
        ax.plot(sim_Q_M[s], label=s)
        if obv_flow is not None:
            ax.plot(obv_flow[s], lw=1, ls="--", c="k")
        #ax.set_ylim([0, 100])
        #ax.set_title(f"Month: {s}", fontsize=10)
        if met_dict is not None:
            ax.set_ylabel(f"{s}\nKGE={round(met_dict['KGE_'+s], 3)}")
        else:
            ax.set_ylabel(s)
        ax.grid(True, linestyle='--', alpha=0.5)
        #ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show()
    
def divs_annual_ts_with_KGE(div_Y, obv_div=None, met_dict=None, pr_nov_to_jun_sum=None):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 6), sharex=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
    for i, ag in enumerate(ag_list):
        ax = axes[i]
        ax.plot(div_Y[ag], label=ag)
        if obv_div is not None:
            ax.plot(obv_div[ag], lw=1, ls="--", c="k")
        #ax.set_ylim([0, 100])
        if met_dict is not None:
            kge = round(met_dict[f"KGE_{ag}"], 3)
            ax.set_ylabel(f"{ag}\nKGE={kge}")
        else:
            ax.set_ylabel(ag)
        ax.grid(True, linestyle='--', alpha=0.5)

    # ax = axes[5]
    if pr_nov_to_jun_sum is not None:
        pr_nov_to_jun_sum = pd.DataFrame({"pr_nov_to_jun_sum":pr_nov_to_jun_sum}, index=div_Y.index)

        # ax.plot(pr_nov_to_jun_sum["pr_nov_to_jun_sum"])
        #ax.plot(df_nov_to_mar_sum["pr_nov_to_mar"])
        threshold = 315 #331#
        for i, v in enumerate(pr_nov_to_jun_sum["pr_nov_to_jun_sum"]):
            if v <= threshold:
                for ax in axes:
                    ax.axvline(pr_nov_to_jun_sum.index[i], c="grey", ls="--", lw=1)

    # threshold = 240
    # for i, v in enumerate(df_nov_to_mar_sum["pr_nov_to_mar"]):
    #     if v <= threshold:
    #         for ax in axes:
    #             ax.axvline(df_nov_to_mar_sum.index[i], c="red", ls="--")

    # ax.plot(div_Y[ag_list_].sum(axis=1), label=ag)
    # ax.plot(obv_div[ag_list_].sum(axis=1), lw=1, ls="--", c="k")

    plt.tight_layout()
    plt.show()
    
def summer_flow_ts_with_policy(sim_Q_M, obv_flow=None, kge=None):
    mask = [True if i.month in [7,8,9] else False for i in sim_Q_M.index]
    if obv_flow is not None:
        obv_789 = obv_flow[mask].resample("YS").mean()["G"]
    sim_789 = sim_Q_M[mask].resample("YS").mean()["G"]

    fh = FigHelper()
    fig, ax = plt.subplots()
    if obv_flow is not None:
        ax.plot(obv_789, label="obv_flow", c="k", ls="--")
    ax.plot(sim_789, label="sim")
    if obv_flow is not None:
        fh.add_conservation_policy(ax, obv_789)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.18), ncol=2, frameon=False)
    if kge is not None:
        ax.set_ylabel(f"Annual mean streamflow of Jul-Sep (cms)\nKGE={kge}")
    else:
        ax.set_ylabel("Annual mean streamflow of Jul-Sep (cms)")
    ax.set_xlabel("Year")
    plt.show()