# By Chung-Yi Lin 2021/02/17

# This file is to process Hydromet files.
# Data download: https://www.usbr.gov/pn/hydromet/yakima/yakwebarcread.html
# Variables and units: https://www.usbr.gov/pn/hydromet/data/hydromet_pcodes.html

"""
Reservoir Storage:
    KEE	Keechelus Reservoir
    KAC	Kachess Reservoir
    CLE	Cle Elum Reservoir
    RIM	Rimrock Reservoir
    BUM	Bumping Reservoir
    CLR	Clear Lake
    EDR	Easton Diversion Dam
    RDR	Roza Dam
    PAR	Yakima River nr. Parker Sunnyside Pool
    PRO	Prosser Dam

Stream Flows
    KEE	    Keechulus Reservoir
    YRCW	Yakima River at Crystal Springs
    KAC	    Kachess Reservoir
    EASW	Yakima River at Easton
    CLE	    Cle Elum Reservoir
    YUMW	Yakima River at Cle Elum
    TNAW	Teanaway River at Forks nr Cle Elum
    YRWW	Yakima River near Horlick
    ELNW	Yakima River near Ellensburg
    CHRW	Cherry Creek(Thrall Rd. Kittitas VAL)
    WONW	Wilson Creek(Thrall Rd. Kittitas VAL)
    UMTW	Yakima River near Umtanum
    RBDW	Yakima River below Roza Dam
    BUM	    Bumping Reservoir
    AMRW	American River near Nile
    LNRW	Little Naches River near Nile
    RIM	    Rimrock Reservoir
    TICW	Tieton River below Tieton Canal Diversion Dam
    CLFW	Naches River near Cliffdell
    NACW	Naches River near Naches
    PARW	Yakima River near Parker
    YRPW	Yakima River near Prosser
    KIOW	Yakima River at Kiona
    YGVW	Yakima River at Euclid Rd Br. near Grandview
    SUCW	Sulpher Creek at Holiday Rd near Sunnyside

Canal Data
    ETCW	Ellensburg Town Canal
    CHCW	Chandler-Prosser Power Canal
    KNCW	Kennewick Canal
    KTCW	Kittitas Canal
    NSCW	Naches Selah Canal
    ROZW	Roza Canal at 11.0 Mile
    RSCW	New Reservtion Canal
    RZCW	Roza Canal at Headworks
    SEXW	Selah/Moxee Canal
    SNCW	Sunnyside Canal
    SOUW	South Naches Channel Company Canal
    TIEW	Tieton Canal
    WESW	Westside Canal
    WOPW	Wapatox Power Canal

Parameter	Description
    AF	Reservoir Active Storage (Acre-Feet)
    BH	Barometric Pressure, Daily Average (mmHg)
    FB	Reservoir Water Surface Elevation (Feet)
    GD	Gauge Height, Daily Average (Feet)
    ID	Reservoir Inflow, Computed Daily Average (Cubic Feet per Second)
    MM	Air Temperature, Daily Average (Degrees F)
    MN	Air Temperature, Daily Minimum (Degrees F)
    MX	Air Temperature, Daily Maximum (Degrees F)
    NT	Total Dissolved Gas (mmHg)
    PC	Precipitation, Cumulative (Inches)
    PE	Pan Evaporation, Daily Total (Inches)
    PP	Precipitation, Daily Total (Inches)
    PU	Cumulative Water Year Precipitation, Oct. 1 to Date (Inches)
    PX	Precipitation, Daily Total, Manually Observed (Inches)
    QD	Discharge, Daily Average (Cubic Feet per Second)
    QJ	Canal Discharge, Daily Average (Cubic Feet per Second)
    QT	Discharge, Totalled Daily Average (Cubic Feet per Second)
    QU	Unregulated Flow, Estimated Daily Average (Cubic Feet per Second)
    QX	Discharge, Daily Maximum (Cubic Feet per Second)
    SO	Accumulated Snow Water, Oct. 1 to Date (Equivalent Inches of Water)
    SE	Snow Water Content (NRCS sites) (Equivalent Inches of Water)
    SP	Snow Water Content (Equivalent Inches of Water)
    SR	Solar Radiation (Langleys)
    TA	Humidity, Daily Average (%)
    UA	Wind Speed, Daily Average (mph)
    UD	Wind Direction, Daily Resultant (Degrees Azimuth)
    WK	Water Temperature, Daily Maximum (Degrees F)
    WI	Water Temperature, Daily Minimum (Degrees F)
    WM	Water Temperature, Daily Maximum (Degrees Celsius)
    WN	Water Temperature, Daily Minimum (Degrees Celsius)
    WR	Wind Run (miles/day)
    WY	Water Temperature, Daily Average (Degrees Celsius)
    WZ	Water Temperature, Daily Average (Degrees F)
    YR	Saturation Percent, Total Dissolved Gas (Computed %)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from copy import deepcopy
import matplotlib.pyplot as plt

def replace_string_in_txt_file(file, original_str, new_str, output_path=None):
    """Open .txt, replace string, and save."""
    # Read in the file
    with open(file, 'r') as file_obj:
        file_data = file_obj.read()
    # Replace the target string
    file_data = file_data.replace(original_str, new_str)
    # Write the file out again
    if output_path is None:
        output_path = file
    with open(output_path, 'w') as file_obj:
        file_obj.write(file_data)

def read_hydromet(file, stn, cols=None, nan_values=["NO_RECORD", "MISSING"]):
    print(file)

    df = pd.read_csv(
        file, skiprows=3, sep=",", skipfooter=1, parse_dates=True, index_col=0, engine='python'
        )
    # Trim leading and trailing spaces from all cells in the DataFrame
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = [c.split()[-1] for c in df.columns]

    if cols is not None:
        df = df[cols]

    if any(df[i].dtypes != 'float64' for i in df.columns):
        if nan_values == ["NO_RECORD", "MISSING"]:
            df = df.replace(to_replace=["NO_RECORD", "MISSING"], value=[np.nan] * 2)
            df = df.apply(pd.to_numeric)
        elif nan_values == ["NO_RECORD"]:
            df = df.replace(to_replace=["NO_RECORD"], value=[np.nan])
            df = df.replace(to_replace=["MISSING"], value=[np.nan])
            df = df.apply(pd.to_numeric)
    return df

def show_hydromet_data_report(df, title="", plot=False):
    """Feed in df extracted by collect_hydromet_files_to_csv."""
    data_dict = {
        "Daily": df,
        "Monthly": df.resample('MS').mean(),
        "Annually": df.resample('YS').mean(),
    }

    if plot:
        for key, val in data_dict.items():
            val.plot(alpha=0.25, title=key + " " + title)
            plt.show()

    for key, val in data_dict.items():
        print(f"{key} {title}'s Correlations:")
        print(val.corr())

    print(f"{title}'s Data Quality:")
    print(df.isnull().sum())

def fill_nan_with_monthly_mean(df):
    df_monthly = df.resample("MS").mean()
    for col in df.columns:
        df_nan = df[col][df[col].isna()]
        for date in df_nan.index:
            month = date.replace(day=1)
            df.loc[date, col] = df_monthly.loc[month, col]
    show_hydromet_data_report(df)
    return df

def collect_hydromet_files_to_csv(file_name_list, col, csv_file_name, data_folder_path, output_path, date_range=None, nan_values=["NO_RECORD", "MISSING"], fill_na=False):
    if file_name_list[:-4] != ".txt":
        file_name_list = [f"{name}.txt" for name in file_name_list]

    df_combined = pd.DataFrame()
    for file_name in file_name_list:
        stn = file_name.split(".")[0]
        file_path = os.path.join(data_folder_path, file_name)
        if isinstance(col, list):
            df = read_hydromet(file_path, stn=stn, cols=col, nan_values=nan_values)
        else:
            df = read_hydromet(file_path, stn=stn, cols=[col], nan_values=nan_values)
        df_combined = pd.concat([df_combined, df], axis=1)

    if not isinstance(col, list):
        df_combined.columns = [name[:-4] for name in file_name_list]

    df_combined.dropna(how="all", inplace=True)

    # Customize section for Yakima River Basin
    if col == "FB" and "KAC.txt" in file_name_list:
        df_combined.loc[df_combined["KAC"] > 3000, "KAC"] = np.nan
    if col == ["MX", "MN", "MM"]:  # Convert degF to degC
        for c in col:
            df_combined[c] = (df_combined[c] - 32) * 5 / 9
        df_combined[df_combined["MN"] < -40] = np.nan
        df_combined[df_combined["MX"] > 40] = np.nan
        df_combined[df_combined["MM"] < -40] = np.nan
        df_combined[df_combined["MM"] > 40] = np.nan

    if {'CLE', 'KAC', 'KEE'}.issubset(df_combined.columns):
        if col in {"AF", "QU", "QD"}:
            df_combined["Upper"] = df_combined[["CLE", "KAC", "KEE"]].sum(axis=1)
        elif col == "PP":
            df_combined["Upper"] = df_combined[["CLE", "KAC", "KEE"]].mean(axis=1)
        else:
            raise ValueError("Given col is not defined.")
        df_combined = df_combined[["BUM", "RIM", "Upper"]]


    df_combined = df_combined.reindex(pd.date_range(df_combined.index[0], df_combined.index[-1], freq='D')).fillna(np.nan)
    if date_range is not None:
        df_combined = df_combined.loc[date_range[0]:date_range[1]]

    show_hydromet_data_report(df_combined)
    if fill_na:
        df_combined = fill_nan_with_monthly_mean(df_combined)

    df_combined.to_csv(os.path.join(output_path, csv_file_name))
    return df_combined

def add_indicator(x, y, ax, indicators=["r"]):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    mu_y, mu_x = np.nanmean(y), np.nanmean(x)
    sig_y, sig_x = np.nanstd(y), np.nanstd(x)

    indicators_dict = {
        "r": np.corrcoef(x, y)[0, 1],
        "r2": np.corrcoef(x, y)[0, 1] ** 2,
        "rmse": np.sqrt(np.nanmean((x - y) ** 2)),
        "NSE": 1 - np.nansum((x - y) ** 2) / np.nansum((x - mu_x) ** 2),
        "CP": 1 - np.nansum((x[1:] - y[1:]) ** 2) / np.nansum((x[1:] - x[:-1]) ** 2),
        "RSR": np.sqrt(np.nanmean((x - y) ** 2)) / sig_x,
        "KGE": 1 - np.sqrt((np.corrcoef(x, y)[0, 1] - 1) ** 2 + (sig_y / sig_x - 1) ** 2 + (mu_y / mu_x - 1) ** 2),
    }

    wanted_indicators = {key: indicators_dict[key] for key in indicators}
    annotation_text = "\n".join([f"{key}: {value:.2f}" for key, value in wanted_indicators.items()])

    slope, intercept, _, _, _ = stats.linregress(x, y)
    ax.plot(x, slope * x + intercept, 'r', label=f'y={slope:.2f}x+{intercept:.2f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, fontsize=9, bbox=props)
    return ax, wanted_indicators

path_raw = r"C:\Users\cl2769\Documents\GitHub\YakimaRiverBasin\data\hydromet_raw"
path_revised = r"C:\Users\cl2769\Documents\GitHub\YakimaRiverBasin\data\hydromet_revised"
path_csv = r"C:\Users\cl2769\Documents\GitHub\YakimaRiverBasin\data\hydromet"

stn_reservoirs = ['BUM', 'CLE', 'KAC', 'KEE', 'RIM']
stn_gauges = ["NACW", "YUMW", "UMTW", 'PARW']
stn_canals  = ['KTCW', 'ROZW', 'RSCW', 'SNCW', 'TIEW']
stn_C1_other_div = ["CTSZ","CACW","CADW","WESW","ETCW"]
stn_G_other_div = ["SEXW", "MOXW", "HUBW", "UNGW","NCOW","WAPW","CODW","GLEW","CYDW","SOUW","NSCW","FRUW","OLDW"]
file_list = stn_reservoirs + stn_gauges + stn_canals + stn_C1_other_div + stn_G_other_div

#%% Replace NO RECORD to NO_RECORD in txt.
"""
for f in file_list:
    replace_string_in_txt_file(
        file=os.path.join(path_raw, f+".txt"),
        original_str="NO RECORD",
        new_str="NO_RECORD",
        output_path=os.path.join(path_revised, f+".txt"))
"""
#%%
Start = "1978-1-1"
End   = "2023-12-31"

df_gauges_regulated_flow_cfs = collect_hydromet_files_to_csv(
    file_name_list=stn_gauges, col="QD",
    csv_file_name = "hydromet_gauges_regulated_flow_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_gauges_unregulated_flow_cfs = collect_hydromet_files_to_csv(
    file_name_list=['NACW', 'YUMW', 'PARW'], col="QU",
    csv_file_name = "hydromet_gauges_unregulated_flow_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_canal_diversion_cfs = collect_hydromet_files_to_csv(
    file_name_list=stn_canals, col="QJ",
    csv_file_name = "hydromet_canal_diversion_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_reservoir_inflow_cfs = collect_hydromet_files_to_csv(
    file_name_list=stn_reservoirs, col="QU",
    csv_file_name = "hydromet_reservoir_inflow_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_reservoir_storage_acft = collect_hydromet_files_to_csv(
    file_name_list=stn_reservoirs, col="AF",
    csv_file_name = "hydromet_reservoir_storage_acft.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_reservoir_release_cfs = collect_hydromet_files_to_csv(
    file_name_list=stn_reservoirs, col="QD",
    csv_file_name = "hydromet_reservoir_release_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_reservoir_unregulated_flow_cfs = collect_hydromet_files_to_csv(
    file_name_list=stn_reservoirs, col="QU",
    csv_file_name = "hydromet_reservoir_unregulated_flow_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_reservoir_prec_in = collect_hydromet_files_to_csv(
    file_name_list=stn_reservoirs, col="PP",
    csv_file_name = "hydromet_reservoir_prec_in.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range = [Start,End],
    fill_na=False
    )

df_C1_other_div = collect_hydromet_files_to_csv(
    file_name_list=stn_C1_other_div,
    col="QJ",
    csv_file_name = "yrb_C1_other_div_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range=[Start, End],
    fill_na=False)

df_G_other_div = collect_hydromet_files_to_csv(
    file_name_list=stn_G_other_div,
    col="QJ",
    csv_file_name = "yrb_G_other_div_cfs.csv",
    data_folder_path = path_revised,
    output_path=path_csv,
    date_range=[Start, End],
    fill_na=False)

#%%
df_C1_other_div.resample("MS").mean().plot()
plt.show()
df_G_other_div.resample("MS").mean().plot()
plt.show()
df_canal_diversion_cfs.resample("MS").mean().plot()
plt.show()
#%%
# Fill na (not sure why)
# mask = df_gauges_regulated_flow_cfs.index.month == 2
# a = df_gauges_regulated_flow_cfs[mask][df_gauges_regulated_flow_cfs[mask].isna().any(axis=1)]["UMTW"]
# df_gauges_regulated_flow_cfs.loc[a.index, "UMTW"] = df_gauges_regulated_flow_cfs[mask][df_gauges_regulated_flow_cfs[mask].isna().any(axis=1)]["YUMW"]*1.79 + 148.06

# mask = df_gauges_regulated_flow_cfs.index.month == 7
# a = df_gauges_regulated_flow_cfs[mask][df_gauges_regulated_flow_cfs[mask].isna().any(axis=1)]["UMTW"]
# df_gauges_regulated_flow_cfs.loc[a.index, "UMTW"] = df_gauges_regulated_flow_cfs[mask][df_gauges_regulated_flow_cfs[mask].isna().any(axis=1)]["YUMW"]*0.83 + 833.01

# df_gauges_regulated_flow_cfs.to_csv(os.path.join(path_csv, "hydromet_gauges_regulated_flow_cfs.csv"))
#%%

