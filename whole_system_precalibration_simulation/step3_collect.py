import os
import sys
import pathnavigator

# Set working directory
root_dir = "working directory to this repo"
pn = pathnavigator.create(root_dir)
pn.chdir()

import pandas as pd
from tqdm import tqdm
from SALib.analyze import delta

def convert_nested_np_to_native_types(obj):
    """
    Convert numpy types to native Python types.
    
    Parameters
    ==========
    obj : object
        The object to convert.
        
    Returns
    =======
    object
        The converted object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)  # Convert numpy integers to Python int
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)  # Convert numpy floats to Python float
    elif isinstance(obj, dict):
        return {k: convert_nested_np_to_native_types(v) for k, v in obj.items()}  # Recursively process dictionaries
    elif isinstance(obj, (list, tuple)):
        converted_items = [convert_nested_np_to_native_types(v) for v in obj]
        return tuple(converted_items) if isinstance(obj, tuple) else converted_items  # Recursively process lists and tuples
    else:
        return obj
    
def to_json(data, file_path, auto_convert=True):
    """
    Save data to a JSON file.
    
    Parameters
    ==========
    data : dict
        Data to be saved.
    file_path : str
        Path to the output JSON file.
    auto_convert : bool, optional
        If True, automatically convert numpy types to native Python types.
    """
    import json
    # Convert the data to a format that JSON can handle
    if auto_convert:
        converted_data = convert_nested_np_to_native_types(data)
    else:
        converted_data = data
    
    with open(file_path, "w") as file:
        json.dump(converted_data, file, indent=4)

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

def to_parquet(data, file_path):
    """
    Save data to a Parquet file.
    
    Parameters
    ==========
    data : pd.DataFrame or dict of pd.DataFrame
        Data to be saved. If a dictionary is provided, it should contain DataFrames.
    file_path : str
        Path to the output Parquet file.
    """
    if isinstance(data, dict):
        for key, df in data.items():
            df.to_parquet(f"{file_path}_{key}.parquet")
    else:
        data.to_parquet(file_path)

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
    

def filter_metric_df(df, filtering_criteria):
    # Filtering criteria
    for k, v in filtering_criteria.items():
        if k in df.columns:
            df = df[df[k] >= v]
        else:
            print(f"Warning: {k} not found in DataFrame columns. Skipping filtering for this key.")
    return df
#%%
def collect_metric_files(out_folder, exp_folder, disable=True):
    print("Collect json files ...")
    metrics_path = pn.experiments.get(exp_folder) / f"{out_folder}_metrics.parquet"
    if metrics_path.exists() is False:
        json_list = os.listdir(pn.outputs.get(out_folder) / "out")
        json_list = [f for f in json_list if f.endswith('.json')]

        df = []
        selected_indx = []
        errs = []
        seed = 1
        for json_file in tqdm(json_list, disable=disable):
            try:
                d = read_json(pn.outputs.get(out_folder) / "out" / json_file)
                d = d[str(seed)] # only have one seed here
                df.append(d)
                idx = int(json_file.split("_")[1].split(".")[0])
                selected_indx.append(idx)
            except:
                errs.append(json_file)
        df = pd.DataFrame(df)
        df.index = selected_indx
        to_parquet(df, metrics_path)
    else:
        print("Metric parquet file exists.")

#%%
def collect_ts_files_old(out_folder, exp_folder, filtering_criteria={"KGE_G(789)": 0, "KGE_Div": 0}, seed=1, hy_idx=None):
    print("Collect df_Y parquet files ...")
    metrics_path = pn.experiments.get(exp_folder) / f"{out_folder}_metrics.parquet"
    df = read_parquet(metrics_path)

    if hy_idx is not None:
        samples = read_parquet(pn.inputs.get('exp4_samples_118240_118245.parquet')).iloc[:,-2:]
        df = df.loc[samples["hydro_idx"]==hy_idx, :]

    df = filter_metric_df(df, filtering_criteria)

    if hy_idx is not None:
        ts_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_ts_selected_hyidx_{hy_idx}.parquet"
        met_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_selected_metrics_hyidx_{hy_idx}.parquet"
    else:
        ts_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_ts_selected.parquet"
        met_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_selected_metrics.parquet"

    df.to_parquet(met_selected_path)

    if ts_selected_path.exists() is False:
        selected_indx = list(df.index)
        parquet_list = [f'Y_{i}_{seed}.parquet'for i in selected_indx]
        df_Y_all = []
        for parquet in tqdm(parquet_list):
            df_Y = read_parquet(pn.outputs.get(out_folder) / 'out_parquet' / parquet)
            df_Y_all.append(df_Y)
        df_Y_all = pd.concat(df_Y_all, axis=0)
        df_Y_all.to_parquet(ts_selected_path)
    else:
        print("df_Y_all parquet file exists.")

def collect_ts_files(out_folder, exp_folder, filtering_criteria={"KGE_G(789)": 0.75, "KGE_Div": 0.6}, seed=1, hy_idx=None, overwrite=False):
    print("Collect df_Y parquet files ...")
    metrics_path = pn.experiments.get(exp_folder) / f"{out_folder}_metrics.parquet"
    df = read_parquet(metrics_path)

    if hy_idx is not None:
        problem = read_json(pn.outputs.get(out_folder) / 'problem.json')
        mask = [hy_idx_[0]==hy_idx for hy_idx_ in problem["pairs"]]
        
        df = df.loc[mask, :]

    df = filter_metric_df(df, filtering_criteria)

    if hy_idx is not None:
        ts_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_ts_selected_hyidx_{hy_idx}.parquet"
        met_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_selected_metrics_hyidx_{hy_idx}.parquet"
    else:
        ts_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_ts_selected.parquet"
        met_selected_path = pn.experiments.get(exp_folder) / f"{out_folder}_selected_metrics.parquet"

    df.to_parquet(met_selected_path)

    if ts_selected_path.exists() is False or overwrite:
        selected_indx = list(df.index)
        parquet_list = [f'Y_{i}_{seed}.parquet'for i in selected_indx]
        df_Y_all = []
        for parquet in tqdm(parquet_list):
            df_Y = read_parquet(pn.outputs.get(out_folder) / 'out_parquet' / parquet)
            df_Y_all.append(df_Y)
        df_Y_all = pd.concat(df_Y_all, axis=0)
        df_Y_all.to_parquet(ts_selected_path)
    else:
        print("df_Y_all parquet file exists.")

def compute_Si_dict(out_folder, exp_folder, ylist=["KGE_G(789)", "KGE_Div"], filtering_criteria={"KGE_G(789)": 0, "KGE_Div": 0}):
    print("Delta moments ...")
    Si_dict_path = pn.experiments.get(exp_folder) / 'Si_dict.json'

    def cal_delta_moment_sensitivity_index(df, problem, samples, selected_indx=None):
        if selected_indx is None:
            selected_indx = list(df.index)
        Si_dict = {}
        for i in tqdm(df.columns, disable=disable):
            Si_dict[i] = delta.analyze(problem, samples[selected_indx,:], df[i].values, method="delta", num_resamples=20)
        return Si_dict

    if Si_dict_path.exists() is False:
        problem = read_json(pn.outputs.get(out_folder) / 'problem.json')
        try:
            samples = read_parquet(pn.outputs.get(out_folder) / 'samples.parquet').values
        except:
            samples = pd.read_csv(pn.outputs.get(out_folder) / 'samples.csv', header=None).values

        metrics_path = pn.experiments.get(exp_folder) / f"{out_folder}_metrics.parquet"
        df = read_parquet(metrics_path)
        df = filter_metric_df(df, filtering_criteria)
        Si_dict = cal_delta_moment_sensitivity_index(df[ylist], problem, samples)
        to_json(data=Si_dict, file_path=Si_dict_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python exp4_combined_collect.py [all|ensemble] [out_folder] [exp_folder]")
        sys.exit(1)

    action = sys.argv[1].lower()

    # Folder settings
    out_folder = sys.argv[2]

    exp_folder = sys.argv[3]
    pn.experiments.mkdir(exp_folder)
    disable = False

    # https://hess.copernicus.org/articles/23/4323/2019/
    # KGE values greater than −0.41 indicate that a model improves upon the mean flow
    # benchmark – even if the model's KGE value is negative.

    if action == "all":
        collect_metric_files(out_folder, exp_folder)
        collect_ts_files(out_folder, exp_folder, filtering_criteria={"KGE_G(789)": 0.78})
    elif action == "ensemble":
        collect_ts_files(out_folder, exp_folder, filtering_criteria={"KGE_G(789)": 0.78}, seed=1, hy_idx=71191, overwrite=True)
        collect_ts_files(out_folder, exp_folder, filtering_criteria={"KGE_G(789)": 0.78}, seed=1, hy_idx=None, overwrite=True)
    else:
        print("Invalid argument. Use 'metrics' to collect metric files or 'ts' to collect time series files.")
        sys.exit(1)         