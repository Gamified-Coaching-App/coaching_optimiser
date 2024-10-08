import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import os
import json
import h5py
from sklearn.model_selection import train_test_split

"""
RunningDataset class manages operations related to loading and processing data.
"""
class RunningDataset:
    """__init__ initialises the dataset with predetermined parameters."""
    def __init__(self):
        self.filename = '../../../data/day_approach_maskedID_timeseries.csv'
        self.WINDOW_DAYS = 7
        self.base_metrics = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 
                             'strength training', 'hours alternative', 'perceived exertion', 
                             'perceived trainingSuccess', 'perceived recovery']
        self.identifiers = ['Athlete ID', 'Date']
        self.class_name = 'injury'
        self.fixed_columns = ['Athlete ID', 'injury', 'Date']
        self.data_types_metrics = [float] * len(self.base_metrics)
        self.data_types_fixed_columns = [int] * len(self.identifiers)
        self.data = pd.read_csv(self.filename)
        self.data.columns = [f"{col}.0" if i < 10 else col for i, col in enumerate(self.data.columns)]
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()

    """
    function to reorder the columns in the dataset before stacking
    """
    def reorder_columns(self, data, days):
        n = 10 * days
        new_order = []
        for i in range(10):
            new_order.extend([(i + 10 * j) % n for j in range(days)])
        data = data.iloc[:, new_order]
        return data

    """
    function converts the dataset to long format required for normalising
    """
    def long_form(self, df):
        df_long = pd.wide_to_long(df, stubnames=self.base_metrics, i=self.fixed_columns, j='Offset', sep='.')
        df_long.reset_index(inplace=True)
        df_long[self.identifiers[1]] = df_long[self.identifiers[1]] - (self.WINDOW_DAYS - df_long['Offset'])
        df_long.drop(columns='Offset', inplace=True)
        df_long.drop_duplicates(subset=self.identifiers, keep='first', inplace=True)
        return df_long
    
    """
    function applies z-score normalisation grouped by athlete to each metric in the dataframe
    """
    def z_score_normalization(self, df):
        for metric in self.base_metrics:
            df[metric] = df.groupby([self.identifiers[0]])[metric].transform(
                lambda x: self.standard_scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
            )
        return df.reset_index(drop=True)
    
    """
    function saves mean and stdv of each metric per athlete to a json file for later use
    """
    def save_stdv_mean_per_athlete(self, df):
        mean_stdv_dict = {}
        for athlete in df[self.identifiers[0]].unique():
            mean_stdv_dict[int(athlete)] = {}
            athlete_df = df[df[self.identifiers[0]] == athlete]
            for metric in self.base_metrics:
                mean = athlete_df[metric].mean()
                stdv = athlete_df[metric].std()
                mean_stdv_dict[athlete][metric] = {'mean': mean, 'stdv': stdv}
        os.makedirs('../data', exist_ok=True)
        with open('../data/mean_stdv.json', 'w') as f:
            json.dump(mean_stdv_dict, f)
    
    """
    function applies min-max normalisation to each metric in the dataframe
    """
    def min_max_normalization(self, df):
        for metric in self.base_metrics:
            df[metric] = self.min_max_scaler.fit_transform(df[metric].values.reshape(-1, 1)).flatten()
        return df.reset_index(drop=True)
    
    """
    function converts the dataset from long format to wide format after normalisation
    """
    def wide_form(self, df_long, days):
        df_long['Date'] = df_long['Date'].astype(int)
        df_long['Athlete ID'] = df_long['Athlete ID'].astype(int)
        df_long['injury'] = df_long['injury'].astype(int)
        df_long = df_long.groupby(self.identifiers[0], as_index=False).apply(self.fill_missing_dates).reset_index(drop=True)
        df_long.sort_values(by=self.identifiers, inplace=True)
        athlete_info = df_long[self.fixed_columns]
        df_rolled = pd.DataFrame(index=athlete_info.index).join(athlete_info)
        for day in range(days):
            shifted = df_long.groupby(self.identifiers[0])[self.base_metrics].shift(day).add_suffix(f'.{days - 1 - day}')
            df_rolled = df_rolled.join(shifted)
        metric_columns = [f'{metric}.{day}' for day in range(days) for metric in self.base_metrics]
        df_rolled = df_rolled[metric_columns + self.fixed_columns]
        df_rolled.dropna(inplace=True)
        df_rolled.reset_index(drop=True, inplace=True)
        df_rolled.sort_values(by=self.identifiers, inplace=True)
        df_rolled[self.identifiers[1]] = df_rolled[self.identifiers[1]] + 1
        df_rolled = df_rolled.sort_values(by=self.identifiers).reset_index(drop=True)
        df_rolled = df_rolled.astype(dict(zip(df_rolled.columns, self.data_types_metrics * days + self.data_types_fixed_columns)))
        return df_rolled
    
    """
    function fills in missing dates for each athlete to ensure continuity in the data set - 
    needed if the athlete has not recorded data for a specific date
    """
    def fill_missing_dates(self, group):
        min_date = group[self.identifiers[1]].min()
        max_date = group[self.identifiers[1]].max()
        int_range = range(min_date, max_date + 1)
        group = group.set_index(self.identifiers[1]).reindex(int_range).rename_axis(self.identifiers[1]).reset_index()
        group[self.identifiers[0]] = group[self.identifiers[0]].ffill()
        return group
    
    """
    function normalises the dataset using the specified method and saves normalisation values to a json file for later use during model training
    """
    def normalise(self, dataset, method = 'sliding-window', min=0, days=14):
        if method == 'sliding-window':
            normalized_data = pd.DataFrame(index=dataset.index, columns=dataset.columns, data=0.0)
            for index, row in dataset.iterrows():
                for start in range(0, 70, 7):
                    scaler = MinMaxScaler(feature_range=(min, 1))
                    end = start + 7
                    block = row[start:end]
                    scaled_block = scaler.fit_transform(block.values.reshape(-1, 1)).flatten()
                    normalized_data.iloc[index, start:end] = scaled_block
            normalized_data.iloc[:, -3:] = dataset.iloc[:, -3:]
            return normalized_data
        
        elif method == 'athlete-history':
            long = self.long_form(dataset)

            self.save_stdv_mean_per_athlete(long)

            self.save_min_max_values(long, '../../../model_export/', 'min_max_values.json')
            self.save_min_max_values(long, '../data/', 'min_max_values.json')

            standardised = self.z_score_normalization(long.copy())
            self.save_min_max_values(standardised, '../data/', 'standardised_min_max_values.json')            
            long = self.min_max_normalization(long)
            wide = self.wide_form(long, days)
            return wide
        else:
            raise ValueError("Invalid normalization method")

    """
    function saves min-max values for later use during model training - needed, e.g., for enforcing constraints based on absolute values
    """
    def save_min_max_values(self, df, export_path, filename):
        df = df.drop(columns=self.fixed_columns)
        min_max_dict = {
        col: {
            'min': df[col].min().item(),  
            'max': df[col].max().item()  
        }
        for col in df.columns
        }
        os.makedirs(export_path, exist_ok=True)
        with open(export_path + filename, 'w') as f:
            json.dump(min_max_dict, f)
        print(f"File {filename} saved to {export_path+filename}")
    
    """
    function converts data from 2D shape (no_samples, no_variables * no_time_steps), i.e., (N, 140) to 
    3D shape (no_samples, no_timesteps, no_variables), i.e., (N, 14, 10)
    """
    def stack(self, df, days):
        df = self.reorder_columns(df, days)
        num_variables = 10  
        time_steps_per_variable = days 
        num_samples = len(df)
        reshaped_data = np.zeros((num_samples, time_steps_per_variable, num_variables))
        
        for index, row in df.iterrows():
            temp_row = np.zeros((num_variables, time_steps_per_variable))
            for var_index in range(num_variables):
                start_col = var_index * time_steps_per_variable
                end_col = start_col + time_steps_per_variable
                temp_row[var_index, :] = row.iloc[start_col:end_col].values
            
            temp_row = temp_row.T
            reshaped_data[index, :, :] = temp_row 
        return reshaped_data

    """
    function normalises, stacks and masks data for training
    """
    def preprocess(self, days=56, train_ratio=0.85):
        print("Preprocessing data...")
        normalisation_method = 'athlete-history'
        norm_min=0
        self.data_normalised = self.normalise(self.data, method=normalisation_method, min=norm_min, days=days)
        X = self.stack(self.data_normalised.drop(columns=self.fixed_columns), days)

        os.makedirs('../data', exist_ok=True)
        with h5py.File('../data/processed_data.h5', 'w') as hf:
            hf.create_dataset('X', data=X.astype(np.float32))
        
        print("Shapes of the datasets: X:", X.shape)
        print("Data saved to ../data/processed_data.h5")

if __name__ == "__main__":
    dataset = RunningDataset()
    dataset.preprocess()