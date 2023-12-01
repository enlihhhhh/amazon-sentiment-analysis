import pandas as pd
import numpy as np
import pickle
import json
import re
import matplotlib.pyplot as plt
import seaborn as sb
import os

# Function to write data to a JSON file
def write_to_json(data, filename):
    """
    Write data to a JSON file.

    Args:
        data: The data (dictionary, list, etc.) to be written to the JSON file.
        filename (str): The name of the JSON file to write.

    Returns:
        bool: True if the data was successfully written to the file, False otherwise.
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        return False

# Function to load data from a JSON file
def load_from_json(filename):
    """
    Load data from a JSON file.

    Args:
        filename (str): The name of the JSON file to read data from.

    Returns:
        dict or list: The loaded data from the JSON file, or an empty dictionary/list if the file doesn't exist.
    """
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"JSON file '{filename}' not found. Returning an empty dictionary.")
        return {}
    except Exception as e:
        print(f"Error loading data from JSON file: {e}")
        return {}
    
def extracting_df(dataframe):

    individual_dataframes = {}

    # Iterate over each column in the original DataFrame
    for column in dataframe.columns:
        # Extract data for the current column
        column_data = dataframe[column].apply(pd.Series)
        
        # Rename the columns to include the original column name as a prefix
        column_data.columns = [f"{column}_{col}" for col in column_data.columns]
        
        # Remove rows where 'Name' is not found
        column_data = column_data[column_data[f"{column}_Name"] != 'Product title not found']
        
        # Store the resulting DataFrame in the dictionary
        individual_dataframes[column] = column_data
        
    return individual_dataframes


def get_information():
    directory = os.getcwd()
    directory = directory + '/Processing_Files'
    all_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_data[os.path.splitext(filename)[0]] = data
        else:
            continue
    return all_data


def standardising_length(dictionary):
    for name, subcat in dictionary.items():
        # Find the maximum length of items for all keys in the subcategory
        if isinstance(subcat, list):
            
            max_length = len(subcat)
        else:    
            
            max_length = max(len(items) for items in subcat.values())

            # Update the length of items for each key in the subcategory
            for key, items in subcat.items():
                if isinstance(items, list):
                    if len(items) < max_length:
                        items += [{}] * (max_length - len(items))
                    dictionary[name][key] = items

    return dictionary

def postprocessingbrandmodel(final_dict):
    
    all_cat_info = {} # Storing dictionaries of info for all entries in a specific category's column
    dataframes_dict = {}
    all_dfs = {}

    for cat, subcats in final_dict.items():
        if (isinstance(subcats, list)): # If there is only one category
            data_dict = {}  # Stores key-value pairs of info
            for item in subcats:
                if (':' in item):
                    name, var = cat.split(':', 1)
                    temp = name.strip()
                    res = f'{cat}_' + temp
                    data_dict[res] = [var.strip()]  # Use strip to remove leading/trailing spaces
                else:
                     data_dict[f'{cat}'] = [None]
            all_cat_info.update({cat : data_dict})
        else: 
            temp_df = pd.DataFrame(subcats)
            temp_df2 = extracting_df(temp_df)
            for subcat_name, subcat_df in temp_df2.items():
                all_cat_info[subcat_name] = {}  # Initialize an empty dictionary for each subcategory
                brand = []
                model = []
                temp = subcat_df[f'{subcat_name}_Information']
                for entry in temp:
                    data_dict = {}  # Move initialization here to update for each entry
                    data_dict[f'{subcat_name}_Ratings'] = subcat_df[f'{subcat_name}_Ratings']
                    data_dict[f'{subcat_name}_Total Number of Ratings'] = subcat_df[f'{subcat_name}_Total Number of Ratings']
                    data_dict[f'{subcat_name}_Price'] = subcat_df[f'{subcat_name}_Price']
                    data_dict[f'{subcat_name}_Information'] = subcat_df[f'{subcat_name}_Information']
                    data_dict[f'{subcat_name}_URL'] = subcat_df[f'{subcat_name}_URL']
                    all_cat_info[subcat_name].update(data_dict)  # Update the data_dict for each entry
    
    var = {}
    for key, items in all_cat_info.items():
        for name, info in items.items():
            df = pd.DataFrame(columns = [name])
            df[name] = info
            var.update({name : df})
    
    return all_cat_info

def all_dfs(dictionary):
    all_df = {}
    for catname, subcatnames in dictionary.items():
        if (isinstance(subcatnames, list)):
            temp_df = pd.DataFrame(subcatnames)
            all_df[catname] = temp_df
        else:
            for subcatname, dic in subcatnames.items():
                temp_df = pd.DataFrame(dictionary[catname][subcatname])
                all_df[subcatname] = temp_df
    return all_df

def brand_modeldfs(dictionary):
    brand_modeldf = {}
    for catname, items in dictionary.items():
        total = []
        for identifier, names in items.items():
            temp = pd.DataFrame({identifier:names})
            total.append(temp)
        brand_modeldf[identifier] = total
    return brand_modeldf

def results(dictionary):
    final_dfs = {}
    for key in dictionary.keys():
        info = dictionary[key][3]
        brand = pd.DataFrame(info[info.columns[0]].apply(
            lambda x: x[0].split(':')[1].strip() if isinstance(x, list) and len(x) > 0 and len(x[0].split(':')) > 1 else 'NaN'
        ))
        brand = brand.rename(columns = {f'{key.split("_")[0]}_Information' : f'{key.split("_")[0]}_Brand'})
        model = pd.DataFrame(info[info.columns[0]].apply(
            lambda x: x[1].split(':')[1].strip() if isinstance(x, list) and len(x) > 1 and len(x[1].split(':')) > 1 else 'NaN'
        ))
        model = model.rename(columns = {f'{key.split("_")[0]}_Information' : f'{key.split("_")[0]}_Model'})
        
        ratings = dictionary[key][0]
        total_ratings = dictionary[key][1]
        price = dictionary[key][2]
        
        dfs = [brand,model,ratings,total_ratings,price]
        
        result = pd.concat(dfs, axis=1)

        result.reset_index(drop=True, inplace=True)
        
        final_dfs.update({key.split("_")[0]:result})
        
    return final_dfs

def extraction():
    all_data = get_information() # Parsing JSON data
    prob_dict = standardising_length(all_data) # Ensuring compaitability with Dataframe Data Structure
    res1 = all_dfs(prob_dict) # Obtaining all DataFrames
    raw_data = postprocessingbrandmodel(prob_dict) # Obtaining dfs for Relevant Categories
    output = brand_modeldfs(raw_data) # Extracting dfs for Relevant Categories
    res = results(output)
    return output
    