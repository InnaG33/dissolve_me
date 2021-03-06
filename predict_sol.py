import pandas as pd
#import scipy.stats as st
import numpy as np
#import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.preprocessing import FunctionTransformer  
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

def estimator():

    # Load the trained model
    solubility_model = load_model("output/SolubilityModel3.h5")

# read the model X and solvents data 
    X_data=pd.read_csv("output/X_model.csv")
    y_data=pd.read_csv("output/y_model.csv")
    del X_data['Unnamed: 0']
    del y_data['Unnamed: 0']

    solvents_info=pd.read_csv("data/solvents_info.csv")

# get element dictionary set  
    element_info = pd.read_csv("data/element_info.csv")
    elements=element_info['Element'].to_list()
    element_dict ={}

    for i in range(len(element_info)):
        element_dict.update({element_info['Element'][i]:element_info['Mw'][i]})

###****************************** Functions ***************************************
# Get Mw from all elemens in a raw
    def get_Mw(df):

        df['Mw_unit']=0
        Mw=0
        for element in elements:
            Mw += df[element]*element_dict[element]
        df['Mw_unit'] = Mw
    
        return(df)

# Calculate each element percenrage in a raw
    def calculate_elements_perc(df, x):
    
        for i in range(len(df)):
            df.loc[i, 'total_els']=df.iloc[i, x:x+7].sum()
        df.iloc[:, x:x+7]=df.iloc[:, x:x+7].div(df.loc[:,'total_els'], axis=0)*100 

        return(df)

###***************************** End of Functions ***************************************

#Input data
    solute_tt=pd.DataFrame()

    for element in element_dict:
        solute_tt.loc[0, element]=int(input(f"input # {element} atoms in polymer unit formula: "))

    solute_tt.loc[0, 'Mw_tot']=float(input("input total (approximate) polymer Mw: "))

# calculate Mw and % elements in input df
    solute_tt = get_Mw(solute_tt)
    solute_tt = calculate_elements_perc(solute_tt, 0)

# calculate Mw and % elements in solvents df
    solvents1=solvents_info.copy()
    solvents1 = get_Mw(solvents1)
    solvents1=calculate_elements_perc(solvents1, 2)

# stretch the input data and rename
    for i in range(1,len(solvents1)):
        solute_tt.loc[i, :]=solute_tt.iloc[0,:]

    solute_tt=solute_tt.rename(columns={"Mw_unit":"Mw_solute",
        "C":"C_solute", "H":"H_solute",
        "O":"O_solute", "N":"N_solute",
        "Cl":"Cl_solute", "Cl":"Cl_solute",
        "Br":"Br_solute", "S":"S_solute"})

# merge two dfs into one
    solute_tt_solvents=pd.concat([solvents1, solute_tt], axis=1)

    solute_tt_solvents=solute_tt_solvents.rename(columns={"C":"C_solvent",
        "H": "H_solvent",
        "O": "O_solvent",
        "N": "N_solvent",
        "Cl": "Cl_solvent",
        "Br": "Br_solvent",
        "S": "S_solvent",
        "Mw_unit":"Mw_solvent"})

# prepare the model df: drop, log transform and scale
    solute_tt_model=solute_tt_solvents.drop(['H_solute', 'total_els', 'Solvent', 'Chem_structure'], axis=1)
    log_transformer = FunctionTransformer(np.log1p)

# Log Transform the molecular weights
    for i in ['Mw_solvent', 'Mw_solute', 'Mw_tot']:
        solute_tt_model[i+'_logtrnsfmed'] = log_transformer.fit_transform(solute_tt_model[[i]]) 
    solute_tt_model = solute_tt_model.drop(['Mw_solvent', 'Mw_solute', 'Mw_tot', 'Br_solute', 'S_solute'], axis=1)


    X = X_data
    y = y_data['solubility_category']
# Scale the X data
    X_scaler = MinMaxScaler().fit(X)
# Label-encode y data
    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    X_tt=solute_tt_model
    X_tt_scaled = X_scaler.transform(X_tt)

# Make predictions for requested polymer unit
    predictions=np.argmax(solubility_model.predict(X_tt_scaled), axis=-1)

# Inverse transfrom encoded output
    encoded_results = label_encoder.inverse_transform(predictions)
    pred_model_df = pd.DataFrame({"Predicted_solubility_category": encoded_results})

# Join predicted with solvents info to get solvents names
    pred_results=pd.concat([solute_tt_solvents, pred_model_df], axis=1)
    pred_solvents=pred_results[['Solvent', 'Predicted_solubility_category']]

    soluble_df = pred_solvents.loc[(pred_solvents['Predicted_solubility_category']=='soluble')]
    non_soluble_df = pred_solvents.loc[(pred_solvents['Predicted_solubility_category']=='non-soluble')]
    theta_df = pred_solvents.loc[(pred_solvents['Predicted_solubility_category']=='theta')]

    results={}
    soluble_list = soluble_df['Solvent'].to_list()
    non_soluble_list = non_soluble_df['Solvent'].to_list()
    theta_list = theta_df['Solvent'].to_list()

    results = {
        "soluble": soluble_list,
        "non_soluble": non_soluble_list,
        "theta": theta_list
        } 

    print(f'for requested structure there are {len(soluble_list)} soluble, {len(non_soluble_list)} non-soluble, and {len(theta_list)} theta solvents')

    return results