import flask
#import predict_sol

from flask import Flask, render_template, jsonify

#import warnings
#warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = flask.Flask(__name__, template_folder='templates')

# Load the trained model
solubility_model = load_model("output/SolubilityModel6.h5")

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
### Tili tili.. trali vali...***************************..Tratan tan tan..**********8***
#####****************************** Algorithm ****************************************

@app.route("/", methods=['GET', 'POST'])
def main():
    
    solute_tt=pd.DataFrame()
    
    if flask.request.method == 'GET':
        return(flask.render_template('index.html', listel=elements))
    
    if flask.request.method == 'POST':
        elnums= flask.request.form.getlist("elnum")
        
        outmolf=""
        
        for i in range(len(elements)):
            solute_tt.loc[0, elements[i]]=float(elnums[i])
            
            outmolf=outmolf+elements[i]+":"+str(elnums[i])+" "
    
        Mw_tot = flask.request.form['Mw_tot']
#        print("entered Mw: ", Mw_tot)
        solute_tt.loc[0, 'Mw_tot']=float(Mw_tot)
        
        outMw=int(Mw_tot)

# calculate Mw and % elements in input df
        solute_tt = get_Mw(solute_tt)
        solute_tt = calculate_elements_perc(solute_tt, 0)

# calculate Mw and % elements in solvents df
        solvents1=solvents_info.copy()
        solvents1 = get_Mw(solvents1)
        solvents1=calculate_elements_perc(solvents1, 2)
        solvents1=solvents1.rename(columns={"Mw_unit":"Mw_solvent"})

# stretch the input data and rename
        for i in range(1,len(solvents1)):
            solute_tt.loc[i, :]=solute_tt.iloc[0,:]

        solute_tt=solute_tt.rename(columns={"Mw_unit":"Mw_solute",
            "C":"C_solute", "H":"H_solute",
            "O":"O_solute", "N":"N_solute",
            "Cl":"Cl_solute", "Cl":"Cl_solute",
            "Br":"Br_solute", "S":"S_solute"})

# merge with the solvents data into one
        solute_tt_solvents=pd.concat([solvents1, solute_tt], axis=1)
        
# prepare the model df: drop, log transform and scale
        solute_tt_model=solute_tt_solvents.drop(['total_els', 'Solvent', 'Chem_structure'], axis=1)

# Log Transform the molecular weights
        solute_tt_model1 = solute_tt_model.copy()
            
        for i in ['Mw_solvent', 'Mw_solute', 'Mw_tot']:
            solute_tt_model1[i+'_log'] = np.log1p(solute_tt_model1[i]) 
            solute_tt_model1.drop(i, axis=1, inplace=True)
            
        solute_tt_model1.drop(['Br_solute', 'S_solute'], axis=1, inplace=True)

        X = X_data
        y = y_data['solubility_category']
# Scale the X data
        X_scaler = MinMaxScaler().fit(X)
# Label-encode y data
        label_encoder = LabelEncoder()
        label_encoder.fit(y)

        X_tt=solute_tt_model1
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

# Separate lists of diff. solvents solubility categories:
        soluble_list = soluble_df['Solvent'].to_list()
        non_soluble_list = non_soluble_df['Solvent'].to_list()
        theta_list = theta_df['Solvent'].to_list()
        lengths1=len(soluble_list)
        lengths2=len(non_soluble_list)
        lengths3=len(theta_list)

        print(f"length of solubles {lengths1} non-solubles {lengths2} thetas {lengths3}")
        
        return flask.render_template('estimated.html',list1=soluble_list,list2=non_soluble_list,list3=theta_list, lengths1=lengths1, lengths2=lengths2, lengths3=lengths3, showres=outmolf, Molw=outMw)

#####****************************** End of Algorithm ****************************************
@app.route("/read_polymers")
def read_polymers():

    polymer_info = pd.read_csv("data/polymer_info.csv")
    polymers=polymer_info.Polymer_name.tolist()
    structures=polymer_info.Monomer_chem_structure.tolist()
    Mwtot=polymer_info.tot_Mw.tolist()

    output = []
    for i in range(len(polymers)): 
        output.append({"Name" : polymers[i]})
        output.append({"Structure" : structures[i]})
        output.append({"Tot Mw" : Mwtot[i]})


    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)