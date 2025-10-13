# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 20:23:28 2025

@author: detec
"""

# %% 
'''Import packages'''
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, ctx, no_update

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.cm as cm
import ast
import sys

initialLink = "https://github.com/Curvedrain/Illinois-Hospital-WWTPs/raw/refs/heads/main/PRIISM_List_Nomination_06SEP24v2.xlsx"
resultsLink = "https://github.com/Curvedrain/Illinois-Hospital-WWTPs/raw/refs/heads/main/Outputted%20Link%20Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx"

priismdata = pd.read_excel(initialLink)
facilitylinkdata = pd.read_excel(resultsLink, 'Facility Link Data')
supplylinkdata = pd.read_excel(resultsLink, 'Supply Link Data', keep_default_na=False)


'''Set up/Reformatting'''        
def columnconvert(df, columnname):
    df[columnname] = df[columnname].apply(ast.literal_eval)
    
#Facilities come before supplies
priismdata=priismdata.sort_values(by=['Supply', 'Node ID']).reset_index(drop=True)

priismdatastrings = priismdata
priismdatastrings["Supply"] = priismdatastrings["Supply"].astype(str)
priismdatastrings[["Supply"]] = priismdatastrings[['Supply']].replace(['0','1'], ['Facility', 'Supply'])

priismdatastrings = priismdatastrings.rename(columns={'x': 'Longitude', 'y': 'Latitude', 'Supply': 'NodeType'})
priismdatastrings['Node ID'] = priismdatastrings['Node ID'].astype(str)

Facilities = priismdata.loc[priismdata["Supply"] == "Facility"]
Supplies = priismdata.loc[priismdata["Supply"] == "Supply"]


# Incorporate data
df1 = facilitylinkdata.drop(['Unnamed: 0'], axis=1)
df2 = supplylinkdata.drop(['Unnamed: 0'], axis=1).replace(['N/A'], ['[\'N/A\']'])


FacilityLinkInfo = df1.copy(deep=True)
SupplyLinkInfo = df2.copy(deep=True)
SupplyLinkInfo.fillna("None")


columnconvert(FacilityLinkInfo, "Facility Lat-Longs")
columnconvert(FacilityLinkInfo, "Linked Supply List")
columnconvert(FacilityLinkInfo, "Supply Lat-Longs")
columnconvert(FacilityLinkInfo, "Probability: Square Method")

columnconvert(SupplyLinkInfo, "Supply Lat-Longs")
columnconvert(SupplyLinkInfo, "Linked Facility List")
columnconvert(SupplyLinkInfo, "Facility Lat-Longs")
columnconvert(SupplyLinkInfo, "Probability: Square Method")


# =============================================================================
# FacilityLinkInfo['Facility Lats'] = FacilityLinkInfo["Facility Lat-Longs"].str[1]
# FacilityLinkInfo['Facility Longs'] = FacilityLinkInfo["Facility Lat-Longs"].str[0]
# 
# SupplyLinkInfo['Supply Lats'] = SupplyLinkInfo["Supply Lat-Longs"].str[1]
# SupplyLinkInfo['Supply Longs'] = SupplyLinkInfo["Supply Lat-Longs"].str[0]
# =============================================================================

NodeIDs = [str(x) + ' (' + str(y) + ')' for x, y in zip(priismdata['Facility Name'], priismdata['Node ID'])]
NodeIDs = ["All Nodes"] + NodeIDs

def createdefaultgraph(mapviewboolean, hiddenlinks = None):
    if mapviewboolean:
        CentralLat = 0.5 * ( min(priismdatastrings["Latitude"].tolist()) + max(priismdatastrings["Latitude"].tolist()))
        CentralLon = 0.5 * ( min(priismdatastrings["Longitude"].tolist()) + max(priismdatastrings["Longitude"].tolist()))
        fig=px.scatter_mapbox(priismdatastrings, lon = "Longitude", lat = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Proposed Links: Illinois Dataset",
                          width = 505, height = 700, zoom=5.6, center= {'lon': CentralLon, 'lat':CentralLat},
                          hover_name = 'Facility Name',
                          hover_data = {'Node ID': True,
                                        'Latitude': True,
                                        'Longitude': True,
                                        'NodeType': False
                       
                        }).update_layout(dragmode='pan')
        fig.update_layout(mapbox_style="open-street-map")
    else:
        fig=px.scatter(priismdatastrings, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Precise Search Graph",
                          width = 505, height = 700,
                          hover_name = 'Facility Name',
                          hover_data = {'Node ID': True,
                                        'Latitude': True,
                                        'Longitude': True,
                                        'NodeType': False
                       
                        }).update_layout(dragmode='pan')
    if mapviewboolean!=True:
        #For each facility
        for i in range (0, FacilityLinkInfo.shape[0]):
            #And each supply linked to said facility
            for j in FacilityLinkInfo.iloc[i]['Linked Supply List']:
                if hiddenlinks == None:
                    #Find color in Red Yellow Green Gradient, and convert to rgba
                    linecolor = cm.RdYlGn(
                        (float(FacilityLinkInfo.iloc[i]['Probability: Square Method'][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)].replace("\'", "").replace("%", ""))/100) 
                        )
                    linecolor = tuple(float(s) for s in str(linecolor).replace("np.float64", "").replace("(", "").replace(")","").split(","))
                elif hiddenlinks == "HideLinks":
                    linecolor = (0,0,0,0)
                
                #Plot the Link
                fig.add_trace(go.Scatter(
                    x = [Facilities.iloc[i]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                    y = [Facilities.iloc[i]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                    #Red Yellow Green Gradient
                    mode = "lines",
                    line_color = 'rgba' + str(linecolor),
                    showlegend = False
                    ))
    fig.update_layout(
        title = {'x':0.5,
                 'xanchor': 'center', 
                 'y':0.98},
        margin=dict(t=32)
        )
    return fig


'''Dash app'''

# Initialize the app with css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(external_stylesheets=external_stylesheets)
server = app.server

# App layout
app.layout = html.Div([
# =============================================================================
#     html.Div(className='row', children='Illinois Proposed Link Data'),
#     
#     html.Div(className='row', children = [
#         #Ouputted Data tables from excel sheet
#         dash_table.DataTable(data=df1.to_dict('records'), page_size=10),
#         dash_table.DataTable(data=df2.to_dict('records'), page_size=10),
#         html.Hr()
#     ]),
# =============================================================================
    html.Div(className = 'row', children = [
        #Interactive features
        
        #dropdown for node specification
        html.Div(className='five columns', children=[
            dcc.Dropdown(NodeIDs, 'All Nodes', id='node dropdown')
        ]),
        html.Div(className='six columns', children=[
            html.H3('Illinois Wastewater Node Search', style={'padding-top': '0px', 'margin-top':'0px', 'margin-bottom':'7px'})    
        ]), 
   ]),
    
   html.Div(className = 'row', children = [
       html.Div(className='five columns', children=[
          #Graph of proposed links
          dcc.Graph(figure={}, config={'scrollZoom': True, "modeBarButtonsToRemove": ['resetScale2d',  'select2d', 'lasso2d']}, id='controls-and-graph'),
          dcc.Checklist(
              id="mapviewcheckfull",
              options=[ {"label": "Map View", "value": True} ],
              value=[]
              )
          ]),
       html.Div(className='six columns', children=[
            html.H6(id='Data Table Name'),
            dash_table.DataTable(id="linkchart", page_size=10, style_cell={'text-align': 'center'}),
            dcc.Graph(figure={}, config={'scrollZoom': True, "modeBarButtonsToRemove": ['resetScale2d',  'select2d', 'lasso2d']}, id='zoomed-graph'),
            dcc.Checklist(
                id="mapviewcheckzoom",
                options=[ {"label": "Map View", "value": True} ],
                value=[]
                )
        ])
    ])
    
])

# %%
'''Interactive feature's callbacks'''

#Update the graph

@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='node dropdown', component_property='value'),
    Input(component_id='mapviewcheckfull', component_property='value')
)

def update_graph(node_chosen, mapviewboolean):
    probmethod_chosen="Probability: Square Method"
    if mapviewboolean:
        CentralLat = 0.5 * ( min(priismdatastrings["Latitude"].tolist()) + max(priismdatastrings["Latitude"].tolist()))
        CentralLon = 0.5 * ( min(priismdatastrings["Longitude"].tolist()) + max(priismdatastrings["Longitude"].tolist()))
        fig=px.scatter_mapbox(priismdatastrings, lon = "Longitude", lat = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Proposed Links: Illinois Dataset",
                          width = 505, height = 700, zoom=5.6, center= {'lon': CentralLon, 'lat':CentralLat},
                          hover_name = 'Facility Name',
                          hover_data = {'Node ID': True,
                                        'Latitude': True,
                                        'Longitude': True,
                                        'NodeType': False
                       
                        }).update_layout(dragmode='pan')
        fig.update_layout(mapbox_style="open-street-map")
    else:
        fig=px.scatter(priismdatastrings, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Proposed Links: Illinois Dataset",
                          width = 505, height = 700,
                          hover_name = 'Facility Name',
                          hover_data = {'Node ID': True,
                                        'Latitude': True,
                                        'Longitude': True,
                                        'NodeType': False
                       
                        }).update_layout(dragmode='pan')
    
    MostProbable = ["Most Probable Links", "Additional Links"]
    LegendGroups = []
    
    if node_chosen == "All Nodes": 

        #For each facility
        for i in range (0, FacilityLinkInfo.shape[0]):
            #And each supply linked to said facility
            for j in FacilityLinkInfo.iloc[i]['Linked Supply List']:
                
                #Find color in Red Yellow Green Gradient, and convert to rgba
                linecolor = cm.RdYlGn((float(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)].replace("\'", "").replace("%", ""))/100))
                linecolor = tuple(float(s) for s in str(linecolor).replace("np.float64", "").replace("(", "").replace(")","").split(","))
                
                if mapviewboolean:
                    #Plot the Link
                    fig.add_trace(go.Scattermapbox(
                        lon = [Facilities.iloc[i]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                        lat = [Facilities.iloc[i]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                        #Red Yellow Green Gradient
                        mode = "lines",
                        line_color = 'rgba' + str(linecolor),
                        legendgroup = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                        legendrank = 1000+int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                        name = MostProbable[int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))],
                        showlegend = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])) not in LegendGroups
                        ))
                    fig.update_traces(hoverinfo='skip')
                    LegendGroups = LegendGroups + [int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))]
                else:
                    #Plot the Link
                    fig.add_trace(go.Scatter(
                        x = [Facilities.iloc[i]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                        y = [Facilities.iloc[i]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                        #Red Yellow Green Gradient
                        mode = "lines",
                        line_color = 'rgba' + str(linecolor),
                        legendgroup = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                        legendrank = 1000+int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                        name = MostProbable[int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))],
                        showlegend = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])) not in LegendGroups
                        ))
                    LegendGroups = LegendGroups + [int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))]
    elif int(node_chosen.split('(')[1].replace(')', "")) in Facilities["Node ID"].to_list():
        node_chosen = int(node_chosen.split('(')[1].replace(')', "")) 
        i = int(Facilities[Facilities['Node ID']==node_chosen].index[0]) 
        
        RelevantNodeLats = [Facilities.iloc[i]["y"]]
        RelevantNodeLongs = [Facilities.iloc[i]["x"]]
        
        #And each supply linked to said facility
        for j in FacilityLinkInfo.iloc[i]['Linked Supply List']:
            #Plot the link
            linecolor = cm.RdYlGn((float(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)].replace("\'", "").replace("%", ""))/100))
            linecolor = tuple(float(s) for s in str(linecolor).replace("np.float64", "").replace("(", "").replace(")","").split(","))
            if mapviewboolean:
                fig.add_trace(go.Scattermapbox(
                    lon = [Facilities.iloc[i]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                    lat = [Facilities.iloc[i]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                    #Red Yellow Green Gradient
                    mode = "lines",
                    line_color = 'rgba' + str(linecolor),
                    legendgroup = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                    legendrank = 1000+int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                    name = MostProbable[int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))],
                    showlegend = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])) not in LegendGroups
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x = [Facilities.iloc[i]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                    y = [Facilities.iloc[i]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                    #Red Yellow Green Gradient
                    mode = "lines",
                    line_color = 'rgba' + str(linecolor),
                    legendgroup = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                    legendrank = 1000+int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])),
                    name = MostProbable[int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))],
                    showlegend = int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen])) not in LegendGroups
                    ))
                
            LegendGroups = LegendGroups + [int(any(FacilityLinkInfo.iloc[i][probmethod_chosen][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)] < k for k in FacilityLinkInfo.iloc[i][probmethod_chosen]))]
            RelevantNodeLats = RelevantNodeLats + [Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']]
            RelevantNodeLongs = RelevantNodeLongs + [Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']]
            
        #Add border around relevant nodes
        
        if mapviewboolean:
            fig.add_trace(go.Scattermapbox(
                lat=[min(RelevantNodeLats)-0.25, min(RelevantNodeLats)-0.25, max(RelevantNodeLats)+0.25,max(RelevantNodeLats)+0.25, min(RelevantNodeLats)-0.25],
                lon=[min(RelevantNodeLongs)-0.25, max(RelevantNodeLongs)+0.25, max(RelevantNodeLongs)+0.25,min(RelevantNodeLongs)-0.25, min(RelevantNodeLongs)-0.25],
                mode='lines',
                fill='toself',
                fillcolor='rgba(135, 206, 250, 0.25)',
                line=dict(color='DarkBlue', width=2),
                showlegend = False
            ))
            fig.update_traces(hoverinfo='skip')
        else:
            fig.add_shape(type="rect",
            x0=min(RelevantNodeLongs) - 0.25, y0=min(RelevantNodeLats) - 0.25, x1=max(RelevantNodeLongs) + 0.25, y1=max(RelevantNodeLats) + 0.25,
            line=dict(
                color="DarkBlue",
                width=2, 
                ),
            fillcolor="LightSkyBlue", opacity=0.5,
            ) 
        
    elif int(node_chosen.split('(')[1].replace(')', "")) in SupplyLinkInfo["Supply Node ID"].tolist():
         node_chosen = int(node_chosen.split('(')[1].replace(')', ""))
         i = int(SupplyLinkInfo[SupplyLinkInfo["Supply Node ID"]==node_chosen].index[0])
         
         RelevantNodeLats = [SupplyLinkInfo.iloc[i]["Supply Lat-Longs"][0]]
         RelevantNodeLongs = [SupplyLinkInfo.iloc[i]["Supply Lat-Longs"][1]]
         
         if SupplyLinkInfo.iloc[i]["Linked Facility List"] != ['N/A']:
            for j in SupplyLinkInfo.iloc[i]["Linked Facility List"]:
                linecolor = cm.RdYlGn((float(SupplyLinkInfo.iloc[i][probmethod_chosen][SupplyLinkInfo.iloc[i]['Linked Facility List'].index(j)].replace("\'", "").replace("%", ""))/100))
                linecolor = tuple(float(s) for s in str(linecolor).replace("np.float64", "").replace("(", "").replace(")","").split(","))
                
                if mapviewboolean:
                    fig.add_trace(go.Scattermapbox(
                        lon = [SupplyLinkInfo.iloc[i]["Supply Lat-Longs"][1], FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID'] == j].iloc[0]["Facility Lat-Longs"][1]], 
                        lat = [SupplyLinkInfo.iloc[i]["Supply Lat-Longs"][0], FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID'] == j].iloc[0]["Facility Lat-Longs"][0]], 
                        #Red Yellow Green Gradient
                        mode = "lines",
                        line_color = 'rgba' + str(linecolor),
                        name = "Proposed Links",
                        showlegend = LegendGroups != [0]
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x = [SupplyLinkInfo.iloc[i]["Supply Lat-Longs"][1], FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID'] == j].iloc[0]["Facility Lat-Longs"][1]], 
                        y = [SupplyLinkInfo.iloc[i]["Supply Lat-Longs"][0], FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID'] == j].iloc[0]["Facility Lat-Longs"][0]], 
                        #Red Yellow Green Gradient
                        mode = "lines",
                        line_color = 'rgba' + str(linecolor),
                        name = "Proposed Links",
                        showlegend = LegendGroups != [0]
                        ))
                LegendGroups = [0]
                RelevantNodeLats = RelevantNodeLats + [FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID'] == j].iloc[0]["Facility Lat-Longs"][0]]
                RelevantNodeLongs = RelevantNodeLongs + [FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID'] == j].iloc[0]["Facility Lat-Longs"][1]]
            if mapviewboolean:
                fig.add_trace(go.Scattermapbox(
                    lat=[min(RelevantNodeLats)-0.25, min(RelevantNodeLats)-0.25, max(RelevantNodeLats)+0.25,max(RelevantNodeLats)+0.25, min(RelevantNodeLats)-0.25],
                    lon=[min(RelevantNodeLongs)-0.25, max(RelevantNodeLongs)+0.25, max(RelevantNodeLongs)+0.25,min(RelevantNodeLongs)-0.25, min(RelevantNodeLongs)-0.25],
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(135, 206, 250, 0.25)',
                    line=dict(color='DarkBlue', width=2),
                    showlegend = False
                ))
                fig.update_traces(hoverinfo='skip')
            else:
                fig.add_shape(type="rect",
                x0=min(RelevantNodeLongs) - 0.25, y0=min(RelevantNodeLats) - 0.25, x1=max(RelevantNodeLongs) + 0.25, y1=max(RelevantNodeLats) + 0.25,
                line=dict(
                    color="DarkBlue",
                    width=2, 
                    ),
                fillcolor="LightSkyBlue", opacity=0.5,
                ) 
                
    fig.update_layout(
        legend=dict(
            font=dict(
                size=10,
                color="black"
                ),
            bgcolor="lightcyan",
            bordercolor="Black",
            borderwidth=1,
            orientation="h",
            entrywidth=93,
            yanchor="top",
            y=1.1,
            x=0
            ),
        title = {'y':0.975, 'x':0.83},
        modebar={
          'orientation': 'v',
          'bgcolor': '#E9E9E9',
          'color': 'black',
          'activecolor': '#9ED3CD'
          }
        )   
          
    return fig

#%% Update the DataTable

#table title
@callback(
    Output(component_id = 'Data Table Name', component_property = 'children'),
    Input(component_id='node dropdown', component_property='value'),
)
def update_datatabletitle(node_chosen):
    if node_chosen == "All Nodes":
        tabletitle = "Summary Data [Select a node for more info]"
    elif int(node_chosen.split('(')[1].replace(')', "")) in FacilityLinkInfo["Facility Node ID"].to_list():
        tabletitle = "All Proposed Links for " + priismdata.loc[priismdata['Node ID']==int(node_chosen.split('(')[1].replace(')', ""))].iloc[0]['Facility Name'] + ' (' + node_chosen.split('(')[1] + "\nCoords: " + df1.loc[df1['Facility Node ID']==int(node_chosen.split('(')[1].replace(')', ""))].iloc[0]['Facility Lat-Longs']
    elif int(node_chosen.split('(')[1].replace(')', "")) in SupplyLinkInfo["Supply Node ID"].to_list():
        tabletitle = "All Facilities with Potential Links to " + priismdata.loc[priismdata['Node ID']==int(node_chosen.split('(')[1].replace(')', ""))].iloc[0]['Facility Name'] + ' (' + node_chosen.split('(')[1]+ "\nCoords: " + df2.loc[df2['Supply Node ID']==int(node_chosen.split('(')[1].replace(')', ""))].iloc[0]['Supply Lat-Longs']
    return tabletitle  
 
#table info
@callback(
    Output(component_id = 'linkchart', component_property = 'data'),
    Input(component_id='node dropdown', component_property='value'),
)
  
def update_datatable(node_chosen):
    probmethod_chosen="Probability: Square Method"
    if node_chosen == "All Nodes":  
        ExplodedProbs = FacilityLinkInfo.explode(str(probmethod_chosen), ignore_index = True)[str(probmethod_chosen)].str.replace('%','').to_list()
        ExplodedProbsAvg = round(np.mean([float(x)  for x in ExplodedProbs]), 2)
        ExplodedProbsStDev = round(np.std([float(x)  for x in ExplodedProbs]),2)
        
        HighestProbs = [max([float(sub.replace('%', '')) for sub in x]) for x in FacilityLinkInfo[str(probmethod_chosen)].to_list()]
        HighestProbsAvg = round(np.mean(HighestProbs), 2)
        HighestProbsStDev = round(np.std(HighestProbs),2)
        
        data = [[ExplodedProbsAvg, ExplodedProbsStDev, HighestProbsAvg, HighestProbsStDev]]
        linkdata = pd.DataFrame(data, columns=['Average Likelihood', 'Likelihood StDev', 'Average Highest Likelihood', 'Highest Likelihood StDev'])
    
    elif int(node_chosen.split('(')[1].replace(')', "")) in FacilityLinkInfo["Facility Node ID"].to_list():
        node_chosen = int(node_chosen.split('(')[1].replace(')', ""))
        #Row of stored data
        linkdatarow = FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID']==node_chosen]
         
        #make dataframe from expanded row
        linkdata = pd.DataFrame()
        linkdata["Linked Supply IDs"] = linkdatarow.iloc[0]["Linked Supply List"]
        linkdata["Supply Lat-Longs"] = linkdatarow.iloc[0]["Supply Lat-Longs"] 
        linkdata["Link Likelihood"] = linkdatarow.iloc[0][probmethod_chosen]
         
        #sort by probabilities for chosen prob method
        linkdata["Link Likelihood"] = [x.replace('%', '') for x in linkdata["Link Likelihood"]]
        linkdata["Link Likelihood"] = linkdata["Link Likelihood"].astype(float)
        linkdata = linkdata.sort_values(by=["Link Likelihood"], ascending=False)
        linkdata["Link Likelihood"] = linkdata["Link Likelihood"].astype(str) + '%'
        
        #add WWTP names
        linkdata["Linked Supply IDs"] = [priismdata.loc[priismdata['Node ID']==supplyid].iloc[0]['Facility Name'] + ' (' + str(supplyid) + ')' for supplyid in linkdata["Linked Supply IDs"].tolist()]
         
    elif int(node_chosen.split('(')[1].replace(')', "")) in SupplyLinkInfo["Supply Node ID"].to_list():
        node_chosen = int(node_chosen.split('(')[1].replace(')', ""))
        #Row of stored data
        linkdatarow = SupplyLinkInfo.loc[SupplyLinkInfo['Supply Node ID']==node_chosen]
        if linkdatarow.iloc[0]["Linked Facility List"] != ['N/A']:
            #make dataframe from expanded row
            linkdata = pd.DataFrame()
            linkdata["Linked Facility IDs"] = linkdatarow.iloc[0]["Linked Facility List"]
            linkdata["Facility Lat-Longs"] = linkdatarow.iloc[0]["Facility Lat-Longs"]
            linkdata["Link Likelihood"] = linkdatarow.iloc[0][probmethod_chosen]
           
            #sort by probabilities for chosen prob method
            linkdata["Link Likelihood"] = [x.replace('%', '') for x in linkdata["Link Likelihood"]]
            linkdata["Link Likelihood"] = linkdata["Link Likelihood"].astype(float)
            linkdata = linkdata.sort_values(by=["Link Likelihood"], ascending=False)
            linkdata["Link Likelihood"] = linkdata["Link Likelihood"].astype(str) + '%'    
            
            #add hospital names
            linkdata["Linked Facility IDs"] = [priismdata.loc[priismdata['Node ID']==facilityid].iloc[0]['Facility Name'] + ' (' + str(facilityid) + ')' for facilityid in linkdata["Linked Facility IDs"].tolist()]
        else: 
            linkdata = df2.loc[df2['Supply Node ID']==node_chosen][["Linked Facility List"]]
    
    #reformatting datatable for output
    linkdata = linkdata.astype(str)   
    linkdata = linkdata.to_dict('records')
    return linkdata

#%% Zoomed in graph
@callback(
    Output(component_id='zoomed-graph', component_property='figure'),
    Input(component_id='node dropdown', component_property='value'),
    Input(component_id='mapviewcheckzoom', component_property='value')
)

def update_zoom_graph(node_chosen, mapviewboolean):
    probmethod_chosen="Probability: Square Method"
    '''fix later ?'''
    if node_chosen == "All Nodes":
        return {}
    elif int(node_chosen.split('(')[1].replace(')', "")) in FacilityLinkInfo["Facility Node ID"].to_list() or int(node_chosen.split('(')[1].replace(')', "")) in SupplyLinkInfo["Supply Node ID"].to_list():
        colorsequence = ["green", "blue"]
        node_chosen = int(node_chosen.split('(')[1].replace(')', ""))
        if node_chosen in FacilityLinkInfo["Facility Node ID"].to_list():
            #Find Relevant Nodes
            FacilityNodes = [node_chosen]
            SupplyNodes = FacilityLinkInfo.loc[FacilityLinkInfo['Facility Node ID']==node_chosen].iloc[0]["Linked Supply List"]
        else:
            #Find Relevant Nodes
            SupplyNodes = [node_chosen]
            FacilityNodes = SupplyLinkInfo.loc[SupplyLinkInfo['Supply Node ID']==node_chosen].iloc[0]["Linked Facility List"]
            #small edit to color sequence (if no linked facilities, we just want blue)
            if 'N/A' in FacilityNodes:
                colorsequence = ["blue"]
                FacilityNodes = []
        priismdatasubset = priismdatastrings[priismdata["Node ID"].isin(FacilityNodes + SupplyNodes)]

        #plot nodes
        if mapviewboolean:
            fig=px.scatter_mapbox(priismdatasubset, lon = "Longitude", lat = "Latitude", color = "NodeType", color_discrete_sequence=colorsequence, title = "Zoomed-In Links for Node " + str(node_chosen),
                              width = 353, height = 490,
                              hover_name = 'Facility Name',
                              hover_data = {'Node ID': True,
                                            'Latitude': True,
                                            'Longitude': True,
                                            'NodeType': False
                                            }
                              ).update_layout(dragmode='pan')
        else:
            fig=px.scatter(priismdatasubset, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=colorsequence, title = "Zoomed-In Links for Node " + str(node_chosen),
                              width = 303, height = 420,
                              hover_name = 'Facility Name',
                              hover_data = {'Node ID': True,
                                            'Latitude': True,
                                            'Longitude': True,
                                            'NodeType': False
                                            }
                              ).update_layout(dragmode='pan')
        #Plot Links
        #since one supply or facility was chosen, can draw links between all nodes in facilities to all nodes in supplies.
        for i in FacilityNodes:
            index = int(Facilities[Facilities['Node ID']==i].index[0]) 
            for j in SupplyNodes:
                #Plot the link between Facility i and Supply j
                linecolor = cm.RdYlGn((float(FacilityLinkInfo.iloc[index][probmethod_chosen][FacilityLinkInfo.iloc[index]['Linked Supply List'].index(j)].replace("\'", "").replace("%", ""))/100))
                linecolor = tuple(float(s) for s in str(linecolor).replace("np.float64", "").replace("(", "").replace(")","").split(","))
                
                if mapviewboolean:
                    fig.add_trace(go.Scattermapbox(
                        lon = [Facilities.iloc[index]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                        lat = [Facilities.iloc[index]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                        #Red Yellow Green Gradient
                        mode = "lines",
                        line_color = 'rgba' + str(linecolor),
                        ))
                    fig.update_traces(hoverinfo='skip')
                    fig.update_layout(mapbox_style="open-street-map")
                else:
                    fig.add_trace(go.Scatter(
                        x = [Facilities.iloc[index]["x"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['x']], 
                        y = [Facilities.iloc[index]["y"], Supplies.loc[Supplies['Node ID'] == j].iloc[0]['y']], 
                        #Red Yellow Green Gradient
                        mode = "lines",
                        line_color = 'rgba' + str(linecolor),
                        ))

        fig.update_layout(showlegend=False,
                          modebar={
                            'bgcolor': '#E9E9E9',
                            'color': 'black',
                            'activecolor': '#9ED3CD'
                            }
                          )
        return fig


'''Run the app'''
#if __name__ == '__main__':

app.run(debug=True)

