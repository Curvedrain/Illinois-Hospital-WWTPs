# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14, 2024

The code requires three inputs: priismdata (facility and supply lat-longs), facilitylinkdata, and supplylinkdata

This code uses plotly dash to create an interactive user interface for viewing links and probabilities.

@author: Ryan Chen (The Cooper Union)
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
# %% 
def columnconvert(df, columnname):
    df[columnname] = df[columnname].apply(ast.literal_eval)

#%%
def outputdash(priismdata, facilitylinkdata, supplylinkdata, searchtype):  
    
    '''Set up/Reformatting'''        

    #Facilities come before supplies
    priismdata=priismdata.sort_values(by=['Supply', 'Node ID']).reset_index(drop=True)
    
    priismdatastrings = priismdata
    priismdatastrings["Supply"] = priismdatastrings["Supply"].astype(str)
    priismdatastrings[["Supply"]] = priismdatastrings[['Supply']].replace(['0','1'], ['Facility', 'Supply'])
    
    priismdatastrings = priismdatastrings.rename(columns={'x': 'Longitude', 'y': 'Latitude', 'Supply': 'NodeType'})
    priismdatastrings['Node ID'] = 'Node ID: ' + priismdatastrings['Node ID'].astype(str)
    
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
    
    def createdefaultgraph():
        fig=px.scatter(priismdatastrings, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Precise Area Search",
                       width = 505, height = 650,
                       hover_name = 'Node ID',
                       hover_data = {'Latitude': True,
                                     'Longitude': True,
                                     'NodeType': False
                                     },
                       ).update_layout(dragmode='pan')
        #For each facility
        for i in range (0, FacilityLinkInfo.shape[0]):
            #And each supply linked to said facility
            for j in FacilityLinkInfo.iloc[i]['Linked Supply List']:
                
                #Find color in Red Yellow Green Gradient, and convert to rgba
                linecolor = cm.RdYlGn(
                    (float(FacilityLinkInfo.iloc[i]['Probability: Square Method'][FacilityLinkInfo.iloc[i]['Linked Supply List'].index(j)].replace("\'", "").replace("%", ""))/100) 
                    )
                linecolor = tuple(float(s) for s in str(linecolor).replace("np.float64", "").replace("(", "").replace(")","").split(","))
                
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
    
    # %% 
    if searchtype == "NodeSearch":
        '''Dash app'''
        
        # Initialize the app with css
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = Dash(external_stylesheets=external_stylesheets)
        
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
                fig=px.scatter_mapbox(priismdatastrings, lon = "Longitude", lat = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Proposed Links: Illinois 2024 Dataset",
                                  width = 505, height = 700, zoom=5.6, center= {'lon': CentralLon, 'lat':CentralLat},
                                  hover_name = 'Node ID',
                                  hover_data = {'Latitude': True,
                                                'Longitude': True,
                                                'NodeType': False
                               
                                }).update_layout(dragmode='pan')
                fig.update_layout(mapbox_style="open-street-map")
            else:
                fig=px.scatter(priismdatastrings, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Proposed Links: Illinois 2024 Dataset",
                                  width = 505, height = 700,
                                  hover_name = 'Node ID',
                                  hover_data = {'Latitude': True,
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
                                      hover_name = 'Node ID',
                                      hover_data = {'Latitude': True,
                                                    'Longitude': True,
                                                    'NodeType': False
                                                    }
                                      ).update_layout(dragmode='pan')
                else:
                    fig=px.scatter(priismdatasubset, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=colorsequence, title = "Zoomed-In Links for Node " + str(node_chosen),
                                      width = 303, height = 420,
                                      hover_name = 'Node ID',
                                      hover_data = {'Latitude': True,
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
        app.run(debug=True, port=8050)
            
    #outputdash(pd.read_excel("PRIISM_List_Nomination_22FEB24.xlsx"), pd.read_excel(os.getcwd()+"/Outputted Link Data.xlsx", 'Facility Link Data'), pd.read_excel(os.getcwd()+"/Outputted Link Data.xlsx", 'Supply Link Data', keep_default_na=False))

    #%%
    if searchtype == "AreaSearch":   
        
        # Initialize the app with css
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = Dash(external_stylesheets=external_stylesheets)
        
        # App layout
        app.layout = html.Div([
            #Precision Graph
            html.Div(className='five columns', children=[
                #Inputs for area
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Longitude", style={'text-align':'center', 'margin-top': '0px', 'margin-bottom': '2px', 'padding-top': '0px', 'padding-bottom': '0px'}),
                            html.Th("Latitude", style={'text-align':'center', 'margin-top': '0px', 'margin-bottom': '2px', 'padding-top': '0px', 'padding-bottom': '0px'}),
                            html.Th("Radius (Miles)", style={'text-align':'center', 'margin-top': '0px', 'margin-bottom': '2px', 'padding-top': '0px', 'padding-bottom': '0px'}),
                            html.Th("", style={'text-align':'center','margin-top': '0px', 'margin-bottom': '2px', 'padding-top': '0px', 'padding-bottom': '0px'})
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(dcc.Input(id='long-input', type='number', value= None, style={'text-align':'center', 'width': 100})),
                            html.Td(dcc.Input(id='lat-input', type='number', value=None, style={'text-align':'center', 'width': 100})),
                            html.Td(dcc.Input(id='radius-input', type='number',value=None, style={'text-align':'center', 'width': 120})),
                            html.Td(html.Button(id='submit-button-state', children='Submit'))
                        ])
                    ])
                ]),
                #Graph of proposed links
                dcc.Graph(figure={}, config={'scrollZoom': True, "modeBarButtonsToRemove": ['resetScale2d',  'select2d', 'lasso2d']}, id='precise-controls-and-graph')
                ]),
            #Freehand Graph
            html.Div(className='five columns', children=[
                
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Choose an area selection method", style={'text-align':'center','margin-top': '0px', 'margin-bottom': '2px', 'padding-top': '0px', 'padding-bottom': '0px', 'width': '100%'}, colSpan = '3'),
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Button(id='Box-Select', children='Box Select')),
                            html.Td(html.Button(id='Pan-Mode', children='Pan Mode')),
                            html.Td(html.Button(id='Lasso-Select', children='Lasso Select'))
                        ])
                    ])
                ]),
                #Graph of proposed links
                dcc.Graph(figure={}, config={'scrollZoom': True, "modeBarButtonsToRemove": ['resetScale2d']}, id='freehand-controls-and-graph'),
            ]),
            
            html.Div(className= 'two columns', children=[
                dcc.RadioItems(['Precise Search', 'Freehand Search'], 'Precise Search', inline = True, id = 'search-type'),
                dash_table.DataTable(id="selectedlinklist", style_cell={'text-align': 'center'})
            ])
        ])
    #%%
        #Update the precision graph
        @callback(
            Output(component_id='precise-controls-and-graph', component_property='figure'),
            Input('submit-button-state', 'n_clicks'),
            State('long-input', 'value'),
            State('lat-input', 'value'),
            State('radius-input', 'value')
        )
        
        def updateprecisiongraph(click, longinput, latinput, radiusinput):
            fig = createdefaultgraph()
            if longinput != None and latinput != None and radiusinput != None:
                #calcs
                r = 3958.8 # miles
                p = np.pi / 180
                
                a = np.sin(radiusinput/2/r) ** 2
                
                extrax = np.arccos(1-2*a / (np.cos(latinput * p) ** 2)) / p
                extray = np.arccos(1-2*a)/p
                #tests
                #print(longinput, latinput, longinput+extrax, longinput-extrax)
                #print(longinput, latinput, latinput+extray, latinput-extray)
                
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=longinput-extrax, y0=latinput-extray, x1=longinput+extrax, y1=latinput+extray,
                    line_color="Red",
                )
            
            return fig
    #%%
        #Update the freehand graph
        @callback(
            Output(component_id='freehand-controls-and-graph', component_property='figure'),
            Input('Box-Select', 'n_clicks'),
            Input('Pan-Mode', 'n_clicks'),
            Input('Lasso-Select', 'n_clicks'),
        )
        
        def updatefreehandgraph(boxselect, panmode, lassoselect):
            button_id = 'Box-Select'
            if ctx.triggered_id != None:
                button_id = ctx.triggered_id 

            fig = createdefaultgraph()
                
            fig.update_layout(
                title = {'text': "Freehand Area Search: " + button_id},
                uirevision='0'
                )
            if button_id == 'Box-Select':
                fig.update_layout(dragmode = 'select')
            
            if button_id == 'Pan-Mode':
                fig.update_layout(dragmode ='pan', uirevision='0')
            
            if button_id == 'Lasso-Select':
                fig.update_layout(dragmode ='lasso')
            
            
            return fig
        
        #List of Links
        @callback(
            Output(component_id = 'selectedlinklist', component_property = 'data'),
            Input('submit-button-state', 'n_clicks'),
            Input(component_id='search-type', component_property='value'),
            State('long-input', 'value'),
            State('lat-input', 'value'),
            State('radius-input', 'value'),
            Input('freehand-controls-and-graph', 'selectedData'),
            Input('Box-Select', 'n_clicks'),
            Input('Pan-Mode', 'n_clicks'),
            Input('Lasso-Select', 'n_clicks')
        )
        def updatelinklist(click, searchtype, longinput, latinput, radiusinput, freehandgraphdata, boxselect, panmode, lassoselect):
            
            NodeList = []
            LinkList = []
            
            if searchtype == 'Precise Search':
                if longinput != None and latinput != None and radiusinput != None:
                    #calcs
                    r = 3958.8 # miles
                    p = np.pi / 180
                    
                    a = np.sin(radiusinput/2/r) ** 2
                    
                    extrax = np.arccos(1-2*a / (np.cos(latinput * p) ** 2)) / p
                    extray = np.arccos(1-2*a)/p
                
                    #For each node
                    for i in range(0, priismdata.shape[0]):
                        longitude = priismdata.iloc[i]['x']
                        latitude = priismdata.iloc[i]['y']
                        
                        #Determine if the node is in the ellipse
                        if ((longitude-longinput)**2) / (extrax ** 2) + ((latitude-latinput)**2) / (extray ** 2) <= 1:
                            NodeList = NodeList + [priismdata.iloc[i]['Node ID'].item()]
            
                    #For each link
                    for i in range(0, FacilityLinkInfo.shape[0]):
                        #Get the facility coords
                        f1 = FacilityLinkInfo.iloc[i]["Facility Lat-Longs"][1]
                        f2 = FacilityLinkInfo.iloc[i]["Facility Lat-Longs"][0]
                        #Get the supply coords
                        for j in range(0,len(FacilityLinkInfo.iloc[i]["Supply Lat-Longs"])):
                            s1 = FacilityLinkInfo.iloc[i]["Supply Lat-Longs"][j][1]
                            s2 = FacilityLinkInfo.iloc[i]["Supply Lat-Longs"][j][0]
                            
                            #If either nodes is in the ellipse (in node list)
                            if FacilityLinkInfo.iloc[i]["Facility Node ID"].item() in NodeList or FacilityLinkInfo.iloc[i]["Linked Supply List"][j] in NodeList:
                                LinkList = LinkList + [(FacilityLinkInfo.iloc[i]["Facility Node ID"].item(), FacilityLinkInfo.iloc[i]["Linked Supply List"][j])]
                            #Otherwise, determine if the link is in the ellipse using quadratic equation (Both nodes outside, but link is through ellipse)
                            else:
                                m = (f2-s2) / (f1-s1)
                                
                                a = extray**2 + (extrax*m) ** 2
                                b = -2 * (longinput * (extray ** 2) + m * (extrax ** 2) * (m * f1 - f2 + latinput))
                                c = (longinput * extray) ** 2 + (extrax * (m * f1 - f2 + latinput)) ** 2 - (extrax * extray) ** 2
                                
                                if b**2 - 4*a*c >= 0:
                                    #print(b**2 - 4*a*c)
                                    #print((f1,f2),(s1,s2))
                                    if min(f1,s1) <= (-b + np.sqrt(b**2 - 4*a*c))/2/a <= max(f1,s1) or min(f1,s1) <= (-b - np.sqrt(b**2 - 4*a*c))/2/a <= max(f1,s1):
                                        LinkList = LinkList + [(FacilityLinkInfo.iloc[i]["Facility Node ID"].item(), FacilityLinkInfo.iloc[i]["Linked Supply List"][j])]                
            
            if searchtype == 'Freehand Search' and freehandgraphdata != None:
                button_id = 'Box-Select'
                if ctx.triggered_id != None:
                    button_id = ctx.triggered_id
                    #print(button_id)
                    
                if button_id == 'Pan-Mode' or ('range' not in freehandgraphdata and 'lassoPoints' not in freehandgraphdata):
                    return no_update
     
                #print(freehandgraphdata)
                pointlist = freehandgraphdata.get('points')
                NodeList = [dictionary.get('hovertext').replace('Node ID: ', '') for dictionary in pointlist]
                NodeList = [int(nodeid) for nodeid in NodeList]
                #Go through all links. If the facility or supply is in the NodeList, so is the link.
                
                for i in range(0, FacilityLinkInfo.shape[0]):
                    for j in range(0,len(FacilityLinkInfo.iloc[i]["Supply Lat-Longs"])):
                        if FacilityLinkInfo.iloc[i]["Facility Node ID"].item() in NodeList or FacilityLinkInfo.iloc[i]["Linked Supply List"][j] in NodeList:
                            LinkList = LinkList + [(FacilityLinkInfo.iloc[i]["Facility Node ID"].item(), FacilityLinkInfo.iloc[i]["Linked Supply List"][j])]
                        else:
                            intersection = 0
                            f1 = FacilityLinkInfo.iloc[i]["Facility Lat-Longs"][1]
                            f2 = FacilityLinkInfo.iloc[i]["Facility Lat-Longs"][0]
                            s1 = FacilityLinkInfo.iloc[i]["Supply Lat-Longs"][j][1]
                            s2 = FacilityLinkInfo.iloc[i]["Supply Lat-Longs"][j][0]
                            m = (f2-s2)/(f1-s1)
                            
                            #add links based on whether range is given from box select or lasso select
                            if 'range' in freehandgraphdata.keys():
                                #print('ranging')
                                xrange = freehandgraphdata.get('range').get('x')
                                yrange = freehandgraphdata.get('range').get('y')
                                #see if intersection is in rectangle and in line segment
                                for xvalue in xrange:
                                    if min(yrange[0], yrange[1]) <= m * (xvalue - f1) + f2 <= max(yrange[0], yrange[1]) and (min(f1,s1) <= xvalue <= max(f1,s1)):
                                        intersection = intersection + 1
                                for yvalue in yrange:
                                    if min(xrange[0], xrange[1]) <= (yvalue - f2)/m + f1 <= max(xrange[0], xrange[1]) and (min(f2,s2) <= yvalue <= max(f1,s2)):
                                        intersection = intersection + 1
                            if 'lassoPoints' in freehandgraphdata.keys():
                                #print("lassoing")
                                xvalues = freehandgraphdata.get('lassoPoints').get('x')
                                yvalues = freehandgraphdata.get('lassoPoints').get('y')
                                for k in range(0, len(xvalues)-1):
                                    #vertical line case
                                    if xvalues[k+1] - xvalues[k] == 0:
                                        xvalue = xvalues[k]
                                        if min(yvalues[k], yvalues[k+1]) <= m * (xvalue - f1) + f2 <= max(yvalues[k], yvalues[k+1]) and (min(f1,s1) <= xvalue <= max(f1,s1)):
                                            intersection = intersection + 1
                                    else:
                                        m1 = (yvalues[k+1] - yvalues[k]) / (xvalues[k+1] - xvalues[k])
                                        #If they have the same slope
                                        if m == m1:
                                            #They must be on the same line and in same interval
                                            if (yvalues[k] - m1 * xvalues[k] == f2 - m * f1) and ((min(f1,s1) <= xvalues[k] <= max(f1,s1)) or (min(f1,s1) <= xvalues[k+1] <= max(f1,s1))):
                                                intersection = intersection + 1
                                        #Otherwise, intersection between the two lines must exist and must be in the interval for both line segments
                                        elif (min(f1,s1) <= (yvalues[k] - m1 * xvalues[k] + m * f1 - f2) / (m-m1) <= max(f1,s1)) and (min(xvalues[k],xvalues[k+1]) <= (yvalues[k] - m1 * xvalues[k] + m * f1 - f2) / (m-m1) <= max(xvalues[k],xvalues[k+1])):
                                            intersection = intersection + 1
                            if intersection != 0:
                                LinkList = LinkList + [(FacilityLinkInfo.iloc[i]["Facility Node ID"].item(), FacilityLinkInfo.iloc[i]["Linked Supply List"][j])]
            print("Node List", NodeList)
            NodeList = [priismdata.loc[priismdata['Node ID']==NodeID].iloc[0]['Facility Name'] + ' (' + str(NodeID) + ')' for NodeID in NodeList]
            print("Link List", LinkList)
            LinkList = ['[' + (priismdata.loc[priismdata['Node ID']==Links[0]].iloc[0]['Facility Name'] + ' (' + str(Links[0]) + ')' ) + ', ' + (priismdata.loc[priismdata['Node ID']==Links[1]].iloc[0]['Facility Name'] + ' (' + str(Links[1]) + ')' ) + ']' for Links in LinkList]
                
            if len(NodeList) > len(LinkList):
                LinkList += [''] * (len(NodeList) - len(LinkList))
            else:
                NodeList += [''] * (len(LinkList) - len(NodeList))
            
            if NodeList == [] and LinkList ==[]:
                NodeList = ['']
                LinkList = ['']
            affectednetworkdata = pd.DataFrame({'Affected Nodes': NodeList, 'Affected Links': LinkList})
            
            #format for plotly dash datatable
            affectednetworkdata = affectednetworkdata.astype(str)   
            affectednetworkdata = affectednetworkdata.to_dict('records')

            return affectednetworkdata
        '''Run the app'''
        #if __name__ == '__main__':
        app.run(debug=True, port=8051)
 
outputdash(pd.read_excel(https://github.com/Curvedrain/Illinois-Hospital-WWTPs/blob/main/PRIISM_List_Nomination_06SEP24v2.xlsx), pd.read_excel(https://github.com/Curvedrain/Illinois-Hospital-WWTPs/blob/main/Outputted%20Link%20Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx, 'Facility Link Data'), pd.read_excel(https://github.com/Curvedrain/Illinois-Hospital-WWTPs/blob/main/Outputted%20Link%20Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx, 'Supply Link Data', keep_default_na=False),"NodeSearch")
outputdash(pd.read_excel(https://github.com/Curvedrain/Illinois-Hospital-WWTPs/blob/main/PRIISM_List_Nomination_06SEP24v2.xlsx), pd.read_excel(https://github.com/Curvedrain/Illinois-Hospital-WWTPs/blob/main/Outputted%20Link%20Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx, 'Facility Link Data'), pd.read_excel(https://github.com/Curvedrain/Illinois-Hospital-WWTPs/blob/main/Outputted%20Link%20Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx, 'Supply Link Data', keep_default_na=False), "AreaSearch")

#outputdash(pd.read_excel("PRIISM_List_Nomination_06SEP24v2.xlsx"), pd.read_excel(r"C:\Users\detec\OneDrive\Documents\Python Scripts\Illinois Wastewater Network Files\Outputted Link Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx", 'Facility Link Data'), pd.read_excel(r"C:\Users\detec\OneDrive\Documents\Python Scripts\Illinois Wastewater Network Files\Outputted Link Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx", 'Supply Link Data', keep_default_na=False),"NodeSearch")
#outputdash(pd.read_excel("PRIISM_List_Nomination_06SEP24v2.xlsx"), pd.read_excel(r"C:\Users\detec\OneDrive\Documents\Python Scripts\Illinois Wastewater Network Files\Outputted Link Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx", 'Facility Link Data'), pd.read_excel(r"C:\Users\detec\OneDrive\Documents\Python Scripts\Illinois Wastewater Network Files\Outputted Link Data_PRIISM_List_Nomination_06SEP24v2.xlsx.xlsx", 'Supply Link Data', keep_default_na=False), "AreaSearch")
