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

NodeIDs = [str(x) + ' (' + str(y) + ')' for x, y in zip(priismdata['Facility Name'], priismdata['Node ID'])]
NodeIDs = ["All Nodes"] + NodeIDs

def createdefaultgraph(mapviewboolean, hiddenlinks = None):
    if mapviewboolean:
        CentralLat = 0.5 * ( min(priismdatastrings["Latitude"].tolist()) + max(priismdatastrings["Latitude"].tolist()))
        CentralLon = 0.5 * ( min(priismdatastrings["Longitude"].tolist()) + max(priismdatastrings["Longitude"].tolist()))
        fig=px.scatter_mapbox(priismdatastrings, lon = "Longitude", lat = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Proposed Links: Illinois Dataset",
                          width = 505, height = 575, zoom=5.6, center= {'lon': CentralLon, 'lat':CentralLat},
                          hover_name = 'Facility Name',
                          hover_data = {'Node ID': True,
                                        'Latitude': True,
                                        'Longitude': True,
                                        'NodeType': False
                       
                        }).update_layout(dragmode='pan')
        fig.update_layout(mapbox_style="open-street-map")
    else:
        fig=px.scatter(priismdatastrings, x = "Longitude", y = "Latitude", color = "NodeType", color_discrete_sequence=["green", "blue"], title = "Precise Search Graph",
                          width = 505, height = 575,
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
        margin=dict(t=32, b=0)
        )
    return fig


'''Dash app'''

# Initialize the app with css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(external_stylesheets=external_stylesheets)
server = app.server

# App layout
app.layout = html.Div([
    html.Div(className='twelve columns', children=[
        html.Div(className='three columns', children=[
            html.P('Illinois Wastewater Area Search', style={'font-weight':'bold', 'font-size':'20px', 'margin-left':'5px', 'margin-bottom':'-1px'})
        ]),
        html.Div(className='three columns', children=[
            html.P('Begin by selecting your search method:')
        ], style={'margin-top':'3px', 'margin-right':'5px', 'text-align':'right'}),
        html.Div(className='four columns', children=[
            dcc.RadioItems(['Precise Search', 'Freehand Search'], 'Precise Search', inline = True, id = 'search-type')
            ], style={'margin-top':'3px', 'margin-left':'15px'})
        ]),
    html.Hr(style={'borderWidth': "0.5vh", "width": "100%", "borderColor": "#283242","opacity": "unset",'margin-top':'0px','margin-bottom':'0px'}),
    html.Div([
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
                    html.Td(dcc.Input(id='long-input', type='number', value= None, style={'text-align':'center', 'width': 100, 'height': 30})),
                    html.Td(dcc.Input(id='lat-input', type='number', value=None, style={'text-align':'center', 'width': 100, 'height': 30})),
                    html.Td(dcc.Input(id='radius-input', type='number',value=None, style={'text-align':'center', 'width': 120, 'height': 30})),
                    html.Td(html.Button(id='submit-button-state', children='Submit', style={'height': '30px', "display": "flex", "alignItems": "center"}))
                ])
            ])
        ]),
        #Graph of proposed links
        dcc.Graph(figure={}, config={'scrollZoom': True, "modeBarButtonsToRemove": ['resetScale2d',  'select2d', 'lasso2d']}, id='precise-controls-and-graph'),
        dcc.Checklist(
            id="mapviewcheckprecision",
            options=[ {"label": "Map View", "value": True} ],
            value=[]
            )
        ]),
    #Freehand Graph
    html.Div(className='five columns', children=[
        
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Choose a freehand area selection method", style={'text-align':'center','margin-top': '0px', 'margin-bottom': '2px', 'padding-top': '0px', 'padding-bottom': '0px', 'width': '100%'}, colSpan = '3'),
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(html.Button(id='Box-Select', children='Box Select', style={'height': '30px', "display": "flex", "alignItems": "center"})),
                    html.Td(html.Button(id='Pan-Mode', children='Pan Mode', style={'height': '30px', "display": "flex", "alignItems": "center"})),
                    html.Td(html.Button(id='Lasso-Select', children='Lasso Select', style={'height': '30px', "display": "flex", "alignItems": "center"})),
                ])
            ])
        ]),
        #Graph of proposed links
        dcc.Graph(figure={}, config={'scrollZoom': True, "modeBarButtonsToRemove": ['resetScale2d']}, id='freehand-controls-and-graph'),
        
        dcc.Checklist(
            id="mapviewcheckfreehand",
            options=[ {"label": "Map View", "value": True} ],
            value=[]
            )
    ]),
    
    html.Div(className= 'two columns', children=[
        dash_table.DataTable(id="selectedlinklist", style_cell={'text-align': 'center'})
    ])
    ])
])
#%%
#Update the precision graph
@callback(
    Output(component_id='precise-controls-and-graph', component_property='figure'),
    Input('submit-button-state', 'n_clicks'),
    Input('mapviewcheckprecision', component_property='value'),
    State('long-input', 'value'),
    State('lat-input', 'value'),
    State('radius-input', 'value')
)

def updateprecisiongraph(click, mapviewboolean, longinput, latinput, radiusinput):
    fig = createdefaultgraph(mapviewboolean)
    if longinput != None and latinput != None and radiusinput != None:
        #calcs
        r = 3958.8 # miles
        p = np.pi / 180
        
        a = np.sin(radiusinput/2/r) ** 2
        
        extrax = np.arccos(1-2*a / (np.cos(latinput * p) ** 2)) / p
        extray = np.arccos(1-2*a)/p
        if mapviewboolean:
            lons = []
            lats = []
            N = 360
            for n in range(0, N):
                lons = lons + [longinput + extrax * np.cos(2*np.pi/N*n)]
                lats = lats + [latinput + extray * np.sin(2*np.pi/N*n)]
                #print(lons)
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line_color = "Red",
                name = "Area"
                )
            )
        else:
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
    Input('mapviewcheckfreehand', component_property='value'),
    Input('Box-Select', 'n_clicks'),
    Input('Pan-Mode', 'n_clicks'),
    Input('Lasso-Select', 'n_clicks'),
)

def updatefreehandgraph(mapviewboolean, boxselect, panmode, lassoselect):
    button_id = 'Box-Select'
    if ctx.triggered_id != None:
        button_id = ctx.triggered_id 

    fig = createdefaultgraph(mapviewboolean)
    
    if button_id == 'mapviewcheckfreehand':
        fig.update_layout(
            title = {'text': "Freehand Area Search: Pan-Mode"},
            uirevision='0'
            )
    else:
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
    Input('mapviewcheckfreehand', component_property='value'),
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
def updatelinklist(mapviewboolean, click, searchtype, longinput, latinput, radiusinput, freehandgraphdata, boxselect, panmode, lassoselect):
    
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
            
        if button_id == 'Pan-Mode' or ('range' not in freehandgraphdata and 'lassoPoints' not in freehandgraphdata):
            return no_update

        pointlist = freehandgraphdata.get('points')
        NodeList = [dictionary.get('hovertext').replace('Node ID: ', '') for dictionary in pointlist]
        NodeList = [int(priismdatastrings[priismdatastrings["Facility Name"] == nodeid].iloc[0]["Node ID"]) for nodeid in NodeList]
        
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
                        if mapviewboolean:
                            xbounds = [freehandgraphdata.get('range').get('mapbox')[0][0],
                                      freehandgraphdata.get('range').get('mapbox')[1][0]]
                            ybounds = [freehandgraphdata.get('range').get('mapbox')[1][1],
                                      freehandgraphdata.get('range').get('mapbox')[0][1]]
                            
                            xrange = [min(xbounds),
                                      max(xbounds)]
                            yrange = [min(ybounds),
                                      max(ybounds)]
                        else:
                            xrange = freehandgraphdata.get('range').get('x')
                            yrange = freehandgraphdata.get('range').get('y')

                        #see if intersection is in rectangle and in line segment
                        for xvalue in xrange:
                            if min(yrange[0], yrange[1]) <= m * (xvalue - f1) + f2 <= max(yrange[0], yrange[1]) and (min(f1,s1) <= xvalue <= max(f1,s1)):
                                intersection = intersection + 1
                        for yvalue in yrange:
                            if min(xrange[0], xrange[1]) <= (yvalue - f2)/m + f1 <= max(xrange[0], xrange[1]) and (min(f2,s2) <= yvalue <= max(f2,s2)):
                                intersection = intersection + 1
                    if 'lassoPoints' in freehandgraphdata.keys():
                        if mapviewboolean:
                            #iterate over list of coordinates
                            coords = freehandgraphdata.get('lassoPoints').get('mapbox')
                            xvalues = [i[0] for i in coords]
                            yvalues = [i[1] for i in coords]
                        else:
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
    NodeList = [priismdata.loc[priismdata['Node ID']==NodeID].iloc[0]['Facility Name'] + ' (' + str(NodeID) + ')' for NodeID in NodeList]
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

app.run_server(debug=True, host='0.0.0.0', port=8051)





