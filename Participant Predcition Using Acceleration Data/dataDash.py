import pandas as pd
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import sys


path = sys.argv[1]

data = pd.read_csv(path, index_col = 0)


options1 = []

options1.append({'label':'x_acceleration','value':'x_acceleration'})
options1.append({'label':'y_acceleration','value':'y_acceleration'})
options1.append({'label':'z_acceleration','value':'z_acceleration'})


options2 = []
for i in range(22):
	options2.append({'label':i,'value':i})

app = dash.Dash(__name__)

app.layout = html.Div(children=[html.H1('Acceleration Dashboard',
										  style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
								html.Div(["Feature Name",dcc.Dropdown(
									id='feature-dropdown',
									options=options1,
									value='x_acceleration'
									
								),],
								style={'font-size':20}),	

								html.Div(["Participant Label",dcc.Dropdown(
									id='participant-dropdown',
									options=options2,
									value=1
									
								),],
								style={'font-size':20}),							
								html.Br(),
								html.Br(),
								html.Div(dcc.Graph(id='line_plot')),

					])


@app.callback(Output(component_id='line_plot',component_property='figure'),
			[Input(component_id='feature-dropdown',component_property='value'),
			Input(component_id='participant-dropdown',component_property='value')])

def get_graph(entered_feature, entered_participant):
	df = data[data.labels==int(entered_participant)].loc[:,['timestamp',entered_feature]]

	gs = go.Scatter(x=df.loc[:,'timestamp'],y=df.loc[:,entered_feature],mode='lines',marker={'color':'green'})
	fig = go.Figure(data=gs)
	
	fig.update_layout(xaxis_title='Timestamps',yaxis_title=entered_feature,title='Timestamps vs Average '+ entered_feature)
	return fig





if __name__=='__main__':
	app.run_server()