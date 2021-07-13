import pandas as pd
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output


airline_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})

app = dash.Dash(__name__)

app.layout = html.Div(children=[html.H1('Airline Dashboard',
                                          style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
								html.Div(["Input Year",dcc.Input(id='input-yr',value='2010',type='number',style={'height':'50px','font-size':35}),],
								style={'font-size':40}),		
								html.Div(["State Abbreviation",dcc.Input(id='input-ab',value='AL',type='text',style={'height':'50px','font-size':35}),],
								style={'font-size':40}),							
								html.Br(),
								html.Br(),
								html.Div(dcc.Graph(id='line_plot')),

					])


@app.callback(Output(component_id='line_plot',component_property='figure'),
			[Input(component_id='input-yr',component_property='value'),
			Input(component_id='input-ab',component_property='value')])

def get_graph(entered_year,entered_state):
	df = airline_data[(airline_data.Year==int(entered_year))&(airline_data.OriginState==entered_state)]
	line_data = df.groupby('Month')['ArrDelay'].mean().reset_index()
	gs = go.Scatter(x=line_data['Month'],y=line_data['ArrDelay'],mode='lines',marker={'color':'green'})
	fig = go.Figure(data=gs)
	
	fig.update_layout(xaxis_title='Month',yaxis_title='ArrDelay',title='Month vs Average Flight Delay Time')
	return fig





if __name__=='__main__':
	app.run_server()
