# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html  
import dash_bootstrap_components as dbc
import base64                                                                                                                                                  
import datetime
import json
import plotly
import io
import numpy as np
from base64 import decodestring
import codecs
import eli5
from eli5.lime import TextExplainer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from flask_caching import Cache

CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem'
}

navbar = dbc.NavbarSimple(
    brand="Assessment Dashboard",
    brand_href="#",
    sticky="top",
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("What this visualisation does."),
                        dcc.Markdown(
                            """
                            Please upload a [Tab-separated file](https://en.wikipedia.org/wiki/Tab-separated_values) that has columns "Essay" and "Grade". Our system will try to visualize important words for your grading decision. After uploading, it might take up to a few minutes to calculate, which words are important, then going through the submissions should go fairly quickly.
                            """
                        ),
                        dcc.Upload(
                            id='upload-corpus',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '95%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },                                                                                                                                                 
                            multiple=False
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.P(id="output-markdown-visualisation")
                    ]
                ),
            ]
        )
    ],
    className="mt-4",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.layout = html.Div([navbar, body])

@app.callback(Output('output-markdown-visualisation', 'children'),
              [Input('upload-corpus', 'contents')],
              [State('upload-corpus', 'filename')])
def parse_corpus(contents, filename):
    if not contents:
        return
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),sep='\t').dropna()
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df.__repr__()

if __name__ == '__main__':
    app.run_server(debug=True)