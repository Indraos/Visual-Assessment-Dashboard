# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components import Div, Span
import bs4 as bs
import dash_bootstrap_components as dbc
import dash_daq as daq
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
import pickle
from flask_caching import Cache
import re

CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.'
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
                        html.H2("How to use this visualisation"),
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
                            daq.NumericInput(
                            disabled=True,
                            id="input-num-submission"), 
                            html.Div(
                                id="output-markdown-text")
                    ]
                ),
            ]
        )
    ],
    className="mt-4",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
cache = Cache(app.server,config=CACHE_CONFIG)
app.layout = html.Div([navbar, body])



def convert_html_to_dash(el,style = None):
    HTML_CONSTRUCTS =  {'span'}
    def __extract_style(el):
        if not el.attrs.get("style"):
            return None
        return {k.strip():v.strip() for k,v in [x.split(": ") for x in el.attrs["style"].split(";")] if k != "background-color"}
    if type(el) is str:
        return convert_html_to_dash(bs.BeautifulSoup(el,'html.parser'))
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = __extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        if name.title().lower() not in HTML_CONSTRUCTS:        
            return contents[0] if len(contents)==1 else html.Div(contents)
        return getattr(html,name.title())(contents,style = style)



@app.callback([Output('input-num-submission', 'label'),
                Output('input-num-submission', 'labelPosition'),
                Output('input-num-submission', 'value'),
                Output('input-num-submission', 'min'),
                Output('input-num-submission', 'max'),
                Output('input-num-submission', 'disabled')],
              [Input('upload-corpus', 'contents')],
              [State('upload-corpus', 'filename')])
def parse_corpus(contents, filename):
    if not contents:
        return "", None, None, None, None, None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),sep='\t')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    df["Grade"].astype("Int64")
    train = df[df["Grade"].notnull()]
    test = df[~df["Grade"].notnull()]
    train_data = train["Essay"]
    train_labels = train["Grade"]
    test_data = test["Essay"]
    train_labels_lower_quartile = train_labels.quantile(q=.25)
    train_labels = (train_labels > train_labels_lower_quartile)
    target_names = ["Low Grade", "High Grade"]

    vec = TfidfVectorizer(min_df=3, stop_words='english',
                          ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    lsa = make_pipeline(vec, svd)

    clf = SVC(C=150, gamma=2e-2, probability=True)
    pipe = make_pipeline(lsa, clf)
    pipe.fit(train_data, train_labels)
    with open("classifier.pickle", 'wb') as out_file:
        pickle.dump(pipe, out_file)
    with open("data.pickle", 'wb') as out_file:
        pickle.dump(train_data, out_file)
    return ['Index of ungraded submission in file {}'.format(filename),
            'right',
            1,
            1,
            len(test_data),
            False]


@cache.memoize()
def load_text(value):
    with open("data.pickle",'rb') as in_file:
        return pickle.load(in_file)[value + 1]

@cache.memoize()
def load_classifier():
    with open("classifier.pickle",'rb') as in_file:
        return pickle.load(in_file)


@app.callback(Output('output-markdown-text', 'children'),
              [Input('input-num-submission', 'value')])
@cache.memoize()
def explain_text(value):
    if not value:
        return None
    te = TextExplainer(random_state=42)
    te.fit(load_text(value), load_classifier().predict_proba)
    prediction_string = te.show_prediction().data
    prediction_string = prediction_string[prediction_string.find("<span style"):-14]
    return html_helper(prediction_string)

def html_helper(raw_string):
    span_list = raw_string.split("><")
    return_spans = []
    for span in span_list[:-1]:
        word = span[:-6].split(">")[1]
        opacity = span.split(">")[0][-5:-2]
        return_spans.append(html.Span(children=word,style={"opacity":.5+.5*float(opacity)}))
    return return_spans

if __name__ == '__main__':
    app.run_server(debug=True)