#!/usr/bin/python
"""
Dashboard for assessment using Natural Language Processing.

Run python3 dashboard.py in a terminal to serve a local copy of the dashboard

Copyright (C) 2019 Andreas Haupt

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import io
import base64
import pickle
import os
from typing import List, Tuple, BinaryIO, Optional

import dash
from dash import dependencies
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import eli5.lime
import bs4
import flask_caching
import pandas as pd

from sklearn.feature_extraction import text
from sklearn import svm
from sklearn import decomposition
from sklearn import pipeline

NAVBAR = dbc.NavbarSimple(brand="Assessment Dashboard",
                          brand_href="#",
                          sticky="top")

EXPLAINING_TEXT = (
    "Please upload a"
    "[Tab-separated file](https://en.wikipedia.org/wiki/Tab-separated_values) "
    "that has columns 'Essay' and 'Grade'. Our system will try to visualize "
    "important words for your grading decision. After uploading, it might take "
    "up to a few minutes to calculate, which words are important, then going "
    "through the submissions should go fairly quickly.")

BODY = dbc.Container(
    [
        dbc.Row([
            dbc.Col(
                [
                    html.H2("How to use this visualisation?"),
                    dcc.Markdown(EXPLAINING_TEXT),
                    dcc.Upload(id="upload-corpus",
                               children=html.Div([
                                   "Drag and Drop or ",
                                   html.A("Select Files")
                               ]),
                               style={
                                   "width": "95%",
                                   "height": "60px",
                                   "lineHeight": "60px",
                                   "borderWidth": "1px",
                                   "borderStyle": "dashed",
                                   "borderRadius": "5px",
                                   "textAlign": "center",
                                   "margin": "10px"
                               },
                               multiple=False),
                ],
                md=4,
            ),
            dbc.Col([
                daq.NumericInput(  # pylint: disable=no-member
                    disabled=True,
                    id="input-num-submission"),
                html.Div(id="output-markdown-text")
            ]),
        ])
    ],
    className="mt-4",
)

APP = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
CACHE = flask_caching.Cache(APP.server,
                            config={
                                "CACHE_TYPE": "filesystem",
                                "CACHE_DIR": "./.flask_cache"
                            })
APP.layout = html.Div([NAVBAR, BODY])


def _html_helper(raw_string: str) -> List[html.Span]:
    """
    Extracts a list of colored spans from string

    Args:
        raw_string: HTML string that is output by an eli5.TextExplainer instance
            upon method execution of show_prediction

    Returns:
        A list of dcc.html.Spans that encode the color

    >>> _html_helper("<span style={opacity=0.88}> </span>")
    [Span(' ')]
    """
    soup = bs4.BeautifulSoup(raw_string, "html.parser")
    span_list = []
    for span in soup.find_all("span"):
        assert len(span.get("style").split(";")) < 3, "malformed input string"
        if len(span.get("style").split(";")) == 2:
            color = span.get("style").split(";")[0].split(":")[1]
            span_list.append(html.Span(span.get_text(), style={'color':
                                                               color}))
        else:
            span_list.append(html.Span(span.get_text()))
    return span_list


@APP.callback([
    dependencies.Output("input-num-submission", "label"),
    dependencies.Output("input-num-submission", "labelPosition"),
    dependencies.Output("input-num-submission", "value"),
    dependencies.Output("input-num-submission", "min"),
    dependencies.Output("input-num-submission", "max"),
    dependencies.Output("input-num-submission", "disabled")
], [dependencies.Input("upload-corpus", "contents")],
              [dependencies.State("upload-corpus", "filename")])
def parse_corpus(contents: str,
                 filename: str) -> Tuple[str, str, int, int, int, bool]:
    """
    Parse submissions and train a classifier

    Parses a file input through the dashboard's upload component, learns a classifier and
    trains a sklearn classifier. As a side effect, it stores test data and the classifier
    in local storage.

    Args:
        Contents: Binary input of the file.
        filename: string denoting the filename of the file.
    """
    if not contents:
        return "", "", 0, 0, 0, False
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    data = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")
    data["Grade"].astype("Int64")
    train = data[data["Grade"].notnull()]
    train_data = train["Essay"]
    train_labels = train["Grade"]
    train_labels_lower_quartile = train_labels.quantile(q=.25)
    train_labels = (train_labels > train_labels_lower_quartile)
    clf = classifier(train_data, train_labels)
    with open("classifier_for_{}.pickle".format(filename), "wb") as out_file:
        pickle.dump(clf, out_file)
    test_data = data[data["Grade"].isnull()]["Essay"]
    with open("{}.pickle".format(filename), "wb") as out_file:
        pickle.dump(list(test_data), out_file)
    return ("Index of ungraded submission in file {}".format(filename),
            "right", 1, 1, len(test_data), False)


def classifier(train_data: pd.Series, train_labels) -> pipeline.Pipeline:
    """
    Trains a classifier on labeled training data.

    Trains an inverse document frequency-vectorised support-vector machine after
    dimension reduction by a singular value decomposition.

    Args:
        train_data: A pd.Series containing the essays as strings
        train_labels: A pd.Series containing the grades as integers.

    Returns:
        An sklearn.Classifier trained on the input data.
    """
    vec = text.TfidfVectorizer(min_df=3,
                               stop_words="english",
                               ngram_range=(1, 2))
    svd = decomposition.TruncatedSVD()
    lsa = pipeline.make_pipeline(vec, svd)
    clf = svm.SVC(gamma="scale", probability=True)
    pipe = pipeline.make_pipeline(lsa, clf)
    pipe.fit(train_data, train_labels)
    return pipe


@CACHE.memoize(timeout=600)
def load_classifier(filename: str) -> pipeline.Pipeline:
    """
    Retrieve the classifier.

    Retrieve the classifier. Utility function to be able to cache the classifier.

    Args:
        filename: To ensure that for a new file it is not read from the cache

    Returns:
        An sklearn.Classifier imported from pickle.
    """
    with open("classifier_for_{}.pickle".format(filename), "rb") as in_file:
        return pickle.load(in_file)


@APP.callback(dependencies.Output("output-markdown-text", "children"),
              [dependencies.Input("input-num-submission", "value")],
              [dependencies.State("upload-corpus", "filename")])
@CACHE.memoize(timeout=600)
def explain_text(value: int, filename: str) -> Optional[List[html.Span]]:
    """
    Return a colorful HTML output given value and filename of file to be explained.

    Train a TextExplainer instance on the data stored in filename and transform it
    for use with dash.

    Args:
        value: integer denoting the index of the ungraded submission to be graded
        filename: a string giving the filename. Needed to not read from cache when not intended.

    Returns:
        A list of html.Spans that return a colorful text output.
    """
    if not value:
        return None
    text_explainer = eli5.lime.TextExplainer(random_state=42)
    with open(filename + ".pickle", "rb") as in_file:
        submission = pickle.load(in_file)[value - 1]
    text_explainer.fit(submission, load_classifier(filename).predict_proba)
    prediction_string = text_explainer.show_prediction(top=(10, 10),
                                                       targets=[1]).data
    return _html_helper(prediction_string)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    APP.run_server()
