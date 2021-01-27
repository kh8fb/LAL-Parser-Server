"""
Server for obtaining parsed output using the downloaded LAL-Parser model.
"""

import click
from collections import OrderedDict
from flask import Flask, request, Response, send_file
import json
import torch


from . import cli_main
from .run_parser import load_model
from .run_parser import run_parser
from .vocabulary import vocabulary

app = Flask(__name__)
MODEL_DICT = {}

@app.route("/model/", methods=["POST"])
def run_model():
    """
    Run the model with the passed input.  The input can be a JSON file with multiple IDs.
    The outputs are saved as a .json file with the id followed by the 
    """
    if request.method == 'POST':
        #data = request.get_json(force=True)
        print(request)
        data = request.json
        print("DATA: ", data)

        sequences = data["sequences"]

        dependencies = run_parser(MODEL_DICT["model"],
                                  sequences,
                                  DEPENDENCY,
                                  CONSTITUENCY)

        content = json.dumps(dependencies)
        return Response(content, mimetype="/application/json",headers={'Content-Disposition':'attatchment;filename=returned_dependencies.json'})


@cli_main.command(help="Start a server and initialize the models for creating parsed outputs.")
@click.option(
    "-h",
    "--host",
    required=False,
    default="localhost",
    help="Host to bind to. Default localhost"
)
@click.option(
    "-p",
    "--port",
    default=8888,
    required=False,
    help="Port to bind to. Default 8888"
)
@click.option(
    "--cuda/--cpu",
    required=True,
    default=True,
    help="Whether or not to run models on CUDA."
)
@click.option(
    "-mp",
    "--model-path",
    required=True,
    help="Path to pretrained LAL model"
)
@click.option(
    "--constituency",
    is_flag=True,
    default=False,
    help="Add constituency parsing to returned JSON",
)
@click.option(
    "--dependency",
    is_flag=True,
    default=False,
    help="Add dependency parsing heads and labels to returned JSON",
)
def serve(
        host,
        port,
        cuda,
        model_path,
        constituency,
        dependency,
):
    global MODEL_DICT, DEPENDENCY, CONSTITUENCY

    try:
        model = load_model(model_path, cuda)        
        MODEL_DICT["model"] = model
    except Exception as e:
        print("An Error occurred: ", e)
        raise e
    DEPENDENCY = dependency
    CONSTITUENCY = constituency

    app.run(host=host, port=port, debug=True)
