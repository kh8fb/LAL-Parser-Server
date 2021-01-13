# LAL Parser Server

This is a Flask-based server intended for running and receiving parsed sentences from the trained [LAL Parser by Khalil Mrini](https://github.com/KhalilMrini/LAL-Parser) .  This package goes with a  Singularity container definition script [here](https://github.com/kh8fb/kh8fb_singularity/tree/master/definition_scripts).  Below are some helpful installation steps and tricks.  

## Installation

First, create an Anaconda environment:

       conda create -n lal-parser-server

Next, activate the environment, cd into this project's directory and install the requirements with

      >>> conda activate lal-parser-server
      >>> cd LAL-Parser-Server/
      >>> pip install -r requirements.txt
      >>> pip install -e . # enables the CLI interface and allows other packages to import it
      >>> cd vocabulary/ 
      >>> pip install -e .

Now your environment is set up and you're ready to go.

Finally, download the model from Google Drive with

	 	  >>> pip install gdown
		      >>> gdown https://drive.google.com/uc?id=1LC5iVcvgksQhNVJ-CbMigqXnPAaquiA2 -O /path/to/model.pt

## Usage

Activate the server directly from the command line with 

	 >>> lal-parser-server -mp /path/to/model.pt --cpu

This command starts the server and loads the model so that it's ready to go when called upon.  To run with the model on CUDA, use  `--cuda` instead of `--cpu`.   You can also provide additional arguments such as the `--hostname` and  `--port` of the server.

After the server has been started, you can receive parsed output in another terminal window with the `curl` command.   This parses the each of the input phrases from `test_json.json`  and dumps the JSON for all of the outputs into `my_output.json`

      >>> curl http://localhost:8888/model/ --data @test_json.json --output my_output.json -H 'Content-Type: application/json; chartset utf-8'

The format of the input JSON should match the following with a "sequences" field and then a sentence matched to each "id":

    >>> cat test_json.json
    {"sequences": {"1": "This is the first input phrase.","2": "This is the last and final input phrase?"}}
    >>> cat my_output.json
    {"Dependencies": {"1": "(S (NP (DT This)) (VP (VBZ is) (NP (DT the) (JJ first) (NN input) (NN phrase))) (. .))", "2": "(S (NP (DT This)) (VP (VBZ is) (NP (DT the) (ADJP (JJ last) (CC and) (JJ final)) (NN input) (NN phrase))) (. ?))"}}


## Citation

If you use the Neural Adobe-UCSD Parser, please cite our [paper](https://arxiv.org/abs/1911.03875) as follows:
```
@article{mrini2019rethinking,
  title={Rethinking Self-Attention: An Interpretable Self-Attentive Encoder-Decoder Parser},
  author={Mrini, Khalil and Dernoncourt, Franck and Bui, Trung and Chang, Walter and Nakashole, Ndapa},
  journal={arXiv preprint arXiv:1911.03875},
  year={2019}
}
```
