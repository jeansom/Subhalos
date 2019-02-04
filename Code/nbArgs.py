import sys, ast
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbparameterise import (
    extract_parameters, replace_definitions, parameter_values
)

nbName = str(sys.argv[1])
nbArgs = str(sys.argv[2])
with open(nbName) as f:
    nb = nbformat.read(f, as_version=4)
    
# Get a list of Parameter objects
orig_parameters = extract_parameters(nb)
# Update one or more parameters
params = parameter_values(orig_parameters, args=nbArgs)

# Make a notebook object with these definitions, and execute it.
new_nb = replace_definitions(nb, params, execute=False)
ep = ExecutePreprocessor(timeout=-1)
ep.preprocess(new_nb, {})
