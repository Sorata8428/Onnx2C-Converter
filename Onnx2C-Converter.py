#!/usr/bin/env python3
import argparse
from genericpath import exists
import onnx
from onnx import numpy_helper
from pathlib import Path
import numpy as np
import sys
import subprocess
import re
import math
from collections import defaultdict
from fractions import Fraction
from functools import reduce

# Operations.py has the interface to create matrix operations using a mixture of variables and numpy arrays and can also serialize them
# A term looks like Add(Mul(array, "v"), Func(Add("q", b), "relu")) where array is a numpy array and b is an operation or another numpy array, "v" and "q" are variable names
# Use serialize_operation(varname, operation) to let varname=operation and get a writeable string
from Operations import * 

#read command line inputs using argparse

#special operation for the -o option
currently_supported = "Currently supported operators: \nMatmul using matrices \nAdd and Bias Add\nReLU and Sigmoid activations \nLinear LSTMs and RNNs with one input"
class _OperationAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(_OperationAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(currently_supported)
        parser.exit()

def displaymatch(match):
    if match is None:
        return None
    return '<Match: %r, groups=%r>' % (match.group(), match.groups())


#create the argument parser
parser = argparse.ArgumentParser(description="Script that takes in the name of an ONNX model in the working directory or a path to one. Produces an equivalent C program using single static assignment.")
parser.add_argument('-o', '--supported-operations', action=_OperationAction, help="Get info on which operations of the ONNX specification the current version of the script supports.")
parser.add_argument('-v', '--verbose', action='store_true', help="Print Frama-C analysis to console.")
parser.add_argument('-l', '--log', nargs='?', const='analysis.txt', default='false', help="Log Frama-C analysis to specified file; this will overwrite the file. Default is \'analysis.txt\'.")
parser.add_argument('--no-analysis', action='store_true', help="Disable Frama-C analysis of the converted C code.")
parser.add_argument('--output-only', action='store_true', help="Suppress Frama-C results including warnings except the range of the output variable(s). Useful to simplify output for programs with many variables.")
parser.add_argument('path', help="Path or name to ONNX model. Name applicable if model in same directory as this script. Default is path mode.")
parser.add_argument('interval', nargs=2, type=float, help="Two numbers denoting the upper and lower value of the input intervals.")
parser.add_argument('--robustness', nargs='?', type=int, const=0, default=-1, help="Check the robustness of a point or of a sequence at the specified position, default 0. The specified sequence has to be present in a file by default called \'input\'. Change the input file via option --input.")
parser.add_argument('-i', '--input', nargs=1, type=str, help="Path to input file if the default \'input\' should be overwritten.")
parser.add_argument('-u', '--unroll', nargs='?', type=int, const=2147483647, default=-1, help="Specify an unrolling depth to apply to RNN-Loops. Set to max int by default to completely unroll loops.")
parser.add_argument('--partial', nargs='?', type=int, const=2147483647, default=-1, help="Set how much of a sequence should be iterated over. Overwrites unroll specification. Primarily used for testing.")
parser.add_argument('--execute', action='store_true', help='Do not convert code to analysable format. Instead use \'input\', compile the program normally and print outputs.')
parser.add_argument('--fraction-expansion', nargs='?', default='false', const=1, help="Convert all values to fractions with given maximum denominator (10 is default, 100 is unfeasable). Then expand via the lcd. Automatically enables the octagon domain for Frama-C. (Only makes sense to use when all activation functions are positively homogenious or only the very last operation is not.)")


# parse command line arguments
try: 
    ns = parser.parse_args(sys.argv[1:])
except argparse.ArgumentError as e:
    print(e)
    exit(1)

if ns.partial > -1:
    ns.unroll = ns.partial

model_path = Path(ns.path)

if not exists(model_path) :
    print("Path to ONNX model is incorrect!")
    exit(1)

onnx_model = onnx.load(model_path)


# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
    exit(1)
else:
    print('The model is valid!')




#get matrix values for the nodes
initializers = onnx_model.graph.initializer
onnx_weights = {}
for initializer in initializers:
    weight = numpy_helper.to_array(initializer)
    onnx_weights[initializer.name] = weight
    
    # limit the amount of decimal points for more compact examples
    #string_weights = np.empty(weight.shape)
    #for index in np.ndindex(weight.shape):
    #    string_weights[index] = "{:.2f}".format(weight[index])
    #onnx_weights[initializer.name] = string_weights

# every node output gets mapped to the corresponding vector name in the c program
# most shape information is handled in the background by Operations.py, but you can manually access them via variable_shapes["output vector name"]
node_variable = {}

# assign which hidden layer in a multi layer recurrent network needs to be read out
recurrent_depth = {}

# since consecutive lstm cells need to be grouped into a single loop, we need to mark already finished cells
covered_recurrents = []

# carry counter for each type of node we support
node_count_dict = defaultdict(int)

datatype = "double"
mini = min
maxi = max
min = ns.interval[0]
max = ns.interval[1]


def variable_name(type : str, number : int) -> str :
    return type + f"_{number}"

def variableName(type : str, number : int, index : tuple) -> str :
    return type + f"_{number}_{str(index)[1:-2]}"



# if we scale back the integer conversion, we have to keep track of the number of MatMuls
# every MatMul adds one to the exponent, since it is homogeneous of order 2 (if one also scales the matrix)
scaling_exponent = 1

# for now we will differentiate between RNNs and FNNs and either remove or retain symbolic input dimensions
is_rnn = False
for node in onnx_model.graph.node:
    if node.op_type in set(["LSTM", "RNN"]):
        is_rnn = True
        break

# iterate through inputs of the graph
for input in onnx_model.graph.input:
    tensor_type = input.type.tensor_type

    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape to construct definite shape:
        # since we now handle LSTMs, we will allow up to one unspecified input dimension for variable length sequences
        variable_dim_encountered = False

        input_dimensions = []
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                input_dimensions.append(d.dim_value)

            elif (d.HasField("dim_param")): # unknown dimension with symbolic name
                
                if not is_rnn:
                    continue
                if not variable_dim_encountered:
                    input_dimensions.append(d.dim_param)
                    variable_dim_encountered = True
                else:
                    print("We do not support Neural Networks with multiple unspecified dimensions. Please set all dimensions up to one that may stay unspecified.")
                    exit(1)
            #else:
                #print ("?", end=", ")  # unknown dimension with no name

        input_name = variable_name("Input", node_count_dict["input_counter"])
        node_variable[input.name] = input_name
        variable_shapes[input_name] = tuple(input_dimensions)

        node_count_dict["input_counter"] += 1
    else:
        print ("unknown rank", end="")
        exit(1)



#we may now iterate through the graph, write to the C file and simultaneously build our node information dict
converted_filename = "converted_onnx.c"
f = open(converted_filename, "w+")
f.write(f"//{model_path.name}\n")


# if we test for robustness, we now specify the input sequence
if ns.execute:
    ns.robustness = 0

if ns.input is not None:
    input_name = ns.input[0]
else:
    input_name = "input"

input_file_length = -1
if ns.robustness > -1:
    try:
        input_file = open(input_name, "r")
    except IOError:
        print(f"File \'{input_name}\' appears to not exist. Please create and provide it with the desired sequence to check.")
        exit(1)

    input_list = list()
    for line in input_file:

        # in case we convert to integers, first save input into onnx_weights and serialize afterwards
        numberlist = line.split(" ")[:-1]
        numberlist = [float(v) for v in numberlist]
        input_list.append(numberlist)

        input_file_length = len(numberlist)

    onnx_weights['Input_0'] = np.array(input_list, dtype=np.float32)    #note that this does not cover multiple different inputs for now
    # to expand this capability, one could choose for expample to save the inputs in a json format dictonary, assigning each input name its input



preamble_string = """
#include "__fc_builtin.h"
#include "math.h"
#define fmax(X, Y) (X < Y ? Y : X)
#define relu(X) fmax(X,0)
#define tanh(X) myTanh(X)
"""

if ns.execute :
    preamble_string = """
#include <stdio.h>
#include <math.h>
#define fmax(X, Y) (X < Y ? Y : X)
#define relu(X) fmax(X,0)
#define tanh(X) myTanh(X)
    """

if ns.fraction_expansion != "false":
    preamble_string += """

long long powi(long long x, unsigned int n) {{
    long long p = x;
    long long res = 1;

    //@ loop unroll {0};
    while(n > 0) {{
        if(n % 2 == 1)
            res *= p;
        p *= p;
        n /= 2;
    }}

    return(res);
}}
    """.format(input_file_length)

preamble_string += """

double sigmoid(double x) {
  if( -x > 0x1.62e42fefa39efp+9 )
    return 0.0;
  else
    return 1.0 / (1.0 + exp(-x));
}

double myTanh(double x) {
  if( x/2 > 0x1.62e42fefa39efp+9 )
    return 1.0;
  else
    return 1.0 - 2 / (exp(2*x) + 1);
}

"""
f.write(preamble_string)
f.write("int main() { \n\n")



# expand all values to be integer values :
scaler = 1      # used to scale back outputs 
if ns.fraction_expansion != "false":

    lc = 1   
    datatype = "long long"
    order = 10 ** int(ns.fraction_expansion)

    def lcm(a : int, b : int) :
        return a * b // math.gcd(a,b)

    def expand_fraction(frac):
        return lc // frac.denominator * frac.numerator

    def fractionalize(array) : # array of floats
        s = array.shape
        #return np.array(list(map(lambda x : Fraction(float(x)).limit_denominator(int(ns.fraction_expansion)), array.ravel()))).reshape(s)#.astype(np.int64)
        return np.array(list(map(lambda x : Fraction(float(x)).limit_denominator(4), array.ravel()))).reshape(s)#.astype(np.int64)

    def integerize(array) :  # array of Fractions
        s = array.shape
        return np.array(list(map(expand_fraction, array.ravel()))).reshape(s).astype(object)

    original = onnx_weights.copy()

    # multiply all values via some order of magnitude in order not to lose relatively large values
    for key, array in onnx_weights.items() :
        onnx_weights[key] = order * array


    # first replace all float/double values via a fraction
    for key, array in onnx_weights.items() :
        onnx_weights[key] = fractionalize(array=array)

    # now find the lcm of all denominators
    for key, array in onnx_weights.items() :
        lc = lcm(lc, reduce(lcm, [x.denominator for x in array.ravel()]) )


    # scale all values by lc to get integer values everywhere
    for key, array in onnx_weights.items() :
        onnx_weights[key] = onnx_weights[key]
        onnx_weights[key] = integerize(array=array)

    maxi_min = integerize(fractionalize(order * np.array([max, min])))
    max, min = maxi_min[0], maxi_min[1]
    
    scaler = lc * order

    max_error = 0.0
    for key, array in onnx_weights.items() :
        max_error = maxi(max_error, np.max(np.abs(scaler * original[key] - array)))
    max_error /= scaler

    print("Maximum L_inf conversion error : ", max_error)
    print("Scaler : ", scaler)


# create static input variables without a dynamic dimension
if not is_rnn:
    if ns.execute:
        interval_string = ""
    elif ns.fraction_expansion == 'false':
        interval_string = f' + Frama_C_double_interval({min},{max})' 
    else :
        interval_string = f" + Frama_C_interval({min},{max})"

    start_point = (lambda x : f'{onnx_weights["Input_0"][0,x]}') if ns.robustness > -1 else (lambda x : "")
    try:
        for input, dimensions in variable_shapes.items():
            for num in range(dimensions[0]):
                f.write(f"{datatype} " + f"{input}_{num}" + f" = {start_point(num)}{interval_string};\n")
    except IndexError:
        print("Assigning input has caused an error. Please check if input has the correct length for this model.")
    f.write("\n")


if ns.robustness > -1 and is_rnn:

    for array, i in zip(onnx_weights["Input_0"], range(onnx_weights["Input_0"].shape[0])):
        
        numberlist = [str(x) for x in array]
        #numberlist = ["{:.6f}".format(float(v)) for v in numberlist]    #inputs in onnx have only 5 non integer digits

        if ns.robustness < len(numberlist) and not ns.execute:
            interval_string = f"+ Frama_C_double_interval({min},{max})"
            if ns.fraction_expansion != "false" :  interval_string = f"+ Frama_C_interval({min},{max})"

            numberlist[ns.robustness] += interval_string
        for j in range(len(numberlist) - 1):
            numberlist[j] += ", "
        line = ''.join(numberlist)

        f.write(f'{datatype} {variableName("Input", 0, (i,))}[] = {{{line}}};\n')  #note that this does not cover multiple different inputs for now


# returns two lists containing 1. all input tensors and 2. names of corresponding variables for non static inputs
def get_inputs(node) -> 'tuple[list[np.ndarray], list[str]]' :
    tensors = []
    ops = []

    for input in node.input:
        if input in onnx_weights:
            tensors.append(onnx_weights[input])
        else:
            ops.append(node_variable[input])

    return tensors, ops


nodelist = onnx_model.graph.node
for node in nodelist:

    ## we only support matrix vector products 
    if node.op_type == "MatMul":

        # we always have exactly one tensor and input
        tensor, prev_var = get_inputs(node)
        tensor, prev_var = tensor[0], prev_var[0]

        var_name = variable_name("MatMul", node_count_dict["matmul_counter"])
        operation = Mul(tensor.T, prev_var)

        f.write(serialize_operation_typed(datatype, var_name, operation))
        f.write("\n")
        
        node_variable[node.output[0]] = var_name
        node_count_dict["matmul_counter"] += 1

        scaling_exponent += 1
    
    elif node.op_type == "Add":

        #we now have 2 inputs, one or none being a tensor associated with this node 
        tensor, prev_vars = get_inputs(node)

        if tensor is not None and ns.fraction_expansion != 'false':
            try :
                tensor[0] *= int(scaler ** (scaling_exponent - 1))
            except OverflowError :
                print("Tensorvalue too large to represent in 64 bits. Scaling factor: ", scaler ** (scaling_exponent - 1))
                exit(1)

        if tensor is not None :
            operation = Add(prev_vars[0], tensor[0])
        else :
            operation = Add(prev_vars[1], prev_vars[0])

        var_name = variable_name("Add", node_count_dict["bias_counter"])
        f.write(serialize_operation_typed(datatype, var_name, operation))
        f.write("\n")

        node_variable[node.output[0]] = var_name
        node_count_dict["bias_counter"] += 1

    elif node.op_type == "Relu":

        #we do not need any tensors for an activation function
        _, prev_var = get_inputs(node)
        prev_var = prev_var[0]


        var_name = variable_name("ReLU", node_count_dict["relu_counter"])
        f.write(serialize_operation_typed(datatype, var_name, Func(prev_var, "relu")))
        f.write("\n")

        node_variable[node.output[0]] = var_name
        node_count_dict["relu_counter"] += 1

    elif node.op_type == "Sigmoid":

        datatype = "double"

        _, prev_var = get_inputs(node)
        prev_var = prev_var[0]

        var_name = variable_name("Sigmoid", node_count_dict["sigmoid_counter"])

        try :
            operation = Func(Div(prev_var, float(scaler ** scaling_exponent)), "sigmoid")      # the div is necessary if we scale back results before feeding them into the sigmoid
        except OverflowError:
            print("Scaler has become to large to fit into a double, cannot scale back value before sigmoid.\n Scaler: ", scaler ** scaling_exponent)
            exit(1)

        f.write(serialize_operation_typed(datatype, var_name, operation))
        f.write("\n")

        node_variable[node.output[0]] = var_name
        node_count_dict["sigmoid_counter"] += 1


    elif node.op_type == "Gemm" :
        # replace gemm with matmul and add node 

        for gemm_node, index in zip(nodelist, range(len(nodelist))):
            if gemm_node == node:
                break

        index += 1
        name = node.name

        matmul_node = onnx.helper.make_node(
            op_type="MatMul",
            inputs=node.input[0:2],
            outputs=[name + "_Glue"])

        bias_node = onnx.helper.make_node(
            op_type="Add",
            inputs=[name + "_Glue", node.input[2]],
            outputs=node.output)

        nodelist.insert(index, bias_node)
        nodelist.insert(index, matmul_node)


    elif node.op_type in set(["LSTM", "RNN"]):
        f.write("\n")

        if node.name in covered_recurrents :
            continue

        # for now we will only feed an input into a recurrent layer, therefore we avoid a lot of potential dataprocessing
        # we also do not allow mixing simple RNN and LSTM layers 

        #get the recurrent nodes we have to put into the same loop
        search_node = node
        recurrent_nodes = [search_node]
        processing_nodes = []
        activations = [[l.decode("utf-8").lower() for l in node.attribute[0].strings]]

        
        # this loop does not cover branching output for the recurrent cell
        for graph_node in onnx_model.graph.node:
            if bool(set(search_node.output).intersection(set(graph_node.input))) : 
                search_node = graph_node
                if search_node.op_type in set(["LSTM", "RNN"]):
                    recurrent_nodes.append(search_node)
                    activations.append([l.decode("utf-8").lower() for l in search_node.attribute[0].strings])

                elif search_node.op_type == "Squeeze" or search_node.op_type == "Slice" : 
                    processing_nodes.append(search_node)

        output_name = processing_nodes[-1].output[0]
        node_count = len(recurrent_nodes)
        recurrent_depth[output_name] = node_count

        counter_type = node.op_type.lower() + "_counter"
        recurrent_counter = node_count_dict[counter_type]

        # Note on order of LSTM inputs:
        # X, W, R, B, ?, initial_h, initial_c
        
        # write initial states
        if node.op_type == "LSTM":
            for i in range(node_count) :
                LSTM_node = recurrent_nodes[i]
                h_initial = onnx_weights[LSTM_node.input[5]]
                c_initial = onnx_weights[LSTM_node.input[6]]
                h_initial = np.squeeze(h_initial)
                c_initial = np.squeeze(c_initial)


                feature_length = h_initial.shape[0]

                f.write(serialize_operation_typed(datatype, f"h_{recurrent_counter}_{i}", Const(h_initial)))
                f.write(serialize_operation_typed(datatype, f"c_{recurrent_counter}_{i}", Const(c_initial)))


        elif node.op_type == "RNN":
            for i in range(node_count) :
                # current conversion to RNN cells assumes initial state as 0
                RNN_node = recurrent_nodes[i]
                feature_length = RNN_node.attribute[1].i
                typ = np.float32 if ns.fraction_expansion == "false" else np.int32
                h_initial = np.zeros(feature_length, dtype=typ)

                f.write(serialize_operation_typed(datatype, f"h_{recurrent_counter}_{i}", Const(h_initial)))

                
        # handle the range of the for loop
        if ns.unroll == -1 and ns.robustness == -1:
            f.write("\nfor(int i = 0; i < 2147483647; i++) {\n\n")

        else:
            if ns.unroll > -1:
                if ns.partial == -1:
                    f.write(f'\n//@ loop unroll {ns.unroll};')
            
                if input_file_length > -1 :
                    f.write(f'\nfor(int i = 0; i < {mini(ns.unroll, input_file_length)}; i++) {{\n\n')
                else :
                    f.write(f'\nfor(int i = 0; i < {ns.unroll}; i++) {{\n\n')

            else :
                f.write(f'\nfor(int i = 0; i < {input_file_length}; i++) {{\n\n')


        if ns.fraction_expansion != 'false' :
            f.write(f'long long factor = {scaler ** (scaling_exponent - 1)} * powi({scaler}, i);\n')


        # set input
        W = onnx_weights[recurrent_nodes[0].input[1]]
        for index in range(W.shape[2]):
            if ns.robustness > -1:
                f.write(f'{datatype} in_{recurrent_counter}_{index} = {variableName("Input", 0, (index, ))}[i];\n')
            else:
                interval_string = f'Frama_C_double_interval({min},{max})' if ns.fraction_expansion == 'false' else f"Frama_C_interval({min},{max})"
                f.write(f"{datatype} in_{recurrent_counter}_{index} = {interval_string};\n")


        def serialize_LSTM_Layer(n : int) :

            LSTM_node = recurrent_nodes[n]
            W = onnx_weights[LSTM_node.input[1]]
            R = onnx_weights[LSTM_node.input[2]]
            B = onnx_weights[LSTM_node.input[3]]

            iofc_seperator = int(W.shape[1] / 4)

            # assign previous output as input
            variable_input = lambda : f'in_{recurrent_counter}'
            if n > 0 :
                variable_input = lambda : f'h_{recurrent_counter}_{n-1}'

            f.write("\n")

            for gate, k in zip(["i", "o", "f", "Ct"], range(4)):
                
                weights    = W[0, k * iofc_seperator : (k+1) * iofc_seperator, :]
                recurrance = R[0, k * iofc_seperator : (k+1) * iofc_seperator, :]
                bias = B[0, k * iofc_seperator : (k+1) * iofc_seperator]

                bias_op = Scale("factor", bias) if ns.fraction_expansion != 'false' else Const(bias)

                operation = Add( Add( Mul( weights, variable_input()), Mul(recurrance, f"h_{recurrent_counter}_{n}")), bias_op)

                if k < 3 : 
                    operation = Func(operation, activations[n][0])
                else : 
                    operation = Func(operation, activations[n][1])

                f.write(serialize_operation_typed(datatype, f"{gate}_{n}", operation))


            f.write("\n")

            # calculate c_t
            operation = Add( Mul(f"f_{n}", f"c_{recurrent_counter}_{n}"), Mul(f"i_{n}", f"Ct_{n}") )
            f.write(serialize_operation(f"c_{recurrent_counter}_{n}", operation))
            
            # calculate h_t
            operation = Mul(f"o_{n}", Func(f"c_{recurrent_counter}_{n}", activations[n][2])) 
            f.write(serialize_operation(f"h_{recurrent_counter}_{n}", operation))

            f.write("\n")


        def serialize_RNN_Layer(n : int) :

            RNN_node = recurrent_nodes[n]
            W = onnx_weights[RNN_node.input[1]][0]
            R = onnx_weights[RNN_node.input[2]][0]
            B = onnx_weights[RNN_node.input[3]][0]
            B = B[0: int(B.shape[0]/2)]


            # assign previous output as input
            variable_input = lambda : f'in_{recurrent_counter}'
            if n > 0 :
                variable_input = lambda : f'h_{recurrent_counter}_{n-1}'

            f.write("\n")

            rec = variable_name("rec", n)
            operation = Mul(R, f"h_{recurrent_counter}_{n}")
            
            f.write(serialize_operation_typed(datatype, rec, operation))
            f.write("\n")

            bias_op = Scale("factor", B) if ns.fraction_expansion != 'false' else Const(B)

            operation = Func( Add( Add( rec, Mul(W, variable_input())), bias_op), activations[n][0])

            f.write(serialize_operation(f"h_{recurrent_counter}_{n}", operation))
            f.write("\n")


        for i in range(node_count) :
            if node.op_type == "LSTM":  serialize_LSTM_Layer(i)
            if node.op_type == "RNN" :  serialize_RNN_Layer(i)

        f.write("}\n\n")

        var_name = f'h_{recurrent_counter}_{node_count-1}'
        #var_name = variable_name(node.op_type, recurrent_counter)
        node_variable[output_name] = var_name
        covered_recurrents.extend([n.name for n in recurrent_nodes])

        #f.write(serialize_operation_typed(datatype, var_name, Const(f"h_{recurrent_counter}_{node_count-1}")))
        f.write("\n")

        node_count_dict[counter_type] += 1
        scaling_exponent += node_count * (input_file_length - 1)


    #else:
        # we will hit multiple nodes where we will not save anything when we need to cover LSTM cells
        
    

output_names = list()

for output in onnx_model.graph.output:
    f.write("\n")

    prev_var = node_variable[output.name]

    var_name = variable_name("Output", node_count_dict["output_counter"])
    for index in range(variable_shapes[prev_var][0]):
        output_names.append(var_name + f'_{index}')

    f.write(serialize_operation_typed(datatype, var_name, Const(prev_var)))
    f.write("\n")

    node_count_dict["output_counter"] += 1

if ns.output_only :
    f.write(f"Frama_C_show_each_output({','.join(output_names)});\n")

if ns.execute :

    def printtype() : return "%lld" if datatype == "long long" else "%f"
    f.write('printf(\"Outputs: ' + (f'{printtype()},' * len(output_names))[:-1] + '\\n\", ' + ','.join(output_names) + ');\n')

            
f.write("return 0;\n}")
f.close()

print("Conversion finished successfully!")

#########################################################
##################### Phase 2 ###########################
#########################################################
#run frama-c
#os.system("frama-c -eva converted_onnx.c > eva-result.txt")

if ns.execute :
    subprocess.run(["gcc", "converted_onnx.c", "-lm"])
    subprocess.run(["./a.out"])
    exit(0)

if ns.no_analysis:
    exit(0)

evaluation = list()
try:
    argument_list = ["frama-c", "-eva", "converted_onnx.c"]
    if ns.output_only : argument_list.insert(2, "-eva-no-print")

    evaluation = subprocess.check_output(argument_list).decode("utf-8")#.splitlines()
except subprocess.CalledProcessError as e:
    print(e)
    exit(1)

print("Frama-C analysis successful!")

if ns.verbose :
    print(evaluation)
if ns.log != "false" :
    with open(ns.log, "w+") as f:
        f.writelines(evaluation)

#parse evaluation into a dictionary
eval_dict = {}
#extraction_regex = re.compile("(?P<name>[a-zA-Z]*_[0-9]*_[0-9]*) ∈ (?P<interval>\{0\}|\[(?P<n1>-?[0-9]*\.[0-9]*) \.\. (?P<n2>-?[0-9]*\.[0-9]*)\])")
extraction_regex = re.compile("(?P<name>[a-zA-Z]*_[0-9]*_[0-9]*) ∈ ((\{(?P<interval>(0|-?[0-9]*\.[0-9]*[eE]?[\-]?[0-9]*))\})|\[(?P<n1>-?[0-9]*\.[0-9]*[eE]?[\-]?[0-9]*) \.\. (?P<n2>-?[0-9]*\.[0-9]*[eE]?[\-]?[0-9]*)\])")
match_iterator = extraction_regex.finditer(evaluation)
for token in match_iterator:
    a = token.group("interval")
    if a is not None:
        eval_dict[token.group("name")] = np.array([a,a], dtype=np.float32)
    else:
        eval_dict[token.group("name")] = np.array([token.group("n1"), token.group("n2")], dtype=np.float32)

del evaluation

# phase 3 should only commence if the converter is handling an FNN
if is_rnn or ns.robustness > -1: exit(0)

#########################################################
##################### Phase 3 ###########################
#########################################################

# we iterate over the variables defined by the previous node to the relu node
# we can then map its value range B to what kind of node its output should be sent to: 
# 1. B subset (-inf, 0] should be a constant 0 node 
# 2. B subset [0, inf) should be an identity node - specifically this case cannot be exactly determined if looking at the relu node directly
# 3. B not subset of either of the above has to be a regular ReLU node
# we additionally group together repeated nodes into one larger node for clarity and assumedly performance
# thus we replace a relu node by a split followed by a sequence of identities, ReLUs and constants that are concatenated again afterwards 
def replace_relu(relu_node): #returns new offset

    new_node_list = list()
    prev_node_name = relu_node.input[0] #relu nodes have only a single input
    prev_var = node_variable[prev_node_name]
 
    relu_indices, id_indices, const_indices = [],[],[]

    split_name= relu_node.name + "-Splitter"
    concat_name= relu_node.name + "-Concat"


    # notice : the current implementation requires shape to be a vector shape and not an array
    for s in range(variable_shapes[prev_var][0]):

        interval = eval_dict[f'{prev_var}_{s}']

        # decide which operation the value range requires
        if interval[1] <= 0.0 :
            const_indices.append(s)
        elif interval[0] >= 0.0 :
            id_indices.append(s)
        else :
            relu_indices.append(s)


    # create the new nodes
    len_id = len(id_indices)
    len_relu = len(relu_indices)
    len_const = len(const_indices)

    need_id = len_id > 0
    need_relu = len_relu > 0
    need_const = len_const > 0


    split_node = onnx.helper.make_node(
        op_type='Split',
        name=split_name,
        inputs=relu_node.input,
        outputs=[split_name + f":output-{i}" for i in range(3)],
        split=[len_id, len_relu, len_const],
        axis=0
    )

    remaining_relu_node = onnx.helper.make_node(
        op_type='Relu',
        name=split_name + ":ReLU",
        inputs=[split_name + ":output-1"],
        outputs=[concat_name + ":ReLU"]
    )

    const_node = onnx.helper.make_node(
        op_type = 'Constant',
        name =split_name + ":Const",
        inputs=[],
        outputs=[concat_name + ":Const"],
        value=onnx.helper.make_tensor('const', onnx.TensorProto.FLOAT, (len_const,), np.zeros( (len_const,), dtype=np.float32))
    )


        
    # group variables in the following sequence
    # identity, relu, constant
    # for this we use permutation matrices, eg we switch columns of the matrix before the split and rows of the matrix after the split 
    # also do not forget the bias added between the matrix and relu 

    # switch bias rows
    for prev_node in onnx_model.graph.node :
        if prev_node_name in prev_node.output: # remove numbering in input names
            break

    (t_name,) = set(prev_node.input).intersection(onnx_weights.keys())
    A = onnx_weights[t_name] # note that A is just a shorthand here
    B = onnx_weights[t_name].copy()

    if need_id : B[:len_id] = A[id_indices]  
    if need_relu : B[len_id : len_id+len_relu] = A[relu_indices]  
    if need_const : B[len_id+len_relu:] = A[const_indices] 

    onnx_weights[t_name] = B
    del B


    # switch previous matrix columns
    (mat_node_name,) = set(prev_node.input).difference(onnx_weights.keys())
    for prev_node in onnx_model.graph.node :
        if mat_node_name in prev_node.output :
            break

    (t_name,) = set(prev_node.input).intersection(onnx_weights.keys())
    A = onnx_weights[t_name] 
    B = onnx_weights[t_name].copy()

    if need_id    : B[:,0:len_id] = A[:,id_indices]  
    if need_relu  : B[:,len_id : len_id+len_relu] = A[:,relu_indices] 
    if need_const : B[:,len_id+len_relu:] = A[:,const_indices] 

    onnx_weights[t_name] = B
    del B

    # switch following rows
    next_node_name = relu_node.output[0]

    for next_node in onnx_model.graph.node :
        if next_node_name in next_node.input:
            break

    extra_node = None
    if next_node.op_type != "MatMul":
        print(relu_node.name, next_node_name, sep="\n")
        print("Shifting back via introduction of permutation matrix not currently supported")
        exit(1)


    (t_name,) = set(next_node.input).intersection(onnx_weights.keys())

    A = onnx_weights[t_name]
    B = onnx_weights[t_name].copy()
    
    if need_id    : B[0:len_id,:] = A[id_indices,:]  
    if need_relu  : B[len_id : len_id+len_relu,:] = A[relu_indices,:]  
    if need_const : B[len_id+len_relu:,:] = A[const_indices,:] 

    onnx_weights[t_name] = B
    del B


    # since we may introduce a new matrix, we can only define concat afterwards
    concat_node = onnx.helper.make_node(
        op_type='Concat',
        name=concat_name,
        axis=0,
        inputs=[split_name + ":output-0", concat_name + ":ReLU", concat_name + ":Const"],
        outputs=[next_node_name] #may have been updated
    )

    new_node_list.extend([split_node, remaining_relu_node, const_node, concat_node])
    return new_node_list


nodes = list()
for node in onnx_model.graph.node:
    if node.op_type == "Relu":
        nodes += replace_relu(node)
    else:
        nodes.append(node)


initializers = [onnx.helper.make_tensor(
    name,
    onnx.TensorProto.FLOAT,
    tensor.shape,
    tensor
) for name, tensor in onnx_weights.items()]

graph_inputs = list()
# we need to get input dimensions and delete symbolic dimensions 
for input in onnx_model.graph.input :
    input_dimensions = []
    for d in input.type.tensor_type.shape.dim:
        # the dimension may have a definite (integer) value or a symbolic identifier or neither:
        if (d.HasField("dim_value")):
            # known dimension
            input_dimensions.append(d.dim_value)

    shape = tuple(input_dimensions)
    tin = onnx.helper.make_tensor_value_info(
        input.name,
        onnx.TensorProto.FLOAT,
        shape
    )
    graph_inputs.append(tin)


graph_outputs = list()
# we need to do the same to the output for the graph to be well formed
for ouput in onnx_model.graph.output :
    output_dimensions = []
    for d in output.type.tensor_type.shape.dim:
        # the dimension may have a definite (integer) value or a symbolic identifier or neither:
        if (d.HasField("dim_value")):
            output_dimensions.append(d.dim_value)
            
    shape = tuple(output_dimensions)
    tout = onnx.helper.make_tensor_value_info(
        output.name,
        onnx.TensorProto.FLOAT,
        shape
    )
    graph_outputs.append(tout)


refactored_graph = onnx.helper.make_graph(
    nodes=nodes,
    name=model_path.name+"-refactored",
    #inputs=onnx_model.graph.input,
    inputs=graph_inputs,
    #outputs=onnx_model.graph.output,
    outputs=graph_outputs,
    initializer=initializers
)


refactored_model = onnx.helper.make_model(refactored_graph, producer_name="Onnx2C-Optimizer")
refactored_model.opset_import[0].version = 12
try:
    refactored_model = onnx.shape_inference.infer_shapes(refactored_model)
except onnx.shape_inference.InferenceError as e:
    print(e)

# Check the model
try:
    onnx.checker.check_model(refactored_model)
except onnx.checker.ValidationError as e:
    print('Converted model is invalid: %s' % e)
    exit(1)
else:
    print('Converted model is valid!')

onnx.save(refactored_model, "refactored.onnx")
print("Finished refactoring the model!")
