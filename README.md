# Onnx2C-Converter

This python script has been developed in the scope of the bachelor's thesis "Verifying Recurrent Neural Networks using Frameworks for Static Program Analysis".
It can convert simple feed foreward and recurrent neural networks to equivalent C code. 
It also equips it with the necessary constructs to perform a value analysis using the EVA plugin of Frama-C. 

### Dependencies

The script directly depends on the python packages:
- numpy
- onnx

Additionally, the program can produce the neural network code without the parts used specific to the value analysis using the command-line option --execute.
If one were to use all features of the script, the additional dependency would be
- Frama-C

### Operations

To aid in further development, the basic operations performed within neural networks can be expressed using the Operator classes
- Add (vector vector)
- Mul (Matrix vector, or componentwiese vector vector)
- Eq
- Scale
- Div
