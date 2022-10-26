from collections import  defaultdict
from typing import Union
import numpy as np
import re

variable_shapes = defaultdict(lambda : None)

class Op :
    def __init__(self, a_var : Union[np.ndarray, "Op", str], b_var : Union[np.ndarray, "Op", str]) -> None:
        self.a = a_var
        self.b = b_var
        self.shape = None

        if isinstance(b_var, np.ndarray) :
            if isinstance(a_var, np.ndarray) :
                del self
                raise Exception("Op Object constructor does not take two numpy arrays. Please calculate operation manually instead.")

            self.shape = b_var.shape
            if b_var.ndim > 1 :
                del self
                raise Exception(f"Op Object constructor expects the second variable, if given as an array, to be of the form (x,). Instead it received {b_var.shape}.")

        elif isinstance(b_var, Op) :
            self.shape = b_var.shape

        elif isinstance(b_var, str) :
            self.shape = variable_shapes[b_var]

    def unravel(self) -> np.ndarray :

        if isinstance(self.a, Op) :
            self.a = self.a.unravel()
        if isinstance(self.b, Op) :
            self.b = self.b.unravel()

        return None

        

class Add (Op) :
    def __init__(self, a_var : Union[np.ndarray, "Op", str], b_var : Union[np.ndarray, "Op", str]) -> None:
        super().__init__(a_var, b_var)

        a_shape = None
        b_shape = self.shape

        if isinstance(a_var, np.ndarray) :
            a_shape = a_var.shape
            self.shape = a_shape

        elif isinstance(a_var, Op) :
            a_shape = a_var.shape
            self.shape =  a_shape

        elif isinstance(a_var, str) :
            a_shape = variable_shapes[a_var]
            if a_shape is None and b_shape is not None :
                variable_shapes[a_var] = b_shape

        if isinstance(b_var, str) and b_shape is None :
            variable_shapes[b_var] = a_shape
            self.shape = a_shape
        

        if a_shape is not None and b_shape is not None and a_shape != b_shape :
            del self
            raise Exception(f"Mismatch in shapes: a:{a_shape}, b:{b_shape}")

        if self.shape is None:
            del self
            raise Exception(f"Add: Please provide some shape hints, perhaps by changing operator order.")

    def unravel(self) -> np.ndarray :
        
        super().unravel()

        get_a_value = (lambda x, a=self.a : a + f'_{x}') if isinstance(self.a, str) else lambda x, a=self.a : a[x] 
        get_b_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x] 


        return np.array([f'{get_a_value(x)} + {get_b_value(x)}' for x in range(self.shape[0])])        


class Mul (Op) :
    def __init__(self, a_var : Union[np.ndarray, "Op", str], b_var : Union[np.ndarray, "Op", str]) -> None:
        super().__init__(a_var, b_var)

        a_shape = None
        b_shape = self.shape

        if isinstance(a_var, np.ndarray):
            a_shape = a_var.shape
            self.shape = (a_shape[0],) 

        elif isinstance(a_var, Op) :
            a_shape = a_var.shape
            self.shape = a_shape

        elif isinstance(a_var, str) :
            a_shape = variable_shapes[a_var]
            if a_shape is None and b_shape is not None : # we presume this only happens for component wise vector products
                variable_shapes[a_var] = b_shape

        if isinstance(b_var, str) and b_shape is None :
            variable_shapes[b_var] = (a_shape[-1],)
            self.shape = (a_shape[0],)

        if a_shape is None :
            del self
            raise Exception("Please always provide a matrix or operation with known shape as the first variable for Mul.")
        

        if a_shape is not None and b_shape is not None and a_shape[-1] != b_shape[0] :
            del self
            raise Exception(f"Shapes incompatible for matrix/vector-component multiplication: a:{a_shape}, b:{b_shape}")


    def unravel(self) -> np.ndarray :
        
        super().unravel()

        get_b_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x]
 
        if isinstance(self.a, str):
            get_a_value = lambda x, a=self.a : a + f'_{x}'
            return np.array([f'{get_a_value(x)} * {get_b_value(x)}' for x in range(self.shape[0])])
        else:
            return np.array([ " + ".join([f'{self.a[x,y]} * {get_b_value(y)}' for y in range(self.a.shape[-1])]) for x in range(self.shape[0])])

# importantly, scale does not remember shape information, since the same scale is used everywhere
class Scale (Op) :
    def __init__(self, a_var: Union[float, str], b_var: Union["Op", str]) -> None:
        super().__init__(a_var, b_var)

        if self.shape is None :
            del self
            raise Exception(f"Scale: Please provide some shape hints, perhaps by changing operator order.")

    def unravel(self) -> np.ndarray:

        super().unravel()

        get_b_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x] 

        if self.a == 1 :
            return np.array([f'{get_b_value(x)}'] for x in range(self.shape[0]))
        else :
            res = [f'{self.a} * {get_b_value(x)}' for x in range(self.shape[0])]
            res = [re.sub(r'(?P<factor>.* \* )-', r'-\1', x) for x in res]
            res = [re.sub(r'--', '', x) for x in res]
            return np.array(res)

class Div (Op) :
    def __init__(self, a_var: Union["Op", str], b_var: float) -> None:
        super().__init__(b_var, a_var)  # note we are switching the variables to reuse some Op code

        if self.shape is None :
            del self
            raise Exception(f"Scale: Please provide some shape hints, perhaps by changing operator order.")

    def unravel(self) -> np.ndarray:

        super().unravel()

        get_b_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x] 

        if self.a == 1 :
            return np.array([f'{get_b_value(x)}' for x in range(self.shape[0])])
        else :
            return np.array([f'{get_b_value(x)} / {self.a}' for x in range(self.shape[0])])

class Const (Op) :
    def __init__(self, a_var: Union[np.ndarray, "Op", str], b_var = None) -> None:
        
        super().__init__(None, a_var)

        if self.shape is None :
            del self
            raise Exception(f"Const: Please provide some shape hints, perhaps by changing operator order.")

    def unravel(self) -> np.ndarray:
        
        super().unravel()

        get_a_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x]
        return np.array([f'{get_a_value(x)}' for x in range(self.shape[0])])

class Eq (Op) :
    def __init__(self, a_var: str, b_var: Union[np.ndarray, "Op", str]) -> None:
        super().__init__(a_var, b_var)

        #if variable_shapes[a_var] is not None :
        #    del self
        #    raise Exception(f"Trying to assign variable {a_var} more than once.")

        if isinstance(b_var, str) :
            self.shape = variable_shapes[b_var]
        else :
            self.shape = b_var.shape

        if self.shape is None :
            del self
            raise Exception(f"Eq: Please provide some shape hints, perhaps by changing operator order.")

        variable_shapes[a_var] = self.shape

    def unravel(self) -> np.ndarray:

        super().unravel()

        get_b_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x] 
        
        return np.array([f'{self.a}_{x} = {get_b_value(x)}' for x in range(self.shape[0])])

class Func (Op):
    def __init__(self, a_var: Union["Op", str], func: str) -> None:
        super().__init__(None, a_var)

        self.func = func

        if self.shape is None :
            del self
            raise Exception(f"Func: Please provide some shape hints, perhaps by changing operator order.")

    def unravel(self) -> np.ndarray:
        
        super().unravel()

        get_a_value = (lambda x, b=self.b : b + f'_{x}') if isinstance(self.b, str) else lambda x, b=self.b : b[x]

        return np.array([f'{self.func}({get_a_value(x)})' for x in range(self.shape[0])])


def serialize_operation(new_var_name : str, operation : Op) -> str :
    return serialize_operation_typed("", new_var_name, operation)

def serialize_operation_typed(datatype : str, new_var_name : str, operation : Op) -> str :
    l = Eq(new_var_name, operation).unravel()
    l = [datatype + " " + stri for stri in l]
    s = ';\n'.join(l)
    s = re.sub('\+ -', '- ', s)
    s += ";\n"
    return s

