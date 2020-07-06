import astpretty
import array
import ctypes
import numpy as np
import ast
import inspect 
import subprocess

op_dict = {}
op_dict[str(ast.Add)] = "+"
op_dict[str(ast.Mult)] = "*"


class Node():
    def __init__(self):
        self.name=None
        self.op=None
        self.parent=None
        self.left=None
        self.right=None


class PointOp():

    # makes cuda kernel based on schedule and kernel info
    def make_cuda_kernel(self, kernel_name, schedule, args, kernel_returns, node_dict):

        #make function prototype
        header = self.make_cuda_kernel_prototype(kernel_name, args, kernel_returns, node_dict)

        # use schedule to make body of kernel, fusing all point ops
        body = self.make_cuda_kernel_body(schedule,args, kernel_returns, node_dict)

        # append header to body
        kernel = [header]+body
        return kernel


    def make_cuda_kernel_prototype(self,kernel_name, args, kernel_returns, node_dict):    
        header = "__global__ void " + kernel_name + "("

        # add args to header
        arg_base = "float* "
        arg_string_list = ["const "+arg_base+arg for arg in args]
        arg_string_list.extend([arg_base+arg for arg in kernel_returns])
        arg_string_list.append("const int n") # add array len arg
        argstring = ", ".join(arg_string_list)

        header = header + argstring + ")"
        return header


    def make_cuda_kernel_body(self, schedule,args, kernel_returns, node_dict):
        body = []
        body.append("{")
        body.append("\tint i = blockDim.x*blockIdx.x+threadIdx.x;")
        body.append("\tif (i>=n){continue;}")

        # make variable mappings for kernel fusion
        fusion_map = {}
        for arg in args+kernel_returns:
            fusion_map[arg] = arg+"_i"
        def fuse(elem):
            return fusion_map[elem] if (elem in fusion_map) else elem
        
        # grap inputs
        for arg in args:
            body.append("\tconst float "+ arg+"_i = " + arg+"[i];")
     
        # do computations
        for element in schedule:
            node = node_dict[element]
            registers = map(str, [element, node.left, node.right])
            result, left, right = map(fuse, registers)
            line = ["\tfloat", result, "=", left, op_dict[str(type(node.op))], right]
            line = " ".join(line) +";"
            body.append(line)

        # commit results
        for arg in kernel_returns:
            body.append("\t"+ arg+"[i] = "+arg+"_i;")

        body.append("}")
        return body


    # topologically sort the augmented operation ast to flatten
    def make_schedule(self, kernel_args,kernel_returns, node_dict):
        stack = []
        visited = {}
        for key,value in node_dict.items():
            visited[key]=False
        #print(visited.items())        

        def schedule_visit(key, node_dict, visited):
            if visited[key]:
                return
            visited[key] = True
            left = node_dict[key].left
            right = node_dict[key].right
            if  left in node_dict and (not visited[left]):
                schedule_visit(left, node_dict, visited)  
            if  right in node_dict and (not visited[right]):
                schedule_visit(right, node_dict, visited)  
            stack.append(key)

        for key,value in node_dict.items():
            if visited[key]:
                continue
            schedule_visit(key, node_dict, visited)
       
        #print(stack)
        return stack  


    def make_c_kernel(self, kernel_name, schedule, args, kernel_returns, node_dict):

        #make function prototype
        header = self.make_c_prototype(kernel_name, args, kernel_returns, node_dict)
        print(header)
        # use schedule to make body of kernel, fusing all point ops
        body = self.make_c_body(schedule,args, kernel_returns, node_dict)
        # append header to body
        kernel = [header]+body
        return header, kernel


    def make_c_prototype(self, kernel_name, args, kernel_returns, node_dict):    
        header = "void " + kernel_name + "("

        # add args to header
        arg_base = "float* "
        arg_string_list = [arg_base+arg for arg in args]
        arg_string_list.extend([arg_base+arg for arg in kernel_returns])
        arg_string_list.append("int n") # add array len arg
        argstring = ", ".join(arg_string_list)
    
        header = header + argstring + ")"
        print(header)
        return header


    def make_c_body(self, schedule,args, kernel_returns, node_dict):
        body = []
        body.append("{")
        body.append("\tfor(int i =0; i<n; i++)")
        body.append("\t{")

        # make variable mappings for kernel fusion
        fusion_map = {}
        for arg in args+kernel_returns:
            fusion_map[arg] = arg+"_i"
        def fuse(elem):
            return fusion_map[elem] if (elem in fusion_map) else elem
        
        # grap inputs
        for arg in args:
            body.append("\t\tconst float "+ arg+"_i = " + arg+"[i];")
     
        # do computations
        for element in schedule:
            node = node_dict[element]
            registers = map(str, [element, node.left, node.right])
            result, left, right = map(fuse, registers)
            line = ["\t\tfloat", result, "=", left, op_dict[str(type(node.op))], right]
            line = " ".join(line) +";"
            body.append(line)
        #body.append("{")
        # commit results
        for arg in kernel_returns:
            body.append("\t\t"+ arg+"[i] = "+arg+"_i;")

        body.append("\t}")
        body.append("}")
        print("\n".join(body))
        return body


def getFuncFromTree(tree, funcName):

    class FuncFinder(ast.NodeVisitor):
        def __init__(self, node_name):
            self.node_name=node_name
            self.node=None

        def visit_FunctionDef(self, node):
            if(node.name == self.node_name):
                self.node=node
                print("i found "+ self.node_name)
                #astpretty.pprint(node)
                self.generic_visit(node)

    #start function
    ff = FuncFinder(funcName)
    ff.visit(tree)
    funcTree=ff.node
    return funcTree


# recursively parse binary expressions
def parseBinOp(graph_node, node, node_dict):
    if isinstance(node.left, ast.BinOp):
        left_node = Node() 
        left_node.name = "temp"+str(node_dict['tmpCnt'])
        node_dict['tmpCnt'] = node_dict['tmpCnt'] +1
        left_node.op = node.left.op
        node_dict[left_node.name] = left_node
        graph_node.left = left_node.name
        parseBinOp(left_node,node.left,node_dict)
    else:
        if isinstance(node.left, ast.Name):
            graph_node.left = node.left.id
        elif isinstance(node.left, ast.Num):
            graph_node.left = node.left.n
 
    if isinstance(node.right, ast.BinOp):
        right_node = Node() 
        right_node.name = "temp"+str(node_dict['tmpCnt'])
        node_dict['tmpCnt'] = node_dict['tmpCnt'] +1
        right_node.op = node.right.op
        node_dict[right_node.name] = right_node
        graph_node.right = right_node.name
        parseBinOp(right_node,node.right,node_dict)
    else:
        if isinstance(node.right, ast.Name):
            graph_node.right = node.right.id
        elif isinstance(node.right, ast.Num):
            graph_node.right = node.right.n
 

def parseAST(tree):
    #make list of each function argument's name
    kernel_args = [arg1.id for arg1 in tree.args.args]
    node_dict = {}
    node_dict['tmpCnt'] = 0
    kernel_returns = []

    assigns = tree.body
    # iterate over each lines/expression, launch recursive binary parser if necessary
    for node in assigns:
        if isinstance(node, ast.Assign):
            graph_node = Node()
            graph_node.name = node.targets[0].id
            graph_node.op = node.value.op
            if (isinstance(node.value, ast.BinOp)):
                parseBinOp(graph_node,node.value, node_dict)
            node_dict[node.targets[0].id] = graph_node
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                kernel_returns.append(node.value.id)
            elif isinstance(node.value, ast.Tuple):
                for return_val in node.value.elts:
                    kernel_returns.append(return_val.id)
    del node_dict['tmpCnt'] # undo dumb hack to make tmpCnt global scope
    # topologically sort the augmented AST 
    return kernel_args, kernel_returns, node_dict


def convertFuncFromTree(tree, kernel_name):

    # seek to desired function node in AST (ie func we want to convert to native)
    funcTree=getFuncFromTree(tree, kernel_name)

    # get kernel arguments, returns, and concise compute dag from AST
    kernel_args, kernel_returns, node_dict = parseAST(funcTree)

    # topologically sort the compute dag to make a flattened compute schedule
    pointOp = PointOp()       
    schedule = pointOp.make_schedule(kernel_args,kernel_returns, node_dict)

    # use flattened compute schedule and kernel args&return lists to make a native kernel string
    kernel = pointOp.make_cuda_kernel(kernel_name, schedule, kernel_args,kernel_returns, node_dict)
    header, kernel = pointOp.make_c_kernel(kernel_name, schedule, kernel_args,kernel_returns, node_dict)
    print("\n".join(kernel))
    return header, kernel

    

def convertFuncFromFile(filename, kernel_name):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
        convertFuncFromTree(tree, kernel_name)

  
def convertFunc(foo):
    source = inspect.getsource(foo)
    tree = ast.parse(source)
    kernel_name = tree.body[0].name   
    header, kernel = convertFuncFromTree(tree, kernel_name)
    import os
    with open(kernel_name+"GEN"+'.c', 'w') as fp: 
        fp.write("\n".join(kernel))
  

def convertFunc2Native(foo):
    source = inspect.getsource(foo)
    tree = ast.parse(source)
    kernel_name = tree.body[0].name   
    header, kernel = convertFuncFromTree(tree, kernel_name)
    import os
    with open(kernel_name+"GEN"+'.c', 'w') as fp: 
        fp.write("\n".join(kernel))
  
    name = kernel_name+"GEN"     
    cmd = ["/usr/local/bin/gcc", "-O2", "-c", "-Wall", "-Werror", "-fpic", kernel_name+"GEN"+'.c']  
    p = subprocess.call(cmd);  
    print(name)
    cmd = ["/usr/local/bin/gcc", "-shared", "-o", name+".so", name+".o"]
    p = subprocess.call(cmd);  

    ft = np.ctypeslib.ndpointer(dtype=np.float32)
    mylib = ctypes.CDLL( name+".so")
    mylib.add2.argtypes = [ft, ft, ft,ft, ctypes.c_int32]
    return mylib.add2


def add2(a, b):
    c = 2*(2*a)+1*b
    d = c*a
    e = c+b
    return d,e 


def main():
    #convertFuncFromFile("raw2.py", "add2")
    a = np.array([1.0, 2., 3., 4., 5.], dtype='float32')
    b = np.array([1., 2., 3., 4., 5.],dtype='float32')
    d = np.array([0., 0., 0., 0., 0.],dtype='float32')
    e = np.array([0., 0., 0., 0., 0.],dtype='float32')

    d1 = np.array([0., 0., 0., 0., 0.],dtype='float32')
    e1 = np.array([0., 0., 0., 0., 0.],dtype='float32')

    add2Native = convertFunc2Native(add2) 
    add2Native(a,b,d,e,5)

    d1, e1 = add2(a,b)
    print(d1)

    print(d)
main()    

