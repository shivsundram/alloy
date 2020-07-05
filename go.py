import astpretty
import ast
 
op_dict = {}
op_dict[str(ast.Add)] = "+"
op_dict[str(ast.Mult)] = "*"


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
        print(arg_string_list)
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



def make_c_prototype(kernel_name, args, kernel_returns, node_dict):    
    header = "void " + kernel_name + "("

    # add args to header
    arg_base = "float* "
    arg_string_list = ["const "+arg_base+arg for arg in args]
    arg_string_list.extend([arg_base+arg for arg in kernel_returns])
    arg_string_list.append("const int n") # add array len arg
    argstring = ", ".join(arg_string_list)

    header = header + argstring + ")"
    return header


"""
#from https://stackoverflow.com/questions/40097590/detect-whether-a-python-string-is-a-number-or-a-letter
def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True
"""
def make_c_body(schedule,args, kernel_returns, node_dict):
    pass

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
 
def convertAST(filename, kernel_name):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
        print(tree.body)
        print(tree.body[0].__dict__)
        print(tree.body[0].args.__dict__)
        kernel_args = []
        for arg1 in tree.body[0].args.args:
            print(arg1)
            kernel_args.append(arg1.id)
        ff = FuncFinder(kernel_name)
        ff.visit(tree)
        funcTree=ff.node
        print(funcTree)
        print(funcTree.__dict__)
        assigns = funcTree.body
        kernel_name = funcTree.name
        node_dict = {}
        node_dict['tmpCnt'] = 0
        pointOp = PointOp()       
        kernel_returns = []
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
        #for key,value in node_dict.items():
        #    print(key,value.__dict__)              
        #print("args", kernel_args)
        #print("returns", kernel_returns)

        # topologically sort the augmented AST 
        schedule = pointOp.make_schedule(kernel_args,kernel_returns, node_dict)
        print("schedule", schedule)

        kernel = pointOp.make_cuda_kernel(kernel_name, schedule, kernel_args,kernel_returns, node_dict)
        print("\n".join(kernel))

def main():
    import sys
    convertAST("raw2.py", "add2")
    sys.exit(0)
    with open("raw2.py", "r") as source:
        tree = ast.parse(source.read())
        print(tree.body)
        print(tree.body[0].__dict__)
        print(tree.body[0].args.__dict__)
        kernel_args = []
        for arg1 in tree.body[0].args.args:
            print(arg1)
            kernel_args.append(arg1.id)
        ff = FuncFinder('add2')
        ff.visit(tree)
        funcTree=ff.node
        print(funcTree)
        print(funcTree.__dict__)
        assigns = funcTree.body
        kernel_name = funcTree.name
        node_dict = {}
        node_dict['tmpCnt'] = 0
        pointOp = PointOp()       
        kernel_returns = []
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
        #for key,value in node_dict.items():
        #    print(key,value.__dict__)              
        #print("args", kernel_args)
        #print("returns", kernel_returns)

        # topologically sort the augmented AST 
        schedule = pointOp.make_schedule(kernel_args,kernel_returns, node_dict)
        print("schedule", schedule)

        kernel = pointOp.make_cuda_kernel(kernel_name, schedule, kernel_args,kernel_returns, node_dict)
        print("\n".join(kernel))
main()    
