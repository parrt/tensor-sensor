import tsensor

# Parse tree definitions
# I found package ast in python3 lib after I built this. whoops. No biggie.
# This tree structure is easier to visit for my purposes here. Also lets me
# control the kinds of statements I process.

class ParseTreeNode:
    def __init__(self):
        self.value = None # used during evaluation
        # self.start = 0 # start token index  UNUSED
        # self.len = 0   # how many tokens?   UNUSED
    def eval(self, frame):
        """
        Evaluate the expression represented by this (sub)tree in context of frame.
        Try any exception found while evaluating and remember which operation that
        was in this tree
        """
        try:
            self.value = eval(str(self), frame.f_locals, frame.f_globals)
        except:
            raise IncrEvalTrap(self)
        # print(self, "=>", self.value)
        return self.value
    @property
    def opstr(self): # the associated token if atom or representative token if operation
        return None
    @property
    def kids(self):
        return []
    def explain(self):
        return None
    def __str__(self):
        pass
    def __repr__(self):
        args = [v+'='+self.__dict__[v].__repr__() for v in self.__dict__ if v!='value' or self.__dict__['value'] is not None]
        args = ','.join(args)
        return f"{self.__class__.__name__}({args})"

class Assign(ParseTreeNode):
    def __init__(self, op, lhs, rhs):
        super().__init__()
        self.op, self.lhs, self.rhs = op, lhs, rhs
    def eval(self, frame):
        self.value = self.rhs.eval(frame)
        self.lhs.value = self.value
        return self.value
    @property
    def opstr(self):
        return self.op.value
    @property
    def kids(self):
        return [self.lhs, self.rhs]
    def __str__(self):
        return str(self.lhs)+'='+str(self.rhs)

class Call(ParseTreeNode):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args
    def eval(self, frame):
        self.func.eval(frame)
        for a in self.args:
            a.eval(frame)
        return super().eval(frame)
    def explain(self):
        arg_msgs = []
        for a in self.args:
            ashape = tsensor.analysis._shape(a.value)
            if ashape:
                arg_msgs.append(f"arg {a} w/shape {ashape}")
        if len(arg_msgs)==0:
            return f"Cause: {self}"
        return f"Cause: {self} tensor " + ', '.join(arg_msgs)
    @property
    def opstr(self):
        fname = str(self.func)
        if isinstance(self.func, Member):
            fname = str(self.func.member)
        return fname+"()"
    @property
    def kids(self):
        return [self.func]+self.args
    def __str__(self):
        args_ = ','.join([str(a) for a in self.args])
        return f"{self.func}({args_})"

class Index(ParseTreeNode):
    def __init__(self, arr, index):
        super().__init__()
        self.arr = arr
        self.index = index
    def eval(self, frame):
        self.arr.eval(frame)
        for i in self.index:
            i.eval(frame)
        return super().eval(frame)
    @property
    def kids(self):
        return [self.arr] + self.index
    def __str__(self):
        i = self.index
        i = ','.join(str(v) for v in i)
        return f"{self.arr}[{i}]"

class Member(ParseTreeNode):
    def __init__(self, op, obj, member):
        super().__init__()
        self.op = op # always DOT
        self.obj = obj
        self.member = member
    def eval(self, frame):
        self.obj.eval(frame)
        # don't eval member as it's just a name to look up in obj
        return super().eval(frame)
    @property
    def opstr(self): # the associated token if atom or representative token if operation
        return self.op.value
    @property
    def kids(self):
        return [self.obj, self.member]
    def __str__(self):
        return f"{self.obj}.{self.member}"

class BinaryOp(ParseTreeNode):
    def __init__(self, op, lhs, rhs):
        super().__init__()
        self.op, self.lhs, self.rhs = op, lhs, rhs
    def eval(self, frame):
        self.lhs.eval(frame)
        self.rhs.eval(frame)
        return super().eval(frame)
    def explain(self):
        opnd_msgs = []
        lshape = tsensor.analysis._shape(self.lhs.value)
        rshape = tsensor.analysis._shape(self.rhs.value)
        if lshape:
            opnd_msgs.append(f"operand {self.lhs} w/shape {lshape}")
        if rshape:
            opnd_msgs.append(f"operand {self.rhs} w/shape {rshape}")
        return f"Cause: {self.op} on tensor " + ' and '.join(opnd_msgs)
    @property
    def opstr(self): # the associated token if atom or representative token if operation
        return self.op.value
    @property
    def kids(self):
        return [self.lhs, self.rhs]
    def __str__(self):
        return f"{self.lhs}{self.op}{self.rhs}"

class UnaryOp(ParseTreeNode):
    def __init__(self, op, opnd):
        super().__init__()
        self.op = op
        self.opnd = opnd
    def eval(self, frame):
        self.opnd.eval(frame)
        return super().eval(frame)
    @property
    def opstr(self):
        return self.op.value
    @property
    def kids(self):
        return [self.opnd]
    def __str__(self):
        return f"{self.op}{self.opnd}"

class ListLiteral(ParseTreeNode):
    def __init__(self, elems):
        super().__init__()
        self.elems = elems
    def eval(self, frame):
        for i in self.elems:
            i.eval(frame)
        return super().eval(frame)
    @property
    def kids(self):
        return self.elems
    def __str__(self):
        if isinstance(self.elems,list):
            elems_ = ','.join(str(e) for e in self.elems)
        else:
            elems_ = self.elems
        return f"[{elems_}]"

class SubExpr(ParseTreeNode):
    # record parens for later display to keep precedence
    def __init__(self, e):
        super().__init__()
        self.e = e
    def eval(self, frame):
        self.e.eval(frame)
        return self.e.value # don't re-evaluate
    @property
    def kids(self):
        return [self.e]
    def __str__(self):
        return f"({self.e})"

class Atom(ParseTreeNode):
    def __init__(self, token):
        super().__init__()
        self.token = token
    @property
    def opstr(self):
        return self.token.value
    def __repr__(self):
        v = f"{{{self.value}}}" if hasattr(self,'value') and self.value is not None else ""
        return self.token.value + v
    def __str__(self):
        return self.token.value


def postorder(t):
    nodes = []
    _postorder(t, nodes)
    return nodes


def _postorder(t, nodes):
    if t is None:
        return
    for sub in t.kids:
        _postorder(sub, nodes)
    nodes.append(t)


def leaves(t):
    nodes = []
    _leaves(t, nodes)
    return nodes


def _leaves(t, nodes):
    if t is None:
        return
    if len(t.kids) == 0:
        nodes.append(t)
        return
    for sub in t.kids:
        _leaves(sub, nodes)


def walk(t, pre=lambda x: None, post=lambda x: None):
    if t is None:
        return
    pre(t)
    for sub in t.kids:
        walk(sub, pre, post)
    post(t)


class IncrEvalTrap(BaseException):
    """
    Used during re-evaluation of python line that threw exception to trap which
    subexpression caused the problem.
    """
    def __init__(self, offending_expr):
        self.offending_expr = offending_expr # where in tree did we get exception?
