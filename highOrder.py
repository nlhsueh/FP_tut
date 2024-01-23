
'''
    function as parameter
'''
def apply(op, x, y):
    return op(x, y)

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

d1, d2 = 3, 5
ops = [add, subtract]

for op in ops:
    r = apply(op, d1, d2)
    print (r)

# ------

'''
    function as tag
'''

def log_before(f):
    def _wrapper(*args, **kwargs):
        print("logging before...")
        f(*args, **kwargs)

    return _wrapper


def log_after(f):
    def _wrapper(*args, **kwargs):
        f(*args, **kwargs)
        print("logging after...")

    return _wrapper


@log_before
@log_after
def greet(name: str):
    print(f"Hello {name}")


# with the log_before and log_after decorators
# the following is semantically equivalent to
# `log_before(log_after(greet))("John")`
greet("John")

# -------