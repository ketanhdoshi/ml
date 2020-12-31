# Lambda functions are ad hoc functions that are comprised of a single statement
# Same as def funcvar(x): return x + 1
funcvar = lambda x: x + 1
print(funcvar(1))

# an_int and a_string are optional, they have default values
# if one is not passed (2 and "A default string", respectively).
def passing_example(a_list, an_int=2, a_string="A default string"):
    a_list.append("A new item")
    an_int = 4
    return a_list, an_int, a_string

# Parameters are passed by reference, but immutable 
# types (tuples, ints, strings, etc) cannot be changed in the caller by the callee. 
my_list = [1, 2, 3]
my_int = 10
print(passing_example(my_list, my_int))
print(my_list, my_int)

# Exceptions are handled with try-except [exceptionname] blocks
def some_function():
    try:
        # Division by zero raises an exception
        10 / 0
    except ZeroDivisionError:
        print("Oops, invalid.")
    else:
        # Exception didn't occur, we're good.
        pass
    finally:
        # This is executed after the code block is run
        # and all exceptions have been handled, even
        # if a new exception is raised while handling.
        print("We're done with that.")

some_function()