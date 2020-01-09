from . import interpreter


def query(func):
    def wrapped(dataset, verbose=False):
        return interpreter.run(func.__doc__, dataset, verbose=verbose)
    return wrapped
