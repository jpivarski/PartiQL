from . import interpreter


def query(func):
    def wrapped(dataset):
        return interpreter.run(func.__doc__, dataset)
    return wrapped
