registry = {}

def register_model(name):
    def decorator(func):
        registry[name] = func
        return func
    return decorator

def getModel(name, args):
    if name in registry:
        return registry[name](**args)
    else:
        raise ValueError(f"Function '{name}' is not registered.")