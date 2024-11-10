registry = {}

def register_model(name):
    def decorator(func):
        registry[name] = func
        return func
    return decorator

def getModel(name, *args, **kwargs):
    if name in registry:
        return registry[name](*args, **kwargs)
    else:
        raise ValueError(f"Function '{name}' is not registered.")