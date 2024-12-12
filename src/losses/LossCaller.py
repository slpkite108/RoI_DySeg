import monai.losses

custom_losses = {}

def register_custom_loss(name):
    def decorator(loss_class):
        custom_losses[name] = loss_class
        return loss_class
    return decorator

def LossCaller(name, args):
    loss_class = getattr(monai.losses, name, None)
    
    if loss_class is None:
        loss_class = custom_losses.get(name, None)
    
    if loss_class is None:
        raise ValueError(f" Loss function '{name}' is not available.")
    
    return loss_class(**args)
    