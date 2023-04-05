import torch.nn as nn

class tiny_VGG(nn.Module):
    def __init__(self,num_class):
        super(tiny_VGG, self).__init__()
        self.layer_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10,10,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layer_2=nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classify=nn.Sequential(
            nn.Flatten(),
            nn.Linear(42135,num_class)
        )

    def forward(self,x):
        out=self.layer_1(x)
        out=self.layer_2(out)
        out=self.classify(out)
        return out

class IOHook:
    """
    This hook attached to a nn.Module, it supports both backward and forward hooks.
    This hook provides access to feature maps from a nn.Module, it can be either
    the input feature maps to that nn.Module, or output feature maps from nn.Module
    Args:
        module (nn.Module): a nn.Module object to attach the hook
        backward (bool): get feature maps in backward phase
    """
    def __init__(self, module: nn.Module, backward=False):
        self.backward = backward
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()



