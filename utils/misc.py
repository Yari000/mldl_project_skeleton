# Just a function to count the number of parameters
def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module

  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define the device to use: use the gpu runtime if possible!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
