import torch
from torch import nn
from torchvision import transforms, models

def create_effnetb2_model(num_classes:int = 3,
                          seed:int = 42):
  """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """

  weights = models.EfficientNet_B2_Weights.DEFAULT
  model = models.efficientnet_b2(weights=weights)
  transform = weights.transforms()

  for param in model.parameters():
    param.requires_grad = True

  torch.manual_seed(seed)
  model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                   nn.Linear(in_features=1408,
                                             out_features=num_classes))

  return model, transform
