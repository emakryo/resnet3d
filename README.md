# resnet3d

- `PET-CT/`: data loader & converter
- `resnet/` : original resnet implementation (https://github.com/tensorflow/models)
- `resnet3d/` : modified 3-dimensional resnet implementation


# model

- Input: 16 * 16 * 16
- Output: 2 (mean & variance)
- Loss: negative log likelihood + weight decay
