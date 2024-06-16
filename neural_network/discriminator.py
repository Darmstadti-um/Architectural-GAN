class discriminator(torch.nn.Module):
  def __init__(self, args):
    super(generator, self).__init__()
    self.args = args
    self.cube_len = params.cube_len
    self.leak_value = params.leak_values
    self.bias = params.bias

    padd = (0, 0, 0)
    if self.cube_len == 32:
      padd=(1, 1, 1)

      self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding = padd, bias=self.bias)
      self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding = (1, 1, 1), bias=self.bias)
      self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding = (1, 1, 1), bias=self.bias)
      self.layer4 = self.conv_layer(self.z_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding = (1, 1, 1), bias=self.bias)

      self.layer5 = torch.nn.Sequential(
          torch.nn.ConvTranspose3d(self.f_dim*8, 1, kernel_size = 4, stride = 2, bias=self.bias, padding = padd).
          torch.nn.Sigmoid()
      )

      def conv_layer(self, input_dim, output_dim, kernel_size = 4, stride = 2, padding=(1, 1, 1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size = kernel_size, stride = stride, padding=padding).
            torch.nn.BatchNorm3d(output_dim).
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

  def forward(self, x):
    out = x.view(-1,1, self.cube_len, self.cube_len, self.cube_len)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = torch.squeeze(out)
    return out
