import paddle
from paddle import nn
from paddle.nn import functional as F


class Attention(nn.Layer):
    def __init__(self, in_planes, ratio, K, temperature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.temperature = temperature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2D(in_planes, hidden_planes, kernel_size=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(hidden_planes, K, kernel_size=1, bias_attr=False)
        )

        if init_weight:
            self._initialize_weights()

    def update_temperature(self):
        if self.temperature > 1:
            self.temperature -= 1

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal(nonlinearity='relu')(m.weight)
                if m.bias is not None:
                    nn.initializer.Constant(0)(m.bias)
            if isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(1)(m.weight)
                nn.initializer.Constant(0)(m.bias)

    def forward(self, x):
        att = self.avgpool(x)
        att = self.net(att)
        att = paddle.reshape(att, [x.shape[0], -1])
        return F.softmax(att / self.temperature, axis=-1)


class DynamicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias_attr=True, K=4,
                 temperature=40, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_attr = bias_attr
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temperature=temperature, init_weight=init_weight)

        # Create weight parameter
        self.weight = paddle.create_parameter(
            shape=[K, out_planes, in_planes // groups, kernel_size, kernel_size],
            dtype='float32',
            default_initializer=nn.initializer.KaimingNormal()
        )

        # Optionally create bias parameter
        if bias_attr:
            self.bias = paddle.create_parameter(
                shape=[K, out_planes],
                dtype='float32',
                default_initializer=nn.initializer.Constant(0)
            )
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.initializer.KaimingUniform()(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        bs, in_planes, h, w = x.shape
        softmax_att = self.attention(x)
        x = paddle.reshape(x, [1, -1, h, w])
        weight = paddle.reshape(self.weight, [self.K, -1])
        aggregate_weight = paddle.mm(softmax_att, weight).reshape([bs * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size])

        if self.bias is not None:
            bias = paddle.reshape(self.bias, [self.K, -1])
            aggregate_bias = paddle.mm(softmax_att, bias).reshape([-1])
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)

        output = paddle.reshape(output, [bs, self.out_planes, h, w])
        return output


if __name__ == '__main__':
    input = paddle.randn(2, 32, 64, 64)
    m = DynamicConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias_attr=False)
    out = m(input)
    print(out.shape)
