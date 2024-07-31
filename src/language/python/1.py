class VGG19(object):
    def __init__(self):
        # set up net
        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
        )

        # conv: channel_out, kernel_size, stride, 1, padding, (1 means dilation)
        # pool: kernel_size, stride
        self.layers_parameters = (
            [64, 3, 1, 1, 1], [], [64, 3, 1, 1, 1], [], [2, 2],
            [128, 3, 1, 1, 1], [], [128, 3, 1, 1, 1], [], [2, 2],
            [256, 3, 1, 1, 1], [], [256, 3, 1, 1, 1], [], [256, 3, 1, 1, 1], [], [256, 3, 1, 1, 1], [], [2, 2],
            [512, 3, 1, 1, 1], [], [512, 3, 1, 1, 1], [], [512, 3, 1, 1, 1], [], [512, 3, 1, 1, 1], [], [2, 2],
            [512, 3, 1, 1, 1], [], [512, 3, 1, 1, 1], [], [512, 3, 1, 1, 1], [], [512, 3, 1, 1, 1], [], [2, 2],
            [[512, 7, 7], [512 * 7 * 7]], [25088, 4096], [], [4096, 4096], [], [4096, 1000], []
        )
        self.input_shapes = []

    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path

        # compute input_shape
        for i in range(len(self.param_layer_name)):
            # N, channel input, height, width
            # input_shape = pycnnl.IntVector(4)
            print(i)
            if i == 0:
                input_shape = [1, 3, 224, 224]
            else:
                last_layer_input_shape = self.input_shapes[i - 1]
                last_layer_parameters = self.layers_parameters[i - 1]
                if 'conv' in self.param_layer_name[i - 1]:
                    input_shape = [1, last_layer_parameters[0], (last_layer_input_shape[2] + 2 * last_layer_parameters[4] - last_layer_parameters[1]) // last_layer_parameters[2] + 1, (last_layer_input_shape[3] + 2 * last_layer_parameters[4] - last_layer_parameters[1]) // last_layer_parameters[2] + 1]
                elif 'relu' in self.param_layer_name[i - 1]:
                    input_shape = last_layer_input_shape
                elif 'pool' in self.param_layer_name[i - 1]:
                    input_shape = [1, last_layer_input_shape[1], (last_layer_input_shape[2] - last_layer_parameters[0]) // last_layer_parameters[1] + 1, (last_layer_input_shape[3] - last_layer_parameters[0]) // last_layer_parameters[1] + 1]
                elif 'flatten' in self.param_layer_name[i - 1]:
                    input_shape = [1, 1, 1, last_layer_input_shape[1] * last_layer_input_shape[2] * last_layer_input_shape[ 3]]  # assert 512*7*7 = 25088
                elif 'fc' in self.param_layer_name[i - 1]:
                    if 'softmax' in self.param_layer_name[i]:
                        input_shape = [1, 1, last_layer_parameters[1]]
                    else:
                        input_shape = [1, 1, 1, last_layer_parameters[1]]
                else:
                    raise ValueError("Unknown layer type for layer: {}".format(self.param_layer_name[i - 1]))
            self.input_shapes.append(input_shape)

if __name__ == "__main__":
    vgg19 = VGG19()
    vgg19.build_model()
    print(vgg19.input_shapes)