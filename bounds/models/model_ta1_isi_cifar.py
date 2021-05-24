"""
Copyright Systems & Technology Research 2020-2021

This module is an interface for the ISI Cifar model to run bound computation
"""
import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import torchvision
from bounds.ta1_eval.mt_net import MT_CNN
import os
from bounds.models import Model, DIR_NAME
import sys
import logging

log = logging.getLogger("str_lwll")


class IsiCifar(Model):
    """A wrapper class for the ISI cifar model"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, checkpoint, data_loc="./data"):
        log.info(
            f"Creating a wrapper class for ISI Cifar model checkpoint {checkpoint}"
        )
        self.__load_data__(checkpoint, data_loc)
        self.__training_data__(checkpoint)

    def __load_data__(self, checkpoint, data_loc):
        log.info(f"Loading ISI model and Cifar dataset")
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if checkpoint == "simple":
            num_classes = 10
            loading_model_path = os.path.join(
                DIR_NAME + "/bound_evaluation_models/isi_cifar_model/sample_model.pth"
            )
            log.info(f"Loading CIFAR10 dataset from {data_loc}")
            self.dataset = torchvision.datasets.CIFAR10(
                root=data_loc, train=True, download=True, transform=transform_test
            )
        elif checkpoint == "full":
            num_classes = 100
            loading_model_path = os.path.join(
                DIR_NAME
                + "/bound_evaluation_models/isi_cifar_model/sample_model_cifar100.pth"
            )
            log.info(f"Loading CIFAR100 dataset from {data_loc}")
            self.dataset = torchvision.datasets.CIFAR100(
                root=data_loc, train=True, download=True, transform=transform_test
            )

        elif (
            checkpoint == "10000"
            or checkpoint == "2500"
            or checkpoint == "4000"
            or checkpoint == "500"
        ):
            num_classes = 100
            loading_model_path = os.path.join(
                DIR_NAME
                + "/bound_evaluation_models/isi_cifar_model/Cifar100_Trained_Mode/"
                + checkpoint
                + "/cifar100_"
                + checkpoint
                + "_labels_model.pth"
            )
            log.info(f"Loading CIFAR100 dataset from {data_loc}")
            self.dataset = torchvision.datasets.CIFAR100(
                root=data_loc, train=True, download=True, transform=transform_test
            )

        else:
            log.error(f"Checkpoint {checkpoint} is invalid for the ISI CIfar model")
            raise Exception("This is not a valid network")

        self.mdl = MT_CNN(num_classes=num_classes, input_channel=3).to(
            device=IsiCifar.device
        )

        checkpoint_load = torch.load(loading_model_path, map_location=IsiCifar.device)
        self.mdl.load_state_dict(checkpoint_load["teacher_network"])
        self.labels = np.array(self.dataset.targets)

        return

    def __training_data__(self, checkpoint):
        log.info(f"Loading training data for ISI Cifar {checkpoint}")
        if checkpoint == "simple":
            file = DIR_NAME + "/bound_evaluation_models/isi_cifar_model/01.txt"
        elif checkpoint == "full":
            file = DIR_NAME + "/bound_evaluation_models/isi_cifar_model/01_cifar100.txt"
        elif (
            checkpoint == "10000"
            or checkpoint == "2500"
            or checkpoint == "4000"
            or checkpoint == "500"
        ):
            file = (
                DIR_NAME
                + "/bound_evaluation_models/isi_cifar_model/Cifar100_Trained_Mode/"
                + checkpoint
                + "/cifar100_"
                + checkpoint
                + "_labels.txt"
            )
        else:
            log.error(f"Checkpoint {checkpoint} is invalid for the ISI Cifar model")
            raise Exception("This is not a valid network")
        f = open(file)
        train = np.array([])
        for x in f:
            y = int(x.split("_")[0])
            train = np.concatenate((train, np.array([y])))
        train = train.astype(int)

        self.train_set = Subset(self.dataset, train)
        self.train_labels = self.labels[train]
        return

    def get_model_weights(self):
        log.info("Getting the current model weights for ISI Cifar model")
        params = np.array([])
        layer_list = [
            "conv1a",
            "conv1b",
            "conv1c",
            "conv2a",
            "conv2b",
            "conv2c",
            "conv3a",
            "conv3b",
            "conv3c",
            "fc1",
        ]
        state_dict = self.mdl.state_dict()
        for layer in layer_list:
            if (layer + ".weight_v") in state_dict.keys():
                weight = Model.weights_from_v_g(
                    state_dict[layer + ".weight_v"].cpu().numpy(),
                    state_dict[layer + ".weight_g"].cpu().numpy(),
                ).flatten()
            else:
                weight = state_dict[layer + ".weight"].cpu().numpy().flatten()
            bias = state_dict[layer + ".bias"].cpu().numpy().flatten()
            params = np.concatenate((params, weight, bias))
        return params

    def set_model_weights(self, params):
        log.info("Updating model weights for the ISI Cifar model")
        layer_list = [
            "conv1a",
            "conv1b",
            "conv1c",
            "conv2a",
            "conv2b",
            "conv2c",
            "conv3a",
            "conv3b",
            "conv3c",
            "fc1",
        ]
        idx = 0
        state_dict = self.mdl.state_dict()
        for layer in layer_list:
            if (layer + ".weight_v") in state_dict.keys():
                wght_shape = state_dict[layer + ".weight_v"].cpu().numpy().shape
                n = state_dict[layer + ".weight_v"].cpu().numpy().size
                new_weights = np.reshape(params[idx : idx + n], wght_shape)
                rshp = [wght_shape[0], -1] + [1] * (len(wght_shape) - 2)
                g = np.sqrt(
                    np.sum(
                        np.power(new_weights.reshape(rshp), 2), axis=1, keepdims=True
                    )
                )
                state_dict[layer + ".weight_v"] = torch.tensor(
                    new_weights, dtype=torch.float32, device=IsiCifar.device
                )
                state_dict[layer + ".weight_g"] = torch.tensor(
                    g, dtype=torch.float32, device=IsiCifar.device
                )
            else:
                wght_shape = state_dict[layer + ".weight"].cpu().numpy().shape
                n = state_dict[layer + ".weight"].cpu().numpy().size
                new_weights = np.reshape(params[idx : idx + n], wght_shape)
                state_dict[layer + ".weight"] = torch.tensor(
                    new_weights, dtype=torch.float32, device=IsiCifar.device
                )
            idx = idx + n
            bias_shape = state_dict[layer + ".bias"].cpu().numpy().shape
            n = state_dict[layer + ".bias"].cpu().numpy().size
            new_bias = np.reshape(params[idx : idx + n], bias_shape)
            state_dict[layer + ".bias"] = torch.tensor(
                new_bias, dtype=torch.float32, device=IsiCifar.device
            )
            idx = idx + n
        self.mdl.load_state_dict(state_dict)
        return

    def model_eval(self, params, dataset):
        log.info("Evaluating the ISI Cifar model")
        self.set_model_weights(params)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )
        self.mdl.eval()
        outputs = None

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = (
                    inputs.to(IsiCifar.device),
                    targets.to(IsiCifar.device),
                )
                _, _, _, out = self.mdl(inputs)
                if outputs is None:
                    outputs = out.cpu().numpy()
                else:
                    outputs = np.concatenate((outputs, out.cpu().numpy()))
        return outputs

    def grad_model_out_weights(self, dataset, params):
        log.info("Calculating the derivative of the ISI Cifar model")
        self.set_model_weights(params)
        layer_list = [
            self.mdl.conv1a,
            self.mdl.conv1b,
            self.mdl.conv1c,
            self.mdl.conv2a,
            self.mdl.conv2b,
            self.mdl.conv2c,
            self.mdl.conv3a,
            self.mdl.conv3b,
            self.mdl.conv3c,
            self.mdl.fc1,
        ]
        gradient = np.zeros((len(dataset.indices), params.size), dtype=np.float32)
        output_vals = None
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            x_in, y = inputs.to(IsiCifar.device), targets
            cat = np.zeros((1, max(self.labels) + 1))
            cat[0, y - 1] = 1
            cat = torch.tensor(cat, device=IsiCifar.device)
            self.mdl.zero_grad()
            self.mdl.eval()
            _, _, _, output = self.mdl(x_in)
            if output_vals is None:
                output_vals = output.cpu().detach().numpy()
                output_vals = np.reshape(output_vals, (1, output_vals.size))
            else:
                output_vals = np.concatenate(
                    (output_vals, output.cpu().detach().numpy()), axis=0
                )
            for layer in layer_list:
                layer.weight.retain_grad()
                layer.bias.retain_grad()
            output.backward(cat)
            grad = np.array([])
            for i in range(len(layer_list)):
                layer = layer_list[i]
                weight = layer.weight.grad.data.cpu().numpy().flatten()
                if False:
                    df_dv = layer.weight_v.grad.data.cpu().numpy()
                    df_dg = layer.weight_g.grad.data.cpu().numpy()
                    v = layer.weight_v.data.cpu().numpy()
                    g = layer.weight_g.data.cpu().numpy()
                    nrm = np.sqrt(
                        np.sum(
                            np.power(
                                v.reshape([v.shape[0], -1] + [1] * (len(v.shape) - 2)),
                                2,
                            ),
                            axis=1,
                            keepdims=True,
                        )
                    )
                    # df_dwi = df_dgi*dgi_dwi + df_d|v|*d|v|_dwi + df_dv*dv_dw
                    df_dw = df_dg * v / nrm + nrm * df_dv / g
                bias = layer.bias.grad.data.cpu().numpy().flatten()
                grad = np.concatenate((grad, weight, bias))
            gradient[batch_idx, :] = grad
        return gradient, output_vals
