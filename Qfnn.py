import pennylane as qml
import torch.nn as nn
from Cla_net import Cla_net
import torch

n_qubits = 4
n_layers = 2

dev=qml.device('default.qubit', wires=n_qubits)

weight_shapes = {"weights": (n_layers, n_qubits)}

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class Qfnn(nn.Module):
    def __init__(self) -> None:
        super(Qfnn,self).__init__()
        self.cla_net=Cla_net()
        self.q_layer=qml.qnn.TorchLayer(qnode,weight_shapes)

    def forward(self,x):
        x=self.cla_net(x)
        x=self.q_layer(x)
        return x
    

# main
if __name__ == "__main__":
    data=torch.randn(10,3,299,299)
    qfnn=Qfnn()
    re=qfnn(data)
    print(re.shape)

    
