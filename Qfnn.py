import pennylane as qml
import torch.nn as nn
from Cla_net import Cla_net
import torch

n_qubits = 4
n_layers = 2

dev=qml.device('default.qubit', wires=n_qubits)

weight_shapes = {"weights": (n_layers, n_qubits)}

@qml.qnode(dev,interface='torch',diff_method='backprop')
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights[0].reshape(1,n_qubits), wires=range(n_qubits),rotation=qml.RX)
    qml.BasicEntanglerLayers(weights[1].reshape(1,n_qubits), wires=range(n_qubits),rotation=qml.RZ)

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class Qfnn(nn.Module):
    def __init__(self,cla_path='') -> None:
        super(Qfnn,self).__init__()
        self.cla_net=Cla_net()
        # self.cla_net.load_state_dict(torch.load(cla_path))
        self.q_layer=qml.qnn.TorchLayer(qnode,weight_shapes)

    def forward(self,x):
        x=self.cla_net(x)
        max=torch.max(x)
        min=torch.min(x)
        x=(x-min)/(max-min)
        x=2*x*torch.pi
        x=self.q_layer(x)
        return x
    

# # main
# if __name__ == "__main__":
# #    #draw circuit
# #    seed=777
# #    torch.manual_seed(seed)
# #    input=torch.randn(10,4)
# # #    weight=torch.tensor([[0.1,0.2,0.3,0.4]])
# # #    re=qnode(input,weight)
# #    net=Qfnn()
# #    re=net(input)
# #    print(re)
#     g=torch.rand(2,4)
#     gg=torch.zeros(1,4)
#     gg[0]=g[0]
#     print(gg.shape)


    
