from slim_grad.parts.neuron import Neuron

class Linear:
    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs=[n(x) for n in self.neurons]
        return outs

    def __repr__(self) -> str:
        return str(len(self.neurons))