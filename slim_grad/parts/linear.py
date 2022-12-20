from slim_grad.parts.neuron import Neuron

class Linear:
    def __init__(self, nin, nout) -> None:
        #TODO: Should we remove the neuron class entirely? This may make it easier to put into matices form
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs=[n(x) for n in self.neurons]
        return outs

    def __repr__(self) -> str:
        return str(len(self.neurons))