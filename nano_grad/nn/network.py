class Network:
    def __init__(self, input) -> None:
        self.network = list(input.values())

    def size(self):
        return sum(weights.size() for weights in self.network)

    def parameters(self):
        return [weights.parameters() for weights in self.network]

    def shape(self):
        return [weights.shape() for weights in self.network]