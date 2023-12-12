import torch

__all__ = ['Stats', 'PrototypeActivationMeter']

class Stats:
    def __init__(self):
        self.count = None
        self.sum = None
        self.sumsq = None
        self.mean = None
        self.std = None

    def update(self, activation):
        if self.count is None:
            # This is the first activation tensor we've seen for this key,
            # so we initialize sums, sumsq, means, and std based on its shape.
            self.count = 0
            self.sum = torch.zeros_like(activation)
            self.sumsq = torch.zeros_like(activation)
            self.mean = torch.zeros_like(activation)
            self.std = torch.zeros_like(activation)

        # Update running stats here...
        self.count += 1
        self.sum += activation
        self.sumsq += activation**2
        self.mean = self.sum / self.count
        self.std = torch.sqrt(self.sumsq / self.count - self.mean**2)
        self.std[torch.isnan(self.std)] = 0

    def state_dict(self):
        return dict(
            count=self.count,
            sum=self.sum,
            sumsq=self.sumsq,
            mean=self.mean,
            std=self.std,
        )

class PrototypeActivationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.prototypes = defaultdict(Stats)

    def update(self, indices, activations):
        for index,activation in zip(indices, activations):
            self.prototypes[index].update(activation)

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.prototypes.items()}

    def __str__(self):
        fmtstr = 'PrototypeActivationMeter:\n'
        for i in range(self.num_classes):
            fmtstr += f'Class {i}: '
            for j in range(self.num_units):
                fmtstr += f'Unit {j}: Mean={self.means[i,j]:.4f}, Std={self.std[i,j]:.4f}; '
            fmtstr += '\n'
        return fmtstr
