import torch


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(abs(pred)+1), torch.log(actual+1)))

class Huber_unbalance(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def forward(self, pred, actual, weights):
        l1_loss = torch.abs(pred - actual)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / self.beta, l1_loss - 0.5 * self.beta)
        if not loss.shape:
            loss = weights*loss
        else:
            loss *= weights.expand_as(loss)
            loss = torch.mean(loss)
        return loss



def loss_function(loss_type, balance, **kwargs):
    if not balance:
        if loss_type == 'Mean_Square_Error':
            loss = torch.nn.MSELoss()
        elif loss_type == "CrossEntropy":
            loss = torch.nn.CrossEntropyLoss()
        elif loss_type == "Mean_Absolute":
            loss = torch.nn.L1Loss()
        elif loss_type == "Mean_Square_Logarithmic_Error":
            loss = RMSLELoss()
        elif loss_type == "Huber_Loss":
            loss = torch.nn.HuberLoss(delta=20)
    else:
        if loss_type == "Huber_Loss":
            loss = Huber_unbalance(beta=20)
        if loss_type == "Huber_Loss":
            loss = Huber_unbalance(beta=20)

    return loss