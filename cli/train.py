
class TrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        
    def train_step(self, batch, prefix='train'):
        loss = F.cross_entropy(logits, spkid)
        lossdict = {f'{prefix}_loss': loss.item()}
        return loss, lossdict
    
    def valid_step(self, batch, prefix='valid'):
        return self.train_step(batch, prefix)

