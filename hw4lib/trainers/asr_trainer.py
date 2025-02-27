from .base_trainer import BaseTrainer

class ASRTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

    def _train_epoch(self, dataloader):
        pass

    def _validate_epoch(self, dataloader):
        pass
    
    def train(self, train_dataloader, val_dataloader):
        pass

    def evaluate(self, dataloader):
        pass    