import lightning as pl
import torch

class LitModule(pl.LightningModule):
    """
    This is a standard PyTorch Lightning module,
    with a few PyTorch-Lightning-specific things added.

    The main things to notice are:
    - Instead of defining different steps for training, validation, and testing,
        we define a single `step` function, and then define `training_step`,
        `validation_step`, and `test_step` as thin wrappers that call `step`.
    """

    def __init__(self, model, number_of_features, sequence_length, past_sequence_length, future_sequence_length,
                 batch_size):
        super().__init__()
        self.model = model
        self.nx = number_of_features
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        string = "training"
        loss = self.step(batch, batch_idx, string)
        return loss

    def validation_step(self, batch, batch_idx):
        string = "validation"
        loss = self.step(batch, batch_idx, string)
        return loss

    def test_step(self, batch, batch_idx):
        string = "test"
        loss = self.step(batch, batch_idx, string)
        return loss

    def step(self, batch, batch_idx, string):
        """
        This is the main step function that is used by training_step, validation_step, and test_step.
        """
        # TODO: You have to modify this based on your task, model and data. This is where most of the engineering
        #  happens!
        x, y = self.prep_data_for_step(batch)
        x = x.float()  # Ensure x is in FP32
        y = y.float()  # Ensure y is in FP32

        y_hat_list = []

        for k in range(self.future_sequence_length):
            y_hat_k = self(x)
            y_hat_list.append(y_hat_k)
            if y_hat_k.dim() < 3:
                y_hat_k = y_hat_k.unsqueeze(1)
            x = torch.cat([x[:, 1:, :].float(), y_hat_k.float()], dim=1)

        y_hat = torch.stack(y_hat_list, dim=1).squeeze()

        loss = self.model.loss_function(y_hat, y)
        self.log(f"{string}_loss", loss)

        return loss

    def prep_data_for_step(self, batch):
        # TODO: This is a hacky way to load one rectangular block from the data, and divide it into x and y of different
        #  sizes afterwards.
        x = batch[:, :self.past_sequence_length, :]
        y = batch[:, self.past_sequence_length:, :]

        return x, y

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        # if parameters: #Previus code
        # if parameters: #Febin
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-3,
                                     eps=1e-5,
                                     # fused=True,
                                     fused=False,
                                     amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            patience=3,
            threshold=1e-4,
            cooldown=2,
            eps=1e-6,
            verbose=True,
        )
        optimizer_and_scheduler = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "training_loss",
                "frequency": 1,
                "strict": True}
        }
        return optimizer_and_scheduler

