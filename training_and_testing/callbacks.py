from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


# Here are some examples for callback. They save your model, when a new best is achieved.
def create_callbacks(model_type):
    if model_type == "MLP" or model_type == "LSTM" or model_type == "GRU":
        cb = \
            [
                ModelCheckpoint(monitor="training_loss",
                                filename=f"{model_type}-checkpoint",
                                mode="min",
                                every_n_epochs=1,
                                save_top_k=2,
                                verbose="True",
                                auto_insert_metric_name="True",
                                save_on_train_epoch_end=True, ),

                ModelCheckpoint(monitor="validation_loss",
                                filename="VAL_CKPT_{validation_loss:.6f}-{training_loss:.6f}-{epoch}",
                                mode="min",
                                every_n_epochs=1,
                                save_top_k=2,
                                verbose="True",
                                auto_insert_metric_name="True", ),

                ModelCheckpoint(filename=f"{model_type}-checkpoint1",
                                every_n_epochs=10,
                                verbose="True",
                                auto_insert_metric_name="True", ),

                LearningRateMonitor(logging_interval='step')
            ]
    else:
        cb = \
            [ModelCheckpoint(monitor="training_loss",
                             filename="TRAIN_CKPT_{training_loss:.6f}-{validation_loss:.6f}-{epoch}",
                             mode="min",
                             every_n_epochs=1,
                             save_top_k=2,
                             verbose="True",
                             auto_insert_metric_name="True",
                             save_on_train_epoch_end=True, ),

             ModelCheckpoint(monitor="validation_loss",
                             filename="VAL_CKPT_{validation_loss:.6f}-{training_loss:.6f}-{epoch}",
                             mode="min",
                             every_n_epochs=1,
                             save_top_k=2,
                             verbose="True",
                             auto_insert_metric_name="True", ),

             ModelCheckpoint(filename="REC_CKPT_{epoch}-{validation_loss:.6f}-{training_loss:.6f}",
                             every_n_epochs=10,
                             verbose="True",
                             auto_insert_metric_name="True", ),

             LearningRateMonitor(logging_interval='step')
             ]

    return cb
