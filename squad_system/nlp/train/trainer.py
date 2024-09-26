import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from squad_system.nlp.dataset.data_model import (
    ClassificationAnswerableItem,
    ClassificationIndicesItem,
)
from squad_system.nlp.dataset.dataset import get_dataloader
from squad_system.nlp.model.model import SquadModel
from squad_system.nlp.train.callbacks import CheckpointCallback
from squad_system.nlp.train.config import TrainConfig
from squad_system.nlp.train.state_model import TrainerState
from squad_system.nlp.utils import get_device


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        device: str,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        model: SquadModel,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        trainer_state: TrainerState,
        checkpoint_callback: CheckpointCallback,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.trainer_state = trainer_state
        self.checkpoint_callback = checkpoint_callback

    def train(self) -> None:
        for epoch in range(self.cfg.epochs):
            print(f"Epoch [{epoch+1}/{self.cfg.epochs}]")
            self.epoch_train(epoch)
            self.epoch_val(epoch)

            self.trainer_state.epochs_train += 1
            self.checkpoint_callback.save(
                self.model, self.optimizer, self.trainer_state
            )

    def epoch_train(self, epoch: int) -> None:
        self.model.train()

        loss_epoch = 0
        steps_train = 0
        y_true = []
        y_pred = []
        with tqdm(total=len(self.dataloader_train)) as pbar:
            for step, batch in enumerate(self.dataloader_train):
                self.optimizer.zero_grad()

                token_ids = batch["token_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(token_ids, attention_mask)

                if self.cfg.task == "classification_answerable":
                    answerable = batch["answerable"].to(self.device)

                    preds = torch.argmax(logits, dim=1)
                    y_true.append(answerable.detach().cpu())
                    y_pred.append(preds.detach().cpu())

                    loss = self.loss_function(logits, answerable)
                if self.cfg.task == "classification_indices":
                    answer_start = batch["answer_start"].to(self.device)
                    answer_end = batch["answer_end"].to(self.device)

                    answer_start_logits, answer_end_logits = (
                        logits[:, :, 0],
                        logits[:, :, 1],
                    )

                    ignored_index = answer_start_logits.size(1)
                    answer_start = answer_start.clamp(0, ignored_index)
                    answer_end = answer_end.clamp(0, ignored_index)

                    answer_start_loss = self.loss_function(
                        answer_start_logits, answer_start
                    )
                    answer_end_loss = self.loss_function(answer_end_logits, answer_end)
                    loss = (answer_start_loss + answer_end_loss) / 2

                loss.backward()
                self.optimizer.step()

                loss_epoch += loss.item()
                steps_train += 1

                pbar.set_description("Training...")
                pbar.set_postfix(loss=f"{loss_epoch/steps_train}")
                pbar.update()

        if self.cfg.task == "classification_answerable":
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            self.trainer_state.accuracies_train.append(accuracy_score(y_true, y_pred))
            self.trainer_state.precisions_train.append(precision_score(y_true, y_pred))
            self.trainer_state.recalls_train.append(recall_score(y_true, y_pred))
        self.trainer_state.current_loss_train = loss_epoch / steps_train
        self.trainer_state.losses_train.append(self.trainer_state.current_loss_train)
        if self.trainer_state.current_loss_train < self.trainer_state.best_loss_train:
            self.trainer_state.best_loss_train = self.trainer_state.current_loss_train
            self.trainer_state.best_epoch_train = epoch + 1

    @torch.no_grad()
    def epoch_val(self, epoch: int) -> None:
        self.model.eval()

        loss_epoch = 0
        steps_val = 0
        y_true = []
        y_pred = []
        with tqdm(total=len(self.dataloader_val)) as pbar:
            for step, batch in enumerate(self.dataloader_val):
                token_ids = batch["token_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(token_ids, attention_mask)

                if self.cfg.task == "classification_answerable":
                    answerable = batch["answerable"].to(self.device)

                    preds = torch.argmax(logits, dim=1)
                    y_true.append(answerable.detach().cpu())
                    y_pred.append(preds.detach().cpu())

                    loss = self.loss_function(logits, answerable)
                if self.cfg.task == "classification_indices":
                    answer_start = batch["answer_start"].to(self.device)
                    answer_end = batch["answer_end"].to(self.device)

                    answer_start_logits, answer_end_logits = (
                        logits[:, :, 0],
                        logits[:, :, 1],
                    )

                    ignored_index = answer_start_logits.size(1)
                    answer_start = answer_start.clamp(0, ignored_index)
                    answer_end = answer_end.clamp(0, ignored_index)

                    answer_start_loss = self.loss_function(
                        answer_start_logits, answer_start
                    )
                    answer_end_loss = self.loss_function(answer_end_logits, answer_end)
                    loss = (answer_start_loss + answer_end_loss) / 2

                loss_epoch += loss.item()
                steps_val += 1

                pbar.set_description("Validating...")
                pbar.set_postfix(loss=f"{loss_epoch/steps_val}")
                pbar.update()

        if self.cfg.task == "classification_answerable":
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            self.trainer_state.accuracies_val.append(accuracy_score(y_true, y_pred))
            self.trainer_state.precisions_val.append(precision_score(y_true, y_pred))
            self.trainer_state.recalls_val.append(recall_score(y_true, y_pred))
        self.trainer_state.current_loss_val = loss_epoch / steps_val
        self.trainer_state.losses_val.append(self.trainer_state.current_loss_val)
        if self.trainer_state.current_loss_val < self.trainer_state.best_loss_val:
            self.trainer_state.best_loss_val = self.trainer_state.current_loss_val
            self.trainer_state.best_epoch_val = epoch + 1


def get_trainer(
    cfg: TrainConfig,
    preprocessed_data_train: list[ClassificationAnswerableItem]
    | list[ClassificationIndicesItem],
    preprocessed_data_val: list[ClassificationAnswerableItem]
    | list[ClassificationIndicesItem],
) -> Trainer:
    """
    Creates the trainer and all its needed components.
    :param cfg: Training config
    :param preprocessed_data_train: Preprocessed train data
    :param preprocessed_data_val: Preprocessed val data
    """
    device = get_device()

    dataloader_train = get_dataloader(
        preprocessed_data_train, cfg.task, cfg.batch_size, True, cfg.num_workers
    )
    dataloader_val = get_dataloader(
        preprocessed_data_val, cfg.task, cfg.batch_size, False, cfg.num_workers
    )

    model = SquadModel(cfg)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(cfg.b1, cfg.b2), eps=cfg.eps
    )

    loss_function = torch.nn.CrossEntropyLoss()

    trainer_state = TrainerState()

    checkpoint_callback = CheckpointCallback(cfg)

    trainer = Trainer(
        cfg,
        device,
        dataloader_train,
        dataloader_val,
        model,
        optimizer,
        loss_function,
        trainer_state,
        checkpoint_callback,
    )

    return trainer
