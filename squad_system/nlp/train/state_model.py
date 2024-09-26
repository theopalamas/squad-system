import sys

from pydantic import BaseModel, Field


class TrainerState(BaseModel):
    """
    Trainer state for logging stats
    """

    epochs_train: int = Field(default=0, description="Number of trained epochs.")
    best_epoch_train: int = Field(
        default=0, description="Epoch with the best train loss."
    )
    best_epoch_val: int = Field(
        default=0, description="Epoch with the best validation loss."
    )
    losses_train: list[float] = Field(default=[], description="List of train losses.")
    losses_val: list[float] = Field(
        default=[], description="List of validation losses."
    )
    best_loss_train: float = Field(
        default=sys.float_info.max, description="Best training loss."
    )
    best_loss_val: float = Field(
        default=sys.float_info.max, description="Best validation loss."
    )
    current_loss_train: float = Field(default=0, description="Current train loss.")
    current_loss_val: float = Field(default=0, description="Current validation loss.")

    accuracies_train: list[float] = Field(
        default=[],
        description="List of train accuracies (Only applicable for classification_answerable).",
    )
    accuracies_val: list[float] = Field(
        default=[],
        description="List of validation accuracies (Only applicable for classification_answerable).",
    )
    precisions_train: list[float] = Field(
        default=[],
        description="List of train precisions (Only applicable for classification_answerable).",
    )
    precisions_val: list[float] = Field(
        default=[],
        description="List of validation precisions (Only applicable for classification_answerable).",
    )
    recalls_train: list[float] = Field(
        default=[],
        description="List of train precisions (Only applicable for classification_answerable).",
    )
    recalls_val: list[float] = Field(
        default=[],
        description="List of validation precisions (Only applicable for classification_answerable).",
    )
