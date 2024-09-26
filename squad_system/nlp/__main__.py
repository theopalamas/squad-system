from typing import Literal

import click

from squad_system.nlp import entrypoints


@click.group()
def cli():
    pass


@cli.command(name="preprocess_data")
@click.option("--data_path_train", type=click.STRING, required=True)
@click.option("--data_path_val", type=click.STRING, required=True)
@click.option("--preprocessed_data_path", type=click.STRING, required=True)
@click.option("--pretrained_lm_name", type=click.STRING, required=True)
@click.option(
    "--task",
    type=click.Choice(["classification_answerable", "classification_indices"]),
    required=True,
)
@click.option("--allow_unanswerable", type=click.BOOL, default=True)
def preprocess_data(
    data_path_train: str,
    data_path_val: str,
    preprocessed_data_path: str,
    pretrained_lm_name: str,
    task: Literal["classification_answerable", "classification_indices"],
    allow_unanswerable: bool,
) -> None:
    entrypoints.preprocess_data(
        data_path_train=data_path_train,
        data_path_val=data_path_val,
        preprocessed_data_path=preprocessed_data_path,
        pretrained_lm_name=pretrained_lm_name,
        task=task,
        allow_unanswerable=allow_unanswerable,
    )


@cli.command(name="train")
@click.option("--cfg_path", type=click.STRING, required=True)
@click.option("--preprocessed_data_path", type=click.STRING, required=True)
@click.option("--checkpoint_path", type=click.STRING, required=True)
def train(cfg_path: str, preprocessed_data_path: str, checkpoint_path: str) -> None:
    entrypoints.train(
        cfg_path=cfg_path,
        preprocessed_data_path=preprocessed_data_path,
        checkpoint_path=checkpoint_path,
    )


@cli.command(name="infer")
@click.option("--data_path", type=click.STRING, required=True)
@click.option("--checkpoint_file_answerable", type=click.STRING, default="")
@click.option("--cfg_file_answerable", type=click.STRING, default="")
@click.option("--checkpoint_file_indices", type=click.STRING, required=True)
@click.option("--cfg_file_indices", type=click.STRING, required=True)
@click.option("--results_file_path", type=click.STRING, required=True)
def infer(
    data_path: str,
    checkpoint_file_answerable: str,
    cfg_file_answerable: str,
    checkpoint_file_indices: str,
    cfg_file_indices: str,
    results_file_path: str,
) -> None:
    entrypoints.infer(
        data_path=data_path,
        checkpoint_file_answerable=checkpoint_file_answerable,
        cfg_file_answerable=cfg_file_answerable,
        checkpoint_file_indices=checkpoint_file_indices,
        cfg_file_indices=cfg_file_indices,
        results_file_path=results_file_path,
    )


if __name__ == "__main__":
    cli()
