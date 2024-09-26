import torch
from tokenizers import Tokenizer

from squad_system.nlp.model.io import load_model
from squad_system.nlp.model.model import SquadModel


class InferencePipeline:
    def __init__(
        self,
        tokenizer_answerable: Tokenizer,
        model_answerable: SquadModel,
        tokenizer_indices: Tokenizer,
        model_indices: SquadModel,
        device: str,
    ) -> None:
        self.tokenizer_answerable = tokenizer_answerable
        self.model_answerable = model_answerable
        self.tokenizer_indices = tokenizer_indices
        self.model_indices = model_indices
        self.device = device

    @torch.no_grad()
    def infer(self, question: str, context: str) -> str:
        """
        This is used for inference based on a question and a context.
        We can either infer directly from the 'model_indices' model (if 'model_answerable' is None)
        or check if the question is answerable first and then try to answer it
        :param question: Question to be answered
        :param context: Context which might contain the answer
        """
        if self.model_answerable:
            encoded_input_answerable = self.tokenizer_answerable.encode_plus(
                question, context, truncation=True, return_tensors="pt"
            )
            outputs_answerable = self.model_answerable(
                encoded_input_answerable["input_ids"].to(self.device),
                encoded_input_answerable["attention_mask"].to(self.device),
            )
            answerable = torch.argmax(outputs_answerable)
            if answerable:
                encoded_input_indices = self.tokenizer_indices.encode_plus(
                    question, context, truncation=True, return_tensors="pt"
                )
                outputs_indices = self.model_indices(
                    encoded_input_indices["input_ids"].to(self.device),
                    encoded_input_indices["attention_mask"].to(self.device),
                )
                outputs_indices = outputs_indices.squeeze()

                answer_start = torch.argmax(outputs_indices[:, 0])
                answer_end = torch.argmax(outputs_indices[:, 1])
                answer_ids = encoded_input_indices["input_ids"][0][
                    answer_start:answer_end
                ]
                answer_tokens = self.tokenizer_indices.convert_ids_to_tokens(answer_ids)
                answer = self.tokenizer_indices.convert_tokens_to_string(answer_tokens)

                return answer
            else:
                return ""
        else:
            encoded_input_indices = self.tokenizer_indices.encode_plus(
                question, context, truncation=True, return_tensors="pt"
            )
            outputs_indices = self.model_indices(
                encoded_input_indices["input_ids"].to(self.device),
                encoded_input_indices["attention_mask"].to(self.device),
            )
            outputs_indices = outputs_indices.squeeze()

            answer_start = torch.argmax(outputs_indices[:, 0])
            answer_end = torch.argmax(outputs_indices[:, 1])
            answer_ids = encoded_input_indices["input_ids"][0][
                answer_start : answer_end + 1
            ]
            answer_tokens = self.tokenizer_indices.convert_ids_to_tokens(answer_ids)
            answer = self.tokenizer_indices.convert_tokens_to_string(answer_tokens)

            return answer


def get_inference_pipeline(
    checkpoint_file_answerable: str = "",
    cfg_file_answerable: str = "",
    checkpoint_file_indices: str = "out/classification_indices_only_ans/checkpoints_model/model_ckpt_2.pt",
    cfg_file_indices: str = "out/classification_indices_only_ans/train_config.json",
    device: str = "cpu",
) -> InferencePipeline:
    """
    Creates the inference pipeline. checkpoint_file_answerable and cfg_file_answerable
    can be left blank if not needed
    :param checkpoint_file_answerable: Model path for 'classification_answerable' model
    :param cfg_file_answerable: Config file for 'classification_answerable' model
    :param checkpoint_file_indices: Model path for 'classification_indices' model
    :param cfg_file_indices: Config file for 'classification_indices' model
    :param device: Device to load to
    """
    if checkpoint_file_answerable:
        _, tokenizer_answerable, model_answerable = load_model(
            checkpoint_file_answerable, cfg_file_answerable, device
        )
    else:
        tokenizer_answerable, model_answerable = None, None
    _, tokenizer_indices, model_indices = load_model(
        checkpoint_file_indices, cfg_file_indices, device
    )

    return InferencePipeline(
        tokenizer_answerable, model_answerable, tokenizer_indices, model_indices, device
    )
