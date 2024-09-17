# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
# to disable flash attn?
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from pprint import pprint

import transformers

from llm_pft.arguments.config_args import ConfigArgs
from llm_pft.arguments.dataset_args import DatasetArgs
from llm_pft.arguments.env_args import EnvArgs
from llm_pft.arguments.model_args import ModelArgs
from llm_pft.arguments.ner_args import NERArgs
from llm_pft.arguments.outdir_args import OutdirArgs
from llm_pft.arguments.privacy_args import PrivacyArgs
from llm_pft.arguments.sampling_args import SamplingArgs
from llm_pft.arguments.trainer_args import TrainerArgs
from llm_pft.dataset.real_dataset import RealDataset
from llm_pft.models.language_model import LanguageModel
from llm_pft.models.model_factory import ModelFactory
from llm_pft.dataset.dataset_factory import DatasetFactory
from llm_pft.utils.output import print_highlighted, print_dict_highlighted

def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            PrivacyArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def fine_tune(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              privacy_args: PrivacyArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """ Fine-tunes a language model (LM) on some text dataset with/without privacy.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        privacy_args = config_args.get_privacy_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(config_args.get_privacy_args()))

    # -- Load the datasets
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
                                                                 ner_args=ner_args, env_args=env_args)
    if dataset_args.limit_dataset_size < len(train_dataset):
        print(f"limit_dataset_size: {dataset_args.limit_dataset_size} ({dataset_args.limit_dataset_size/len(train_dataset)})")
        train_dataset = train_dataset.select(list(range(dataset_args.limit_dataset_size)))

    # -- Load the LM
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    # -- Print configuration
    output_folder = outdir_args.create_folder_name()

    print_highlighted(f"Saving LM to: {output_folder}. Train Size: {len(train_dataset)},"
                      f" Eval Size: {len(eval_dataset)}")
    print_highlighted(f"Train Sample: {train_dataset.shuffle().first()}")

    # -- Fine-tune the LM
    lm.fine_tune(train_dataset, eval_dataset, train_args, privacy_args)

    # -- Print using the LM
    pprint(lm.generate(SamplingArgs(N=1)))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fine_tune(*parse_args())
# ----------------------------------------------------------------------------
