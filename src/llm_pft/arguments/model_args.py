# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    """ This class encapsulates all parameters for a language model. """
    CONFIG_KEY = "model_args"

    model_ckpt: str = field(default=None, metadata={
        "help": "path to the checkpoint of the model."
    })

    architecture: str = field(default="gpt2", metadata={
        "help": "the architecture of the model",
        "choices": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                    "llama2-7b"]
    })

    pre_trained: bool = field(default=True, metadata={
        "help": "True: load pre-trained, public weights of the model. If additionally, a checkpoint is provided,"
                "      we always load the checkpoint last."
                "False: Randomly initialize model."
    })

    tokenizer_use_fast: bool = field(default=True, metadata={
        "help": "whether to set the flag use_fast in the tokenizer loading function."
    })

    tokenizer_max_length: int = field(default=1024, metadata={
        "help": "Limit the max token length."
    })

    peft: str = field(default="none", metadata={
        "help": "peft strategy",
        "choices": ["none", "lora"]
    })

    lora_r: int = field(default=4, metadata={
        "help": "lora dim",
    })

    lora_alpha: int = field(default=32, metadata={
        "help": "lora scaling",
    })

    lora_dropout: int = field(default=0., metadata={
        "help": "dropout rate",
    })
    
    device_map: str = field(default=None)

    def hash(self, suffix=""):
        """ Compute a unique hash based on this dict"""
        rep_dict = {
            "checkpoint": self.model_ckpt,
            "pre_trained": self.pre_trained,
            "tokenizer_max_length": self.tokenizer_max_length,
            "suffix": suffix,
        }
        if self.peft != 'none':
            rep_dict['peft'] = self.peft
            if self.peft == 'lora':
                rep_dict['lora_r'] = self.lora_r
                rep_dict['lora_alpha'] = self.lora_alpha
                rep_dict['lora_dropout'] = self.lora_dropout
        return hashlib.sha256(repr(rep_dict).encode('utf-8')).hexdigest()

