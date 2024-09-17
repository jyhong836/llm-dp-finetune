# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

# import dp_transformers
import numpy as np
import torch
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    TrainerCallback
from transformers.optimization import get_linear_schedule_with_warmup

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset
from ..utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback
from ..utils.output import print_highlighted
from ..utils.web import is_valid_url, download_and_unzip


@dataclass
class GeneratedText:
    text: str  # the generated text
    score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])
    
    def __len__(self):
        return len(self.data) if self.data is not None else 0


class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """ A wrapper class around a huggingface LM.
        """
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """ Gets the maximum size of the context """
        if hasattr(self._lm.config, 'n_positions'):
            return self._lm.config.n_positions
        else:
            return 1e12

    @abstractmethod
    def tokenizer(self):
        """ Returns this model's tokenizer. """
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def load(self, verbose: bool = False) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint.
        """
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
                                                    use_fast=self.model_args.tokenizer_use_fast)
        loaded_peft_model = False

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'.")

            if is_valid_url(self.model_args.model_ckpt):
                self.model_args.model_ckpt = download_and_unzip(self.model_args.model_ckpt)
            if self.model_args.peft == 'none':
                self._lm = model_cls.from_pretrained(self.model_args.model_ckpt, return_dict=True, device_map='auto')
            elif self.model_args.peft == 'lora':
                from peft.peft_model import PeftModel
                self._lm = model_cls.from_pretrained(self.model_args.architecture, return_dict=True, device_map='auto')
                print(f"Load peft model: lora..")
                self._lm = PeftModel.from_pretrained(
                    self._lm, self.model_args.model_ckpt, return_dict=True, device_map='auto')
                loaded_peft_model = True
            else:
                raise NotImplementedError(f"peft mode: {self.model_args.peft}")
            self._lm.eval()
        elif self.model_args.pre_trained:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")
            model_kwargs = {}
            if self.model_args.device_map is not None:
                model_kwargs['device_map'] = self.model_args.device_map
            self._lm = model_cls.from_pretrained(
                self.model_args.architecture, return_dict=True, **model_kwargs 
                ).eval()
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = model_cls(config=self.get_config())

        if self.model_args.peft != 'none' and not loaded_peft_model:
            # need to change for different models
            # lora_target_modules = [
            #     "q_proj",
            #     "v_proj",
            # ]
            lora_target_modules = ['q_proj','k_proj','v_proj','o_proj']
            if self.model_args.peft == 'lora':
                from peft import LoraConfig, PromptTuningConfig, PeftModel
                peft_config = LoraConfig(
                    lora_alpha=self.model_args.lora_alpha,
                    lora_dropout=self.model_args.lora_dropout,
                    r=self.model_args.lora_r,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules
                )
            if peft_config is not None:
                from peft import get_peft_model
                self._lm = get_peft_model(self._lm, peft_config)
                self._lm.print_trainable_parameters()

        self._tokenizer.padding_side = "right"
        # if self._tokenizer.pad_token is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._lm.config.pad_token_id = self._lm.config.eos_token_id
        if not (self.model_args.model_ckpt or self.model_args.pre_trained):
            self._lm.to(self.env_args.device)
        return self

    def substring_perplexity(self, seq: str, substring: str) -> float:
        """ Computes the perplexity of a substring in a string.
        For example: seq="My name is Ronald and I like hamburgers.", substring="Ronald",
        then this function computes the perplexity of generating "Ronald" given prefix "My name is".
        """
        original_mode = self._lm.training
        self._lm.eval()

        txt = seq[:seq.index(substring) + len(substring)]
        input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
        substring_len = len(self._tokenizer.encode(substring, truncation=True))
        target_ids = input_ids.clone()
        target_ids[:, :input_ids.size(1) - substring_len] = -100
        with torch.no_grad():
            outputs = self._lm(input_ids, labels=target_ids)
        loss, _, num_tokens = outputs[:3]

        perplexity = torch.exp(loss / num_tokens)

        self._lm.training = original_mode
        return perplexity.cpu().item()

    def autocomplete(self, sampling_args: SamplingArgs):
        """ Predicts the top-1 most probable next tokens. """
        return self.generate(sampling_args)[0]

    def print_sample(self, prompt=None):
        self._lm.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64))
        print_highlighted(data[0].text)
        return data[0].text

    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using the sampling args.
        """
        self._lm.eval()

        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt]
        )

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].repeat(r, 1)
        attention_mask = inputs['attention_mask'].repeat(r, 1)

        def generate_batch(input_ids, attention_mask) -> List[GeneratedText]:
            """ Helper function to generate a single batch of text.
            """
            input_len = input_ids.size(1)
            out = self._lm.generate(
                input_ids=input_ids.to(self.env_args.device),
                attention_mask=attention_mask.to(self.env_args.device),
                max_length=min(self.n_positions, input_len + sampling_args.seq_len),
                do_sample=sampling_args.do_sample,
                top_k=sampling_args.top_k,
                top_p=sampling_args.top_p,
                output_scores=True,
                return_dict_in_generate=True
            )

            generated_texts: List[GeneratedText] = []
            for text, score in zip(
                    self._tokenizer.batch_decode(out.sequences, skip_special_tokens=False),
                    [torch.softmax(x, 1) if sampling_args.as_probabilities else x for x in out.scores]
            ):
                generated_texts.append(GeneratedText(text=text, score=score.detach().cpu()))
            return generated_texts

        generated_data: List[GeneratedText] = []
        num_batches = int(np.ceil(sampling_args.N / self.env_args.eval_batch_size))
        for _ in tqdm(
                range(num_batches),
                disable=not sampling_args.generate_verbose,
                desc="Generating with LM"
        ):
            generated_data.extend(generate_batch(input_ids, attention_mask))

        return GeneratedTextList(data=generated_data)

    def tokenize_datasets(self, datasets: List[RealDataset], column_name="text", pre_remove_columns=False) -> List:
        """ Tokenizes the 'text' column of a list of dataset using this model's tokenizer """
        tokenize_function = lambda x: self._tokenizer(x[column_name], truncation=True, max_length=self.model_args.tokenizer_max_length)

        processed_datasets = []
        for dataset in datasets:
            hf_dataset = dataset.get_hf_dataset()
            if pre_remove_columns:
                hf_dataset = hf_dataset.remove_columns([c for c in hf_dataset.column_names if c not in [column_name]])
            
            hf_dataset = hf_dataset.map(tokenize_function, batched=True)

            if pre_remove_columns:
                # FIXME token_type_ids may be needed somewhere.
                hf_dataset = hf_dataset.remove_columns([c for c in hf_dataset.column_names if c in [column_name, 'token_type_ids']])
            processed_datasets.append(hf_dataset)
        return processed_datasets

    def perplexity(self, data: Union[list, str], offset=0, max_length=0, apply_exp=True, verbose=True,
                   return_as_list: bool = False) -> float:
        """ Compute the perplexity of the model on a string.
        """
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of tokens viewed
        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
            input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True, max_length=self.model_args.tokenizer_max_length)).unsqueeze(0).to(self.env_args.device)
            target_ids = input_ids.clone()

            if offset > 0:  # ignore everything up to the offset
                target_ids[:, :offset] = -100

            tgt_len = (target_ids.size(1) - offset)
            if max_length > 0:  # ignore everything except offset:offset+max_length
                target_ids[:, offset + max_length:] = -100
                tgt_len = max_length

            with torch.no_grad():
                outputs = self._lm(input_ids, labels=target_ids)
            loss, logits = outputs[:2]
            if return_as_list:
                nlls.append(loss.cpu().detach())
            else:
                nlls.append(loss.cpu().detach())
                ctr += tgt_len

        self._lm.training = original_mode
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())

    # def _fine_tune_dp(self,
    #                   train_dataset: RealDataset,
    #                   eval_dataset: RealDataset,
    #                   train_args: TrainerArgs,
    #                   privacy_args: PrivacyArgs):

    #     with train_args.main_process_first(desc="Tokenizing datasets"):
    #         eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
    #         assert not train_args.remove_unused_columns, "DP does not support remove_unused_columns which can not work with GradSampleModule"
    #         print("Tokenizing Train and Eval Datasets ..")
    #         hf_train_dataset, hf_eval_dataset = self.tokenize_datasets(
    #             [train_dataset, eval_dataset], 
    #             pre_remove_columns=not train_args.remove_unused_columns  # if trainer does not remove (e.g., for DP), we will remove columns here (hard-coded may not apply for all)
    #             )
    #         print('done')

    #     # self._lm = self._lm.to(self.env_args.device)
    #     self._lm.train()

    #     data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

    #     # transfer privacy args
    #     dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
    #                                                         target_epsilon=privacy_args.target_epsilon,
    #                                                         target_delta=privacy_args.target_delta,
    #                                                         per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)

    #     trainer = dp_transformers.dp_utils.OpacusDPTrainer(
    #         args=train_args,
    #         model=self._lm,
    #         train_dataset=hf_train_dataset,
    #         eval_dataset=hf_eval_dataset,
    #         data_collator=data_collator,
    #         privacy_args=dpt_privacy_args,
    #     )

    #     try:
    #         trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    #     finally:
    #         eps_prv = trainer.get_prv_epsilon()
    #         eps_rdp = trainer.get_rdp_epsilon()
    #         trainer.log({
    #             "final_epsilon_prv": eps_prv,
    #             "final_epsilon_rdp": eps_rdp
    #         })
    #         print(f"saving model..")
    #         trainer.save_model()
    #         # trainer.model is a GradSampleModule.
    #         trainer.model._module.save_pretrained(trainer.args.output_dir)
    #         trainer.model._module.config.save_pretrained(trainer.args.output_dir)
    #         trainer.model._module.generation_config.save_pretrained(trainer.args.output_dir)
    #     self._lm.eval()

    def fine_tune(self,
                  train_dataset,
                  eval_dataset,
                  train_args: TrainerArgs,
                  privacy_args: PrivacyArgs):
        """ Fine-Tune the LM with/without DP
        """
        if privacy_args.target_epsilon > 0:
            # return self._fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
            # return self._fine_tune_fast_dp(train_dataset, eval_dataset, train_args, privacy_args)
            return self._fine_tune_fast_dp_ZERO(train_dataset, eval_dataset, train_args, privacy_args)
        return self._fine_tune(train_dataset, eval_dataset, train_args)

    def _fine_tune_fast_dp(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   privacy_args: PrivacyArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        !Note: This only allow single GPU. Don't use auto devicemap.
        """
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        #train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset], pre_remove_columns=not train_args.remove_unused_columns)
        print("Done Tokenizing!")
        print("model:", self._lm)
        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)
        
        
        
        params = tuple(param for param in self._lm.parameters() if param.requires_grad)
        names = tuple(name for name, param in self._lm.named_parameters() if param.requires_grad)
        num_trainable_params = sum(param.numel() for param in params)
        print(f"Number of trainable params: {num_trainable_params / 1e6:.4f} million")
        print(f'Number of total params: {sum(param.numel() for param in self._lm.parameters()) / 1e6:.3f} million')

        # print(json.dumps(names, indent=4))

        # TODO: Using a single gigantic parameter group is okay only when `weight_decay` is 0.
        #   Biases and LM parameters should not be decayed perhaps even with privacy.
        optimizer = torch.optim.AdamW(
            params=params,
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            eps=train_args.adam_epsilon,
        )
        trainer.optimizer = optimizer

        # Create the lr_scheduler.
        try:
            num_GPUs=torch.distributed.get_world_size()
        except:
            num_GPUs=1
            
        #if train_args.logical_batch_size!=None:
        #    trainer.args.gradient_accumulation_steps=train_args.logical_batch_size/train_args.per_device_train_batch_size/num_GPUs
        #else:
        logical_batch_size=trainer.args.gradient_accumulation_steps*train_args.per_device_train_batch_size*num_GPUs

        num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)
        #if train_args.lr_decay:
        #    trainer.lr_scheduler = get_linear_schedule_with_warmup(
        #        trainer.optimizer,
        #        num_warmup_steps=train_args.warmup_steps,
        #        num_training_steps=t_total,
        #    )
        #else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
        
        from fastDP import PrivacyEngine
        privacy_engine = PrivacyEngine(
            module=self._lm,
            batch_size=logical_batch_size,
            sample_size=len(train_dataset),
            epochs=train_args.num_train_epochs,
            max_grad_norm=privacy_args.max_grad_norm_dp,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            #accounting_mode=privacy_args.accounting_mode,
            #clipping_mode=privacy_args.clipping_mode,
            #clipping_fn=privacy_args.clipping_fn,
            #clipping_style=privacy_args.clipping_style,
            #origin_params=['wte','wpe'],
            origin_params=None,
            num_GPUs=1,
            torch_seed_is_fixed=True,
        )
        
        # Originally, these could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta
        
        # print('privacy_args: ')
        # print(json.dumps(privacy_args.__dict__, indent=4))
        privacy_engine.attach(optimizer)

        try:
            trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        finally:
            print(f"saving to {trainer.args.output_dir}")
            trainer.save_model()
            # trainer.model.save_pretrained(trainer.args.output_dir)
            trainer.model.config.save_pretrained(trainer.args.output_dir)
            trainer.model.generation_config.save_pretrained(trainer.args.output_dir)
        self._lm.eval()
    

    def _fine_tune_fast_dp_ZERO(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   privacy_args: PrivacyArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        Use deepspeed to train on multiple GPUs.
        """
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset], pre_remove_columns=not train_args.remove_unused_columns)
        print("Done Tokenizing!")

        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)
        
        
        
        params = tuple(param for param in self._lm.parameters() if param.requires_grad)
        names = tuple(name for name, param in self._lm.named_parameters() if param.requires_grad)
        num_trainable_params = sum(param.numel() for param in params)
        print(f"Number of trainable params: {num_trainable_params / 1e6:.4f} million")
        print(f'Number of total params: {sum(param.numel() for param in self._lm.parameters()) / 1e6:.3f} million')

        # print(json.dumps(names, indent=4))

        # TODO: Using a single gigantic parameter group is okay only when `weight_decay` is 0.
        #   Biases and LM parameters should not be decayed perhaps even with privacy.
        optimizer = torch.optim.AdamW(
            params=params,
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            eps=train_args.adam_epsilon,
        )
        trainer.optimizer = optimizer

        # Create the lr_scheduler.
        try:
            num_GPUs=torch.distributed.get_world_size()
            # len(eval(os.environ['CUDA_VISIBLE_DEVICES']))
        except:
            num_GPUs=1
            
        #if train_args.logical_batch_size!=None:
        #    trainer.args.gradient_accumulation_steps=train_args.logical_batch_size/train_args.per_device_train_batch_size/num_GPUs
        #else:
        logical_batch_size=trainer.args.gradient_accumulation_steps*train_args.per_device_train_batch_size*num_GPUs

        num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)
        #if train_args.lr_decay:
        #    trainer.lr_scheduler = get_linear_schedule_with_warmup(
        #        trainer.optimizer,
        #        num_warmup_steps=train_args.warmup_steps,
        #        num_training_steps=t_total,
        #    )
        #else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
        
        from fastDP import PrivacyEngine_Distributed_Stage_2_and_3
        privacy_engine = PrivacyEngine_Distributed_Stage_2_and_3(
            module=self._lm,
            batch_size=logical_batch_size,
            sample_size=len(train_dataset),
            epochs=train_args.num_train_epochs,
            max_grad_norm=privacy_args.max_grad_norm_dp,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            #accounting_mode=privacy_args.accounting_mode,
            #clipping_mode=privacy_args.clipping_mode,
            #clipping_fn=privacy_args.clipping_fn,
            clipping_style="layer-wise",
            origin_params=None,  # !
            num_GPUs=num_GPUs,
            torch_seed_is_fixed=True,
            # torch_seed_is_fixed=privacy_args.torch_seed_is_fixed,
        )
        
        # Originally, these could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        try:
            trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        finally:
            print(f"saving to {trainer.args.output_dir}")
            trainer.save_model(trainer.args.output_dir)  # only save from main_process
            # don't use: # trainer.model.save_pretrained(trainer.args.output_dir)  
            trainer.model.config.save_pretrained(trainer.args.output_dir)
            trainer.model.generation_config.save_pretrained(trainer.args.output_dir)
            # print(f"NOT saving to {trainer.args.output_dir}... Quit")
        self._lm.eval()

    def _fine_tune(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        """
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        print("Done Tokenizing!")

        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)

        try:
            trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        finally:
            print(f"training ended. Saving to {trainer.args.output_dir}")
            
            trainer.save_model(trainer.args.output_dir)  # only save from main_process
            # don't use: # trainer.model.save_pretrained(trainer.args.output_dir)  
            trainer.model.config.save_pretrained(trainer.args.output_dir)
            trainer.model.generation_config.save_pretrained(trainer.args.output_dir)
        self._lm.eval()
