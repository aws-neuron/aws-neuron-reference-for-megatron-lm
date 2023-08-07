#coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications copyright Amazon Web Services and its affiliates. All rights reserved.

"""Pretrain utilities."""

from collections import namedtuple
from datetime import datetime, timezone
import os
import json
import math
import sys
import time
import shutil
from typing import Any, Dict, List
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron import print_rank_2D
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import Checkpoint
from megatron.model import Float16Module
from megatron.model import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import get_forward_backward_func
from megatron.utils import report_memory, unload_all_models
import queue
from os import path
import numpy as np

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
torch.cuda.DoubleTensor = lambda t: torch.DoubleTensor(t).to(xm.xla_device())
torch.cuda.FloatTensor = lambda t: torch.FloatTensor(t).to(xm.xla_device())
torch.cuda.IntTensor = lambda t: torch.IntTensor(t).to(xm.xla_device())
torch.cuda.LongTensor = lambda t: torch.LongTensor(t).to(xm.xla_device())
torch.cuda.current_device = lambda: xm.xla_device()

stats = {'iteration': [], 'consumed_samples': [], 'time': [], 'learning_rate': [], 'global_batch_size': [], 'lm_loss': [], 'loss_scale': [], 'grad_norm': [], 'skipped_iterations': [], 'nan_iterations': [],'forward-compute': [], 'backward-compute': [], 'backward-params-all-reduce': [], 'backward-embedding-all-reduce': [], 'optimizer-copy-to-main-grad': [], 'optimizer-unscale-and-check-inf': [], 'optimizer-clip-main-grad': [], 'optimizer-copy-main-to-model-params': [], 'optimizer': [], 'batch-generator': [], 'params_norm':[]}
Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])


class TrainingMetrics:
    def __init__(self,json_file):
        self.json_file = json_file

    def read_modify_write_file(self, data, key: str = "metrics") -> None:
        """
        data (dict of training parameters or list of metrics): Data to update in the file.
        key (str): the dictionary key under which data is to be recorded
        """
        result_dict = {}
        print(f"Writing data to the provided results file: {self.json_file}")
        if os.path.exists(self.json_file):
            with open(self.json_file) as json_file:
                result_dict = json.loads(json_file.read()) or result_dict
        print(f"Updating with {key} data: {data}")
        if result_dict:
            try:
                # handle internal named entity if present
                results = result_dict[next(iter(result_dict))]
            except Exception:
                results = result_dict
            current = results.get(key)
            if not current:
                results[key] = data
            else:
                if isinstance(current, list):
                    current.extend(data)
                elif isinstance(current, dict):
                    current.update(data)
        else:
            result_dict["results"] = {key: data}
        with open(self.json_file, 'w') as json_file:
            json.dump(result_dict, json_file)

    def store_metrics(self, metrics: List[Metric]) -> None:
        """
        Writes collected metrics to the file.

        """
        data = [
            {
                "MetricName": metric.name,
                "MeasuredValue": metric.value,
                "Units": metric.units,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "AdditionalData": metric.additional_data,
            } for metric in metrics
        ]
        self.update(data=data, key="metrics")

    def store_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Writes specified model and configuration parameters to the file.

        """
        self.update(data=parameters, key="parameters")

    def update(self, **kwargs: Any) -> None:
        """
        Write specified data to the output file.
        """
        self.read_modify_write_file(**kwargs)


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, allow_no_cuda=True)
#                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    #global _TRAIN_START_TIME
    #start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    #start_time_tensor = xm.all_reduce('min', start_time_tensor)
    #torch.distributed.all_reduce(start_time_tensor,
    #                             op=torch.distributed.ReduceOp.MIN)
    #_TRAIN_START_TIME = start_time_tensor.item()
    #print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
    #    time.time() - _TRAIN_START_TIME))
    #print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    #Don't compile any hlos for dataset generation
    if args.gpt_dataset_generation_only == True:
        os.environ['NEURON_EXTRACT_GRAPHS_ONLY'] = "1"
        os.environ['NEURON_FALL_BACK_TO_NULL_NEFF'] = "1"

    #Only generate dataset for rank 0
    if mpu.get_tensor_model_parallel_rank() != 0 and args.gpt_dataset_generation_only == True:
        return

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider,
                                                               model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup').start()
    if (args.gpt_dataset_generation_only == True and mpu.get_tensor_model_parallel_rank() == 0) \
        or args.gpt_dataset_generation_only == False:
        if args.virtual_pipeline_model_parallel_size is not None:
            all_data_iterators = [
                build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
                for _ in range(len(model))
            ]
            train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
            valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
            test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator \
                = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])
    print_rank_0('training ...')

    #Do not start training step if we only want to generate dataset
    if args.gpt_dataset_generation_only == True:
        return

    iteration = 0
    #if args.do_train and args.train_iters > 0:
    #TODO: after data iterators done, check for args.do_train
    if args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, False)

    #if args.save and iteration != 0:
    #    save_checkpoint(iteration, model, optimizer, lr_scheduler)

    #if args.do_test:
    #    # Run on test data.
    #    prefix = 'the end of training for test data'
    #    evaluate_and_print_results(prefix, forward_step_func,
    #                               test_data_iterator, model,
    #                               0, True)

    return stats


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        parameters = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            parameters), flush=True)
        if torch.distributed.get_rank() == 0:
            if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
                tm = TrainingMetrics("/tmp/test_dict.json")
                # TODO: when pipeline parallel support comes in, need to aggregate parameters instead of
                # just multiplying by tensor parallel degree
                tm.store_parameters(
                    {"Parameters": f"{round(parameters * args.tensor_model_parallel_size / 1e9, 1)}B"}
                )

    # GPU allocation.
    for model_module in model:
        #model_module.cuda(torch.cuda.current_device())
        model_module.to(xm.xla_device())

    # TODO: Add cmdline arg to choose to do all reduce in CPU vs device
    assert len(model) == 1 #Hack restriction, only expecting one model per rank
    model[0].may_sync_initial_word_embeddings()

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            print('accum grads in fp32:{}, use_contiguous_buffers:{},'.format(args.accumulate_allreduce_grads_in_fp32,args.use_contiguous_buffers_in_local_ddp))
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]

        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func, model_type):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)

    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if (args.load or args.load_xser) and not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        load_arg='load' if args.load is not None else 'load_xser'
        timers = get_timers()
        timers('load-checkpoint').start()
        #Staggering load checkpoints
        for i in range(0, mpu.get_tensor_model_parallel_world_size()):
            if mpu.get_tensor_model_parallel_rank() == i:
                print_rank_2D("+ Loading from checkpoint!")
                args.iteration = load_checkpoint(model, optimizer, lr_scheduler, load_arg=load_arg)
                print_rank_2D(f'Finished Loading Phase-{i}')
            xm.rendezvous(f'load-chkpt-phase-{i}')
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0

    print_rank_0("Finished loading from checkpoint")

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if 'TORCHXLA_PROFILE_PORT' in os.environ :
      port = int(os.environ['TORCHXLA_PROFILE_PORT'])
      import torch_xla.debug.profiler as xp
      server = xp.start_server(port)

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    #Allowing this alternate way of zero'ing gradient results in a NEFF reduction
    #optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=False)

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only=False)
    # Empty unused memory
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    xm.mark_step() #adding this mark step alleviates memory pressure

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('backward-params-all-reduce').start()
        for model_module in model:
            model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce').start()
    if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            unwrapped_model = model[0]
        elif mpu.is_pipeline_last_stage(ignore_virtual=True):
            unwrapped_model = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            unwrapped_model = model[0]
        unwrapped_model = unwrap_model(
            unwrapped_model, (torchDDP, LocalDDP, Float16Module))

        if unwrapped_model.share_word_embeddings:
            word_embeddings_weight = unwrapped_model.word_embeddings_weight()
            if args.DDP_impl == 'local':
                grad = word_embeddings_weight.main_grad
            else:
                grad = word_embeddings_weight.grad
            torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
    timers('backward-embedding-all-reduce').stop()
    # Update parameters.
    timers('optimizer').start()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        lr_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


class Throughput:
    def __init__(self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()
        self.throughput_peak = 0
        self.throughput_sum = 0
        self.throughputs = []

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        self.throughputs.append(throughput)
        return throughput


def training_markstep_closure(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad, thr, golden_loss,
                 model, optimizer, lr_scheduler, ckpts):
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    assert args.tensorboard_log_interval == args.log_interval, "We want both intervals to overlap to synchronize fetching Loss tensors from device!!"

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'

    assert skipped_iter == 0

    master_tprank0_only = mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0

    # Checkpointing
    saved_checkpoint = False
    if (args.save or args.save_xser) and args.save_interval and \
        iteration % args.save_interval == 0 and \
        not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        cp_path = save_checkpoint_and_time(iteration, model, optimizer,
                                    lr_scheduler)
        saved_checkpoint = True

        if master_tprank0_only:
            ckpts.checkpoint_list.append(os.path.dirname(os.path.dirname(cp_path)))
            print_rank_2D(f"Added checkpoint {cp_path} to tracking list, now at {ckpts.num_checkpoints()} count.")
            # Keep only most recent checkpoint if flag set
            if ckpts.num_checkpoints() > 1 and args.keep_last_checkpoint_only:
                print_rank_2D(f"Keeping last checkpoint only (--keep_last_checkpoint_only).")
                ckpts.keep_recent_checkpoint()

    xm.rendezvous(f'End checkpoint deletion')

    # Advanced iterations.
    total_loss_dict[advanced_iters_key] = total_loss_dict.get(
        advanced_iters_key, 0) + 1

    # Return immediately and avoid device synchronization
    if iteration % args.tensorboard_log_interval != 0:
        return

    """Log training information such as losses, timing, ...."""
    # Add this to copy loss tensors to cpu:
    loss_dict = {key: value.cpu().item() for key, value in loss_dict.items()}
    grad_norm = grad_norm.cpu().item() if grad_norm is not None else grad_norm
    #params_norm = params_norm.item() if params_norm is not None else params_norm
    num_zeros_in_grad = num_zeros_in_grad.item() if num_zeros_in_grad is not None else num_zeros_in_grad
    loss_scale = loss_scale.item() if loss_scale is not None else loss_scale

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key]

    throughput = thr.get_throughput()
    throughput_peak = thr.throughput_peak
    thr.throughput_sum += throughput
    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0 ) and \
       is_last_rank():
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
        if throughput is not None:
            writer.add_scalar('throughput', throughput, iteration)
            writer.add_scalar('throughput vs samples', throughput,
                              args.consumed_train_samples)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed()
        elapsed_time_per_iteration = elapsed_time / total_iterations
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        stats['iteration'].append(iteration)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        stats['consumed_samples'].append(args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        stats['time'].append(elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        stats['learning_rate'].append(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        stats['global_batch_size'].append(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key]:
                avg = total_loss_dict[key].cpu().item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                    stats['lm_loss'].append(avg)
                #compare with golden for testing
                if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
                    step_0start = iteration - 1
                    if step_0start < len(golden_loss) and step_0start >= 0:
                        np.testing.assert_allclose(avg, float(golden_loss[step_0start]), rtol=2e-1)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
            stats['grad_norm'].append(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
            stats['num_zeros'].append(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
            stats['params_norm'].append(params_norm)
        log_string += ' throughput: {:.3f} |'.format(throughput)
        if throughput > throughput_peak:
            thr.throughput_peak = throughput
        total_loss_dict[advanced_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        if is_last_rank():
            data_parallel_degree = mpu.get_data_parallel_world_size()
            num_workers = data_parallel_degree * args.tensor_model_parallel_size * args.pipeline_model_parallel_size
            last_loss = 0.0
            if len(stats['lm_loss']) > 0:
                last_loss = stats['lm_loss'][-1]
            microsteps = iteration * args.global_batch_size // args.micro_batch_size
            # TODO: for some reason, args.global_batch_size is emitted as 1 even
            # though it is set to 64; is it related to batch size rampup?.
            tm = TrainingMetrics("/tmp/test_dict.json")
            additional_data = {"Iteration": iteration, "Microstep": microsteps}
            metric_data = [
                Metric("Loss", round(last_loss, 4), "", additional_data),
                Metric("Throughput", round(throughput, 4), "seq/s", additional_data),
            ]
            tm.store_metrics(metric_data)
            tm.store_parameters(
                {
                    "Workers": num_workers,
                    "Data parallel degree": mpu.get_data_parallel_world_size(),
                    "Pipeline parallel degree": args.pipeline_model_parallel_size,
                    "Tensor parallel degree": args.tensor_model_parallel_size,
                    "Batch size": args.global_batch_size,
                    "Micro-batch size": args.micro_batch_size,
                    "Sequence length": args.seq_length,
                    "Data type": str(args.params_dtype),
                    "Learning rate": learning_rate,
                    "Layers": args.num_layers,
                    "Hidden size": args.hidden_size,
                    # TODO: add parameters (6.7B or 5B)
                    "Number attention heads": args.num_attention_heads,
                    "Max positional embeddings size": args.max_position_embeddings,
                    "Environment variables": {
                        variable: value
                        for variable, value in os.environ.items()
                        if variable.startswith("NEURON") or variable.startswith("XLA")
                    },
                    "Iterations": args.train_iters,
                    "Model": "Megatron-LM GPT",
                    "World size": args.world_size,
                }
            )
    if iteration == args.train_iters and is_last_rank():
        tm = TrainingMetrics("/tmp/test_dict.json")
        additional_data = {"Iteration": iteration, "Microstep": microsteps}
        metric_data = [
            Metric("Final loss", round(last_loss, 4), "", additional_data),
            Metric("Peak throughput", round(thr.throughput_peak, 4), "seq/s", additional_data),
            Metric("Average throughput", round(sum(thr.throughputs)/len(thr.throughputs), 4), "seq/s", additional_data)
        ]
        tm.store_metrics(metric_data)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers('save-checkpoint').start()
    cp_path = save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers('save-checkpoint').stop()
    timers.log(['save-checkpoint'])
    return cp_path


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers('interval-time').start()
    print_datetime('before the start of training step')
    report_memory_flag = False # Avoid unnecessary resource prints. To add trn1 resources query once available..
    #throughput = Throughput(args.micro_batch_size, get_num_microbatches(), args.global_batch_size, args.tensorboard_log_interval)
    throughput = Throughput(args.micro_batch_size, mpu.get_data_parallel_world_size(), get_num_microbatches(), args.tensorboard_log_interval)
    ckpts = Checkpoint()
    golden_loss_file = "./golden_loss.txt"
    golden_loss = []
    if path.exists(golden_loss_file):
        with open(golden_loss_file, "r") as golden:
            golden_loss = golden.readlines()
        print("Loaded {} loss values from {}".format(str(len(golden_loss)), golden_loss_file))

    iteration_d = torch.IntTensor([0]).to(xm.xla_device())

    while iteration <= args.train_iters:
        update_num_microbatches(args.consumed_train_samples)
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       lr_scheduler)
        iteration += 1
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()

        # Logging.
        loss_scale = optimizer.get_loss_scale()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        for key in loss_dict:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.FloatTensor([0.0]).to(xm.xla_device()))
            total_loss_dict[key] = torch.where(iteration_d % args.log_interval == 0, loss_dict[key].detach(), loss_dict[key].detach() + total_loss_dict[key].detach())
        iteration_d = iteration_d.detach() + 1

        xm.add_step_closure(training_markstep_closure, (loss_dict, total_loss_dict, optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale, report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad, throughput, golden_loss,
                                          model, optimizer, lr_scheduler, ckpts))

        if mpu.get_pipeline_model_parallel_world_size() > 1 and (iteration == 1 or iteration == 64):
            # TODO: Currently torch-xla generates multiple graphs for same purpose.
            # For example, there are 4-5 graphs of optimizer, each differing in a single
            # line. Each model ends up taking some memory resulting in device OOM. To tack this
            # we unload all the models at step 64 (this is a place we see new compiles) and load
            # fresh set of steady state models. This way we only have one copy of each part in
            # memory. Fix the issue related to compile on one line difference and remove
            # the below lines.
            unload_all_models()

        #XLA uses add_step_closure instead
        #report_memory_flag = training_markstep_closure(loss_dict, total_loss_dict,
        #                                  optimizer.param_groups[0]['lr'],
        #                                  iteration, loss_scale,
        #                                  report_memory_flag, skipped_iter,
        #                                  grad_norm, params_norm, num_zeros_in_grad)
        # Autoresume
        #if args.adlr_autoresume and \
        #   (iteration % args.adlr_autoresume_interval == 0):
        #    check_adlr_autoresume_termination(iteration, model, optimizer,
        #                                      lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
          args.do_valid:
           prefix = 'iteration {}'.format(iteration)
           evaluate_and_print_results(prefix, forward_step_func,
                                      valid_data_iterator, model,
                                      iteration, False)

        # Exiting based on duration
        #if args.exit_duration_in_mins:
        #    train_time = (time.time() - _TRAIN_START_TIME) / 60.0
        #    done_cuda = torch.cuda.IntTensor(
        #        [train_time > args.exit_duration_in_mins])
        #    torch.distributed.all_reduce(
        #        done_cuda, op=torch.distributed.ReduceOp.MAX)
        #    done = done_cuda.item()
        #    if done:
        #        if not saved_checkpoint:
        #            save_checkpoint_and_time(iteration, model, optimizer,
        #                                     lr_scheduler)
        #        print_datetime('exiting program after {} minutes'.format(train_time))
        #        sys.exit()

        # Exiting based on iterations
        #if args.exit_interval and iteration % args.exit_interval == 0:
        #    if not saved_checkpoint:
        #        save_checkpoint_and_time(iteration, model, optimizer,
        #                                 lr_scheduler)
        #    torch.distributed.barrier()
        #    print_datetime('exiting program at iteration {}'.format(iteration))
        #    sys.exit()

    xm.rendezvous(f'Training Done')
    return iteration

def evaluate_markstep_closure(prefix, total_loss_dict, iteration):

    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = {key: value.cpu().item() for key, value in total_loss_dict.items()}
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key])
        ppl = math.exp(min(20, total_loss_dict[key]))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key],
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key],
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)

def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            forward_backward_func = get_forward_backward_func()
            loss_dicts = forward_backward_func(
                forward_step_func, data_iterator, model, optimizer=None,
                timers=None, forward_only=True)

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    xm.add_step_closure(evaluate_markstep_closure, (prefix, total_loss_dict, iteration))


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    #if mpu.get_tensor_model_parallel_rank() == 0:
    if True:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        device = xm.xla_device()
        #global _TRAIN_START_TIME
        #start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
        #dummy_sync = torch.DoubleTensor([1.0]).to(device)
        ## Build the datasets.
        #if mpu.get_tensor_model_parallel_rank() == 0:
        #    train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
        #        train_val_test_num_samples)
        #    torch.distributed.all_reduce(dummy_sync, async_op=True)
        #    xm.mark_step()
        #else:
        #    torch.distributed.all_reduce(dummy_sync, async_op=True)
        #    xm.mark_step()
        #    train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
        #        train_val_test_num_samples)
#
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        train_device_dataloader = pl.MpDeviceLoader(
            train_dataloader, device,
            batches_per_execution = get_num_microbatches() if mpu.get_pipeline_model_parallel_world_size() > 1 else 1)
        valid_device_dataloader = pl.MpDeviceLoader(valid_dataloader, device)
        test_device_dataloader = pl.MpDeviceLoader(test_dataloader, device)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        #flags = torch.cuda.LongTensor(
        #    [int(do_train), int(do_valid), int(do_test)])
        args.do_train = do_train
        args.do_valid = do_valid
        args.do_test = do_test
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    #(TODO)do need  this in future
    #torch.distributed.broadcast(flags,
    #                            mpu.get_tensor_model_parallel_src_rank(),
    #                            group=mpu.get_tensor_model_parallel_group())
    #args.do_train = flags[0].item()
    #args.do_valid = flags[1].item()
    #args.do_test = flags[2].item()


    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        #train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
        #                      else iter(cyclic_iter(train_dataloader))
        train_device_data_iterator = iter(train_device_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_device_dataloader))
    else:
        #train_data_iterator = None
        train_device_data_iterator = None

    if valid_dataloader is not None:
        #valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
        #                      else iter(cyclic_iter(valid_dataloader))
        valid_device_data_iterator = iter(valid_device_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_device_dataloader))
    else:
        #valid_data_iterator = None
        valid_device_data_iterator = None

    if test_dataloader is not None:
        #test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
        #                     else iter(cyclic_iter(test_dataloader))
        test_device_data_iterator = iter(test_device_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_device_dataloader))
    else:
        #test_data_iterator = None
        test_device_data_iterator = None

    #return train_data_iterator, valid_data_iterator, test_data_iterator
    return train_device_data_iterator, valid_device_data_iterator, test_device_data_iterator


