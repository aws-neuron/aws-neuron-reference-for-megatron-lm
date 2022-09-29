# coding=utf-8
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

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np
import torch
import torch_xla.core.xla_model as xm

from megatron import (get_args,
                      mpu,
                      print_rank_0,
                      print_rank_2D,
                      update_num_microbatches,
                      utils)

_CHECKPOINT_VERSION = None
_ARGS_TO_SAVE =  { 'num_layers', 'hidden_size', 'num_attention_heads', 'tensor_model_parallel_size', \
                    'consumed_train_samples', 'consumed_valid_samples', 'pipeline_model_parallel_size' }


def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, \
            "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None):
        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name)
        else:
            #checkpoint_value = getattr(checkpoint_args, arg_name)
            checkpoint_value = checkpoint_args[arg_name]
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument value ({}).'.format(
                            arg_name, checkpoint_value, args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    if args.vocab_file:
        _compare('max_position_embeddings')
        _compare('make_vocab_size_divisible_by')
        _compare('padded_vocab_size')
        _compare('tokenizer_type')
    if get_checkpoint_version() < 3.0:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0:
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration,
                        release=False):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # Use both the tensor and pipeline MP rank.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}'.format(
                                mpu.get_tensor_model_parallel_rank()),
                            'model_optim_rng.pt')
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}_{:03d}'.format(
                            mpu.get_tensor_model_parallel_rank(),
                            mpu.get_pipeline_model_parallel_rank()),
                        'model_optim_rng.pt')


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def read_metadata(tracker_filename):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    max_iter = iteration
    # Commenting out the code as it deadlocks with staggered checkpointing strategy
    # Get the max iteration retrieved across the ranks.
    # iters_cuda = torch.cuda.LongTensor([iteration])
    # torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX, async_op=True)
    # max_iter = iters_cuda[0].item()

    # We should now have all the same iteration.
    # If not, print a warning and chose the maximum
    # iteration across all ranks.
    if iteration != max_iter:
        print('WARNING: on rank {} found iteration {} in the '
              'metadata while max iteration across the ranks '
              'is {}, replacing it with max iteration.'.format(
                  rank, iteration, max_iter), flush=True)
    return max_iter, release


def save(data, file_or_path):
  """Saves the input data into a file.

  The saved data is transferred to PyTorch CPU device before being saved, so a
  following `torch.load()` will load CPU data.
  Care must be taken when working with views. Instead of saving views it's
  recommended that you recreate them after the tensors have been loaded and
  moved to their destination device(s).

  Args:
    data: The input data to be saved. Any nested combination of Python objects
      (list, tuples, sets, dicts, ...).
    file_or_path: The destination for the data saving operation. Either a file
      path or a Python file object. If `master_only` is ``False`` the path or
      file objects must point to different destinations as otherwise all the
      writes from the same host will override each other.
  """
  should_chkpt = True if mpu.get_data_parallel_rank() == 0 else False
  cpu_data = xm._maybe_convert_to_cpu(data, convert=should_chkpt)

  for tp_rank in range(0, mpu.get_tensor_model_parallel_world_size()):
      my_tp_rank = mpu.get_tensor_model_parallel_rank()
      should_write_data = True if mpu.get_data_parallel_rank() == 0 and my_tp_rank == tp_rank else False

      #Staggering save checkpoints
      if should_write_data:
          print_rank_2D('file_or_path:{}'.format(file_or_path))
          ensure_directory_exists(file_or_path)
          torch.save(cpu_data, file_or_path)

      xm.rendezvous(f'chktp-save-{tp_rank}')

      if should_write_data:
          print_rank_2D('successfully saved checkpoint')


def save_checkpoint(iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = utils.unwrap_model(model)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # Arguments, iteration, and model.
    state_dict = {}
    args_dict = vars(args)

    #saving only necessary args, TODO(Debug Checkpointing generating extra hlos)
    for arg in _ARGS_TO_SAVE:
        state_dict[arg] = args_dict[arg]
    if args.vocab_file:
        state_dict['max_position_embeddings'] = args_dict['max_position_embeddings']
        state_dict['make_vocab_size_divisible_by'] = args_dict['make_vocab_size_divisible_by']
        state_dict['padded_vocab_size'] = args_dict['padded_vocab_size']
        state_dict['tokenizer_type'] = args_dict['tokenizer_type']

    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = iteration
    
    if len(model) == 1:
        state_dict['model'] = model[0].state_dict_for_save_checkpoint()
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()

    # RNG states.
    if not args.no_save_rng:
        print_rank_0("saving_rng")
        state_dict['random_rng_state'] = random.getstate()
        rng_state = state_dict['random_rng_state']
        print_rank_0(str(type(state_dict['random_rng_state'])))
        print_rank_0("after rng type")
        for r in rng_state:
            print_rank_0(str(type(r)))
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
        state_dict['rng_tracker_states'] \
            = mpu.get_cuda_rng_tracker().get_states()

    checkpoint_name = get_checkpoint_name(args.save, iteration)
    master_only = mpu.get_data_parallel_rank() == 0
    if master_only:
        print_rank_2D('checkpoint_name:{}'.format(checkpoint_name))
        ensure_directory_exists(checkpoint_name)
    
    if args.save_xser:
        xser.save(state_dict, checkpoint_name, (not master_only), global_master=True)
    else:
        save(state_dict, checkpoint_name)
    # we don;t need this barrier as save above has it
    # Wait so everyone is done (necessary)
    #if torch.distributed.is_initialized():
    #    torch.distributed.barrier()

    # And update the latest iteration
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Replace torch barrier with rendezvous
    # Wait so everyone is done (not necessary)
    #if torch.distributed.is_initialized():
    #    torch.distributed.barrier()
    xm.rendezvous('Checkpoint Done')

def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, 'module'):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_splits, num_attention_heads_per_partition,
             hidden_size_per_attention_head) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_attention_heads_per_partition,
             hidden_size_per_attention_head, num_splits) +\
             input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t

def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model)==1
            model = model[0]
        for name, param in model.named_parameters():
            if name.endswith(('.query_key_value.weight', '.query_key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith(('.key_value.weight', '.key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(" succesfully fixed query-key-values ordering for"
                    " checkpoint version {}".format(checkpoint_version))

def load_checkpoint(model, optimizer, lr_scheduler, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    # print_rank_0("loading")
    args = get_args()
    load_dir = getattr(args, load_arg)

    model = utils.unwrap_model(model)

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    print_rank_0(f' loading checkpoint from {args.load} at iteration {iteration}')

    # Load the checkpoint.
    try:
        if args.load_xser:
            state_dict = xser.load(checkpoint_name)
        else:    
            state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        from megatron.fp16_deprecated import loss_scaler
        # For backward compatibility.
        print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        if args.load_xser:
            state_dict = xser.load(checkpoint_name)
        else:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    # set checkpoint version
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=strict)
    
    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0

    #if 'args' in state_dict:
    # loading only the args we need to verify
    checkpoint_args = {}
    for k, v in state_dict.items():
        if k in _ARGS_TO_SAVE:
            checkpoint_args[k] = v

    #vocab file only args
    if args.vocab_file:
        checkpoint_args['max_position_embeddings'] = state_dict['max_position_embeddings']
        checkpoint_args['make_vocab_size_divisible_by'] = state_dict['make_vocab_size_divisible_by']
        checkpoint_args['padded_vocab_size'] = state_dict['padded_vocab_size']
        checkpoint_args['tokenizer_type'] = state_dict['tokenizer_type']
        
    check_checkpoint_args(checkpoint_args)
    args.consumed_train_samples = int(getattr(checkpoint_args,
                                            'consumed_train_samples', 0))
    update_num_microbatches(consumed_samples=args.consumed_train_samples)
    args.consumed_valid_samples = int(getattr(checkpoint_args,
                                            'consumed_valid_samples', 0))
    #else:
    #    print_rank_0('could not find arguments in the checkpoint ...')

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            #Converting random rng state from list to tuple for compatibility
            #(TODO) Error when loading rng random states checkpointing 
            rng_state = state_dict['random_rng_state']
            for i in range(len(rng_state)):
                if type(rng_state[i]) is list:
                    rng_state[i] = tuple(rng_state[i])
            print_rank_0(tuple(rng_state))
            random.setstate(tuple(rng_state))
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            # Check for empty states array
            if not state_dict['rng_tracker_states']:
                raise KeyError
            mpu.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {args.load} '
                 f'at iteration {iteration}')

    return iteration


def load_biencoder_checkpoint(model, only_query_model=False,
        only_context_model=False, custom_load_path=None):
    """
    selectively load retrieval models for indexing/retrieving 
    from saved checkpoints
    """

    args = get_args()

    model = utils.unwrap_model(model)

    load_path = custom_load_path if custom_load_path is not None else args.load

    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    checkpoint_name = get_checkpoint_name(load_path, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ret_state_dict = state_dict['model']

    if only_query_model:
        ret_state_dict.pop('context_model')
    if only_context_model:
        ret_state_dict.pop('query_model')

    assert len(model) == 1
    model[0].load_state_dict(ret_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


