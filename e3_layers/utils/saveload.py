from typing import List, Tuple, Union
import sys
import logging
import contextlib
import contextvars
import tempfile
from pathlib import Path
import shutil
import os
import ase
import torch


# accumulate writes to group for renaming
_MOVE_SET = contextvars.ContextVar("_move_set", default=None)

def saveMol(batch, type_names=None, idx=0, workdir='', filename='tmp'):
    """
    saves a molecule in gromacs
    """
    if type_names is None:
        type_names = list(ase.atom.atomic_numbers.keys())
    lines = []
    idx = 0
    lines.append('title')
    lines.append(f'{batch["_n_nodes"][idx].item()}')
    for i in range(batch['_n_nodes'][idx]):
        species = type_names[batch[idx]['species'][i]]
        line = f"{1:>5}{f'none':>5}{species:>5}{i:>5}"
        x, y, z = batch[idx]['pos'][i]*0.1 # A to nm
        line += f'{x:>8.3f}{y:>8.3f}{z:>8.3f}'
        line += f'{0.:>8.4f}{0.:>8.4f}{0.:>8.4f}'
        lines.append(line)
 #   lines.append('0 0 0')
    filename = os.path.join(workdir, filename)+'.gro'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    return filename
      
def saveProtein(batch, workdir, idx=0, filename='tmp'):
    codification = ['ALA',
    'ARG',
    'ASP',
    'ASN',
    'CYS',
    'GLU',
    'GLN',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL', 'UNK']
    aa_ids = {i:key for i, key in enumerate(codification)}
    def id2name(x):
        return aa_ids[x]
    filename = os.path.join(workdir, filename)+'.pdb'
    item = batch[idx]
    with open(filename, "w") as f:
        for i in range(item['_n_nodes']):
            for j, key in enumerate(['C', 'N', 'CA', 'O']):
                # https://cupnet.net/pdb-format/
                atom = 'ATOM'
                atom_id = i*4+j
                res = id2name(item['species'][i].item())
                
                if 'id' in item:
                    res_id = item['id'][i].item()+1
                else:
                    res_id = i+1
                x, y, z = item[key][i]
                x, y, z = x.item(), y.item(), z.item()
                chain_id = chr(ord('A')+item['chain_id'][i].item())

                line = f"{atom:6s}{atom_id:5d} {key:^4s} {res:3s} {chain_id:1s}{res_id:4d}{'':1s}"
                line += f"   {x:8.3f}{y:8.3f}{z:8.3f}{0:6.2f}{0:6.2f}          {key[0]:>2s}{'':2s}\n"
                f.write(line)
        f.write('TER\nEND\n')
    return filename

def _delete_files_if_exist(paths):
    # clean up
    # better for python 3.8 >
    if sys.version_info[1] >= 8:
        for f in paths:
            f.unlink(missing_ok=True)
    else:
        # race condition?
        for f in paths:
            if f.exists():
                f.unlink()


def _process_moves(moves: List[Tuple[bool, Path, Path]]):
    """blocking to copy (possibly across filesystems) to temp name; then atomic rename to final name"""
    try:
        for _, from_name, to_name in moves:
            # blocking copy to temp file in same filesystem
            tmp_path = to_name.parent / (f".tmp-{to_name.name}~")
            shutil.move(from_name, tmp_path)
            # then atomic rename to overwrite
            tmp_path.rename(to_name)
    finally:
        _delete_files_if_exist([m[1] for m in moves])


# allow user to enable/disable depending on their filesystem
_ASYNC_ENABLED = "true"
assert _ASYNC_ENABLED in ("true", "false")
_ASYNC_ENABLED = _ASYNC_ENABLED == "true"

if _ASYNC_ENABLED:
    import threading
    from queue import Queue

    _MOVE_QUEUE = Queue()
    _MOVE_THREAD = None

    # Because we use a queue, later writes will always (correctly)
    # overwrite earlier writes
    def _moving_thread(queue):
        while True:
            moves = queue.get()
            _process_moves(moves)
            # logging is thread safe: https://stackoverflow.com/questions/2973900/is-pythons-logging-module-thread-safe
            logging.debug(f"Finished writing {', '.join(m[2].name for m in moves)}")
            queue.task_done()

    def _submit_move(from_name, to_name, blocking: bool):
        global _MOVE_QUEUE
        global _MOVE_THREAD
        global _MOVE_SET

        # launch thread if its not running
        if _MOVE_THREAD is None:
            _MOVE_THREAD = threading.Thread(
                target=_moving_thread, args=(_MOVE_QUEUE,), daemon=True
            )
            _MOVE_THREAD.start()

        # check on health of copier thread
        if not _MOVE_THREAD.is_alive():
            _MOVE_THREAD.join()  # will raise exception
            raise RuntimeError("Writer thread failed.")

        # submit this move
        obj = (blocking, from_name, to_name)
        if _MOVE_SET.get() is None:
            # no current group
            _MOVE_QUEUE.put([obj])
            # if it should be blocking, wait for it to be processed
            if blocking:
                _MOVE_QUEUE.join()
        else:
            # add and let the group submit and block (or not)
            _MOVE_SET.get().append(obj)

    @contextlib.contextmanager
    def atomic_write_group():
        global _MOVE_SET
        if _MOVE_SET.get() is not None:
            # nesting is a no-op
            # submit along with outermost context manager
            yield
            return
        token = _MOVE_SET.set(list())
        # run the saves
        yield
        _MOVE_QUEUE.put(_MOVE_SET.get())  # send it off
        # if anyone is blocking, block the whole group:
        if any(m[0] for m in _MOVE_SET.get()):
            # someone is blocking
            _MOVE_QUEUE.join()
        # exit context
        _MOVE_SET.reset(token)

    def finish_all_writes():
        global _MOVE_QUEUE
        _MOVE_QUEUE.join()
        # ^ wait for all remaining moves to be processed

else:

    def _submit_move(from_name, to_name, blocking: bool):
        global _MOVE_SET
        obj = (blocking, from_name, to_name)
        if _MOVE_SET.get() is None:
            # no current group just do it
            _process_moves([obj])
        else:
            # add and let the group do it
            _MOVE_SET.get().append(obj)

    @contextlib.contextmanager
    def atomic_write_group():
        global _MOVE_SET
        if _MOVE_SET.get() is not None:
            # don't nest them
            yield
            return
        token = _MOVE_SET.set(list())
        yield
        _process_moves(_MOVE_SET.get())  # do it
        _MOVE_SET.reset(token)

    def finish_all_writes():
        pass  # nothing to do since all writes blocked


@contextlib.contextmanager
def atomic_write(
    filename: Union[Path, str, List[Union[Path, str]]],
    blocking: bool = True,
    binary: bool = False,
):
    aslist: bool = True
    if not isinstance(filename, list):
        aslist = False
        filename = [filename]
    filename = [Path(f) for f in filename]

    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context(
                tempfile.NamedTemporaryFile(
                    mode="w" + ("b" if binary else ""), delete=False
                )
            )
            for _ in filename
        ]
        try:
            if not aslist:
                yield files[0]
            else:
                yield files
        except:  # noqa
            # ^ noqa cause we want to delete them no matter what if there was a failure
            # only remove them if there was an error
            _delete_files_if_exist([Path(f.name) for f in files])
            raise

        for tp, fname in zip(files, filename):
            _submit_move(Path(tp.name), Path(fname), blocking=blocking)


def save_file(
    item,
    filename: str,
    enforced_format: str = None,
    blocking: bool = True,
    supported_formats: dict=dict(torch=["pth", "pt"], yaml=["yaml", "yml"],
                                 json=["json"], pickle=["pickle", "pkl"], npz=["npz"])
):
    """
    Save file. It can take yaml, json, pickle, json, npz and torch save
    """

    # check whether folder exist
    path = os.path.dirname(os.path.realpath(filename))
    if not os.path.isdir(path):
        logging.debug(f"save_file make dirs {path}")
        os.makedirs(path, exist_ok=True)

    format, filename = adjust_format_name(
        supported_formats=supported_formats,
        filename=filename,
        enforced_format=enforced_format,
    )

    with atomic_write(
        filename,
        blocking=blocking,
        binary={
            "json": False,
            "yaml": False,
            "pickle": True,
            "torch": True,
            "npz": True,
        }[format],
    ) as write_to:
        if format == "json":
            import json

            json.dump(item, write_to)
        elif format == "yaml":
            import yaml

            yaml.dump(item, write_to)
        elif format == "torch":
            import torch

            torch.save(item, write_to)
        elif format == "pickle":
            import pickle

            pickle.dump(item, write_to)
        elif format == "npz":
            import numpy as np

            np.savez(write_to, item)
        else:
            raise NotImplementedError(
                f"Output format {format} not supported:"
                f" try from {supported_formats.keys()}"
            )

    return filename


def load_file(filename: str, enforced_format: str = None,
              supported_formats: dict=dict(torch=["pth", "pt"], yaml=["yaml", "yml"],
                                           json=["json"], pickle=["pickle", "pkl"], npz=["npz"])):
    """
    Load file. Current support form
    """
    if enforced_format is None:
        format = match_suffix(supported_formats=supported_formats, filename=filename)
    else:
        format = enforced_format

    if not os.path.isfile(filename):
        abs_path = str(Path(filename).resolve())
        raise OSError(f"file {filename} at {abs_path} is not found")

    if format == "json":
        import json

        with open(filename) as fin:
            return json.load(fin)
    elif format == "yaml":
        import yaml

        with open(filename) as fin:
            return yaml.load(fin, Loader=yaml.Loader)
    elif format == "torch":
        import torch

        return torch.load(filename)
    elif format == "pickle":
        import pickle

        with open(filename, "rb") as fin:
            return pickle.load(fin)
    elif format == "npz":
        import numpy as np

        return np.load(filename, allow_pickle=True)
    else:
        raise NotImplementedError(
            f"Input format not supported:" f" try from {supported_formats.keys()}"
        )


def adjust_format_name(
    supported_formats: dict, filename: str, enforced_format: str = None
):
    """
    Recognize whether proper suffix is added to the filename.
    If not, add it and return the formatted file name

    Args:

        supported_formats (dict): list of supported formats and corresponding suffix
        filename (str): initial filename
        enforced_format (str): default format

    Returns:

        newformat (str): the chosen format
        newname (str): the adjusted filename

    """
    if enforced_format is None:
        newformat = match_suffix(supported_formats=supported_formats, filename=filename)
    else:
        newformat = enforced_format

    newname = f"{filename}"

    add_suffix = True
    suffix = supported_formats[newformat]

    if not isinstance(suffix, (set, list, tuple)):
        suffix = [suffix]

    if len(suffix) > 0:
        for suf in suffix:
            if filename.endswith(f".{suf}"):
                add_suffix = False

        if add_suffix:
            suffix = suffix[0]
            newname += f".{suffix}"

    return newformat, newname


def match_suffix(supported_formats: str, filename: str):
    """
    Recognize format based on suffix

    Args:

        supported_formats (dict): list of supported formats and corresponding suffix
        filename (str): initial filename

    Returns:

        format (str): the recognized format

    """
    for form, suffs in supported_formats.items():
        if isinstance(suffs, (set, list, tuple)):
            for suff in suffs:
                if filename.lower().endswith(f".{suff}"):
                    return form
        else:
            if filename.lower().endswith(f".{suffs}"):
                return form

    return list(supported_formats.keys())[0]

def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.isfile(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_path, state):
    saved_state = {
      'optimizer': state['optimizer'].state_dict(),
      'model': state['model'].state_dict(),
      'ema': state['ema'].state_dict(),
      'step': state['step']
    }
    torch.save(saved_state, ckpt_path)