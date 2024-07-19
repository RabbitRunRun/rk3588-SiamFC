# encoding: utf-8

import io
import os
import sys
import time
import inspect
import traceback
from typing import Any, Optional, Iterable, Union, List

__all__ = [
    'logdown',
    'default_argument_extensions',
]

default_argument_extensions = [
    '.py',
    '.txt',
    '.json',
    '.yaml',
    '.yml',
    '.toml',
    '.json5',
    '.xml',
    ".dino",
]


class DoubleWriter(io.TextIOBase):
    def __init__(self, a: io.TextIOBase, b: io.TextIOBase):
        self.__a = a
        self.__b = b

    def write(self, s: str):
        self.__a.write(s)
        self.__b.write(s)

    def flush(self):
        self.__a.flush()
        self.__b.flush()


class LogFile(io.TextIOBase):
    def __init__(self, stdout: Any, stderr: Any, logfile: str):
        self.__stdout = stdout
        self.__stderr = stderr
        self.__logfile = logfile

        self.__file = open(self.__logfile, "wb")
        self.__buf = ''

    @property
    def stdout(self) -> io.TextIOBase:
        return DoubleWriter(self.__stdout, self)

    @property
    def stderr(self) -> io.TextIOBase:
        return DoubleWriter(self.__stderr, self)

    @property
    def file(self):
        return self.__file

    def write(self, s: str):
        self.__buf += s

        while True:
            endl = self.__buf.find('\n')
            if endl >= 0:
                i = endl + 1
                s = self.__buf[:i]
                self.__buf = self.__buf[i:]
            else:
                break
            if s:
                rollback = s.rfind('\r')
                if rollback >= 1 and rollback == len(s) - 2:
                    rollback = s.rfind('\r', 0, rollback - 1)
                if rollback >= 0:
                    s = s[rollback + 1:]
                s = s.replace('\r', '')
                if s:
                    self.__file.write(s.encode('utf-8'))

    def flush(self) -> None:
        self.__file.flush()

    def close(self) -> None:
        if self.__buf:
            self.__file.write(self.__buf.encode('utf-8'))
            self.__buf = ''
        self.__file.close()

    def cleanup(self) -> None:
        s = self.__buf
        self.__buf = ''

        if s:
            rollback = s.rfind('\r')
            if rollback >= 0 and rollback == len(s) - 1:
                rollback = s.rfind('\r', 0, rollback - 1)
            if rollback >= 0:
                s = s[rollback + 1:]
            # treat last \r as \n to make inline log will be saved after exception
            s = s.replace('\r', '\n')
            if s:
                self.__file.write(s.encode('utf-8'))

    @property
    def closed(self) -> bool:
        return self.__file.closed


class LogDown(object):
    def __init__(self, logfile: str):
        self.__logfile = logfile

        self.__stdout: Any = sys.stdout
        self.__stderr: Any = sys.stderr

        self.__log: LogFile

        self.__header = ''
        self.__footer = ''

    @property
    def log(self):
        return self.__log

    @property
    def header(self):
        return self.__header

    @header.setter
    def header(self, v: str):
        # replace \r as \r is not welcome
        v = v.replace('\r', '')
        self.__header = v

    @property
    def footer(self):
        return self.__footer

    @footer.setter
    def footer(self, v: str):
        # replace \r as \r is not welcome
        v = v.replace('\r', '')
        self.__footer = v

    def __enter__(self):
        sys.__stdout__.write(f'Saving log into {self.__logfile}\n')

        self.__stdout: sys.stdout
        self.__stderr: sys.stderr

        log_root = os.path.split(self.__logfile)[0]
        if log_root:
            os.makedirs(log_root, exist_ok=True)
        self.__log = LogFile(sys.stdout, sys.stderr, self.__logfile)

        sys.stdout = self.__log.stdout
        sys.stderr = self.__log.stderr

        if self.__header:
            self.__log.file.write(self.__header.encode('utf-8'))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__log.cleanup()

        handled: Optional[bool] = None
        if exc_type is not None:
            handled = False
            tb = traceback.format_tb(exc_tb)
            e = str(exc_val)
            self.__log.write('Traceback (most recent call last):\n')
            for line in tb:
                self.__log.write(line)
            self.__log.write(f'{exc_type.__name__}: {e}\n')
            self.__log.cleanup()

        if self.__footer:
            self.__log.file.write(self.__footer.encode('utf-8'))

        self.__log.close()

        sys.stdout = self.__stdout
        sys.stderr = self.__stderr

        return handled


def try_find_exists_file(path: str, roots: List[str]) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isabs(path):
        return ''
    for root in roots:
        full = os.path.join(root, path)
        if os.path.isfile(full):
            return full
    return ''


def parse_argument_files(
        target_root: str,
        include_argument_extensions: Iterable[str] = None,
        exclude_argument_extensions: Iterable[str] = None,
        max_argument_file_size=10240,
        extern_argument_root: Union[str, Iterable[str]] = None) -> List[str]:
    # check extern_argument_root
    if extern_argument_root is None:
        extern_argument_root = []
    elif isinstance(extern_argument_root, str):
        extern_argument_root = [extern_argument_root]
    extern_argument_root = list(map(str, extern_argument_root))
    extern_argument_root.append(target_root)

    # get include arguments
    support = set(default_argument_extensions)
    if include_argument_extensions is not None:
        support = support.union(set(include_argument_extensions))
    if exclude_argument_extensions is not None:
        support = support.difference(set(exclude_argument_extensions))

    header = ["## Argument files", '']

    if not support:
        header.append("> No supported extensions have been set.")
        header.append('')
        return header

    if len(sys.argv) <= 1:
        header.append("> The program did not set any arguments.")
        header.append('')
        return header

    min_arg_len = len(min(support, key=len))
    max_arg_len = len(max(support, key=len))

    # find extension are supported arguments
    support_args: List[str] = []
    for arg in sys.argv[1:]:
        ext: str = os.path.splitext(arg)[-1]
        if not min_arg_len <= len(ext) <= max_arg_len:
            continue
        ext = ext.lower()
        if ext not in support:
            continue
        support_args.append(arg)

    # find if arguments exists
    notfound_args: List[str] = []
    exists_args: List[str] = []
    exists_filepath: List[str] = []
    for arg in support_args:
        path = arg
        equal = path.find('=')
        if equal > 0:
            path = arg[equal + 1:]
        path = try_find_exists_file(path, extern_argument_root)
        if path:
            exists_args.append(arg)
            exists_filepath.append(path)
        else:
            notfound_args.append(arg)

    if not support_args:
        header.append("> No arguments that could be files were found.")
        header.append('')
        return header

    if not exists_args:
        header.append("Following arguments that could be files:  ")
        header.extend([f"- {v}" for v in support_args])
        header.append('')
        header.append('But not found in disk. Tried in:  ')
        header.extend([f"- {v}" for v in extern_argument_root])
        header.append('')
        return header

    header.append("Found arguments which could be files:  ")
    header.extend([f"- {v}" for v in exists_args])
    header.append('')

    written = set()
    for arg, path in zip(exists_args, exists_filepath):
        if path in written:
            continue
        written.add(path)

        filename = os.path.split(path)[-1]
        ext = os.path.splitext(filename)[-1]
        if ext and ext[0] == '.':
            ext = ext[1:]
        ext = ext.lower()

        header.append(f"### Argument `{filename}`")
        header.append('')

        header.append(f'Usage: ```{arg}```  ')
        header.append('')
        header.append(f'> {path}  ')
        header.append('')

        content = ''
        error = ''
        has_limit = False

        try:
            with open(path, 'rb') as f:
                limit = None
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)
                if 0 < max_argument_file_size < file_size:
                    limit = max_argument_file_size
                    has_limit = True
                content = f.read(limit).decode('utf-8')
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"

        if error:
            header.append(f"Can not read argument file because:")
            header.append('```')
            header.append(error)
            header.append('```')
        else:
            if has_limit:
                header.append(
                    f"> File is larger than {max_argument_file_size}. The displayed content will be intercepted.")
                header.append('')

            s = content
            if not s or s[-1] != '\n':
                s += '\n'
            header.append(f"```{ext}")
            header.append(f"{s}```")

        header.append('')

    return header


def logdown(log: str = None,
            catch_arguments=False,
            include_argument_extensions: Iterable[str] = None,
            exclude_argument_extensions: Iterable[str] = None,
            max_argument_file_size=10240,
            extern_argument_root: Union[str, Iterable[str]] = None):
    """
    Notice: the argument write into markdown only support utf-8
    :param log: the output log filename, if not set, the log will write in the call `logdown` source file's folder.
    :param catch_arguments: if check and write each parameter file into log
    :param include_argument_extensions: the external argument file extensions you want check. like ['.c', '.cpp'].
    :param exclude_argument_extensions: the external argument file extensions you never want check. like ['.so'].
    :param max_argument_file_size: the argument file max size, if the file is too large.
    :param extern_argument_root: when check argument is file no not, will check if file exists in extern_argument_root
    :return:
    """
    datetime = time.localtime()

    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename
    caller_filepath = caller_frame.frame.f_globals['__file__']

    target = caller_filepath if caller_filepath else caller_filename
    target_content = ''
    target_error = ''

    if target:
        target_ext: str = os.path.splitext(target)[-1]
        target_ext = target_ext.lower()
        if target_ext == ".py" and os.path.isfile(target):
            try:
                with open(target, 'rb') as f:
                    target_content = f.read().decode('utf-8')
            except Exception as e:
                target_error = f"{type(e).__name__}: {str(e)}"
    else:
        target = "unknown"

    target_root, target_filename = os.path.split(target)

    if log is None:
        log_name = os.path.splitext(target_filename)[0]
        pid = os.getpid()
        log_datetime = time.strftime(f'%Y%m%d-%H%M%S-{pid}', datetime)
        log_filename = f'{log_name}-{log_datetime}.md'
        log = os.path.join(target_root, "logs", log_filename)

    ld = LogDown(log)
    header = []
    footer = []

    header.append(f"# Log Down")
    header.append('')
    header.append(f"> {target}")
    header.append('')
    header.append(f"Start time: `{time.strftime(f'%Y-%m-%d %H:%M:%S', datetime)}`  ")
    if len(sys.argv) > 1:
        header.append(f"Arguments:  ```{' '.join(sys.argv[1:])}```")
    else:
        header.append(f"Arguments:  ")
    header.append('')
    header.append('[TOC]')
    header.append('')

    if target_content:
        s = target_content
        if not s or s[-1] != '\n':
            s += '\n'

        header.append("## Source file")
        header.append('')
        header.append(f"> {target_filename}")
        header.append('')
        header.append(f"```python")
        header.append(f"{s}```")
        header.append('')

    if target_error:
        header.append("## Source file")
        header.append('')
        header.append(f"> {target_filename}")
        header.append('')
        header.append(f"Can not read source file because:")
        header.append('```')
        header.append(target_error)
        header.append('```')
        header.append('')

    if catch_arguments:
        argument_files = parse_argument_files(
            target_root,
            include_argument_extensions=include_argument_extensions,
            exclude_argument_extensions=exclude_argument_extensions,
            max_argument_file_size=max_argument_file_size,
            extern_argument_root=extern_argument_root)
        header.extend(argument_files)

    header.append("## Log")
    header.append('')
    header.append(f"```log")
    header.append('')

    footer.append('')
    footer.append(f"```")

    ld.header = '\n'.join(header)
    ld.footer = '\n'.join(footer)

    return ld
