from torchtext import data
import re

class UDDataset(data.Dataset):
    """Defines a dataset for conllu format. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", skip_begin_keyword='#', **kwargs):
        """
        :param path:
        :param fields:
        :param encoding:
        :param separator:
        :param skip_begin_keyword: skip lines which are start with the given keyword. Default: '#'
        :param kwargs:
        """
        examples = []
        columns = []

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                elif line.startswith('#'):
                    continue
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(UDDataset, self).__init__(examples, fields,
                                                     **kwargs)


class LCDataset(data.Dataset):
    """Defines a dataset for conllu format. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", skip_begin_keyword='#', tag=None, **kwargs):
        """
        :param path:
        :param fields:
        :param encoding:
        :param separator:
        :param skip_begin_keyword: skip lines which are start with the given keyword. Default: '#'
        :param kwargs:
        """
        examples = []
        columns = [[] for _ in range(3)] if 'UD' in str(path) else [[] for _ in range(2)]
        lang = re.split('/', str(path))[-2] if tag is None else tag

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                if 'UD' in str(path):
                    match = re.findall('^[0-9]+\t', line)
                line = line.strip()
                if line == "":
                    if columns:
                        columns[-1] = lang
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = [[] for _ in range(3)] if 'UD' in str(path) else [[] for _ in range(2)]
                elif 'UD' in str(path) and len(match) == 0:
                    continue
                elif line.startswith('#'):
                    continue
                else:
                    cols = line.split(separator)
                    # if '.' in cols[0]: continue
                    if 'UD' in str(path):
                        if len(cols) < 2:
                            continue
                        for i in range(2):
                            columns[i].append(cols[i])
                        # columns[-1].append(lang)
                    else:
                        if len(cols) < 1:
                            continue
                        columns[0].append(cols[0])
                        # columns[1].append(lang)

            if columns:
                columns[-1] = lang
                examples.append(data.Example.fromlist(columns, fields))
        super(LCDataset, self).__init__(examples, fields,
                                                     **kwargs)