#!/usr/bin/env python
import os
import string
import random
import bz2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    # Paths to check for the training dump (e.g. Wikipedia dump.xml.bz2)
    TRAIN_DUMP_PATHS = [
        'data/dump.xml.bz2',
        'dump.xml.bz2',
        '/job/data/dump.xml.bz2',
    ]

    @classmethod
    def load_training_data(cls, max_texts=None):
        """
        Load training data from a bz2-compressed XML dump (e.g. Wikipedia).
        Looks for dump.xml.bz2 in data/, current dir, or /job/data (Docker).
        Returns a list of text strings (one per article/page).
        If max_texts is set, only the first that many texts are returned (for debugging).
        """
        dump_path = None
        for path in cls.TRAIN_DUMP_PATHS:
            if os.path.isfile(path):
                dump_path = path
                break
        if dump_path is None:
            return []

        texts = []
        with bz2.open(dump_path, 'rt', encoding='utf-8', errors='replace') as f:
            in_text = False
            current = []
            for line in f:
                if not in_text:
                    if '<text' in line:
                        in_text = True
                        start = line.find('>') + 1
                        if '</text>' in line[start:]:
                            end = line.find('</text>', start)
                            texts.append(line[start:end])
                            if max_texts is not None and len(texts) >= max_texts:
                                return texts
                            in_text = False
                        else:
                            current = [line[start:].rstrip('\n')]
                else:
                    if '</text>' in line:
                        end = line.find('</text>')
                        current.append(line[:end])
                        texts.append('\n'.join(current))
                        if max_texts is not None and len(texts) >= max_texts:
                            return texts
                        current = []
                        in_text = False
                    else:
                        current.append(line.rstrip('\n'))
        return texts

    @classmethod
    def load_test_data(cls, fname):
        """
        Load test data: one string per line (UTF-8). Each string is the context
        for which the model must predict the next character (3 guesses).
        """
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line.rstrip('\n')  # strip newline only, preserve rest
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
