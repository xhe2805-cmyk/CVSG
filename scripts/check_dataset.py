"""Validate required splits and a few sample shapes in a sentiment dataset."""
import argparse
from pathlib import Path
import pickle


def check(path):
    with Path(path).open('rb') as handle:
        data = pickle.load(handle, encoding='latin1')
    if not isinstance(data, dict):
        raise ValueError('Expected a dictionary of train, dev/valid, and test splits')
    for key in ('train', 'dev', 'test'):
        actual = 'valid' if key == 'dev' and 'dev' not in data else key
        if actual not in data or not len(data[actual]):
            raise ValueError(f'Missing or empty split: {key}')
        for sample in data[actual][:min(3, len(data[actual]))]:
            (words, vision, audio), label, sample_id = sample
            if len(words) != len(vision) or len(words) != len(audio):
                raise ValueError(f'Split {key}: word/vision/audio lengths differ')
        print(f'{key}: {len(data[actual])} examples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='Dataset pickle path')
    args = parser.parse_args()
    check(args.path)
