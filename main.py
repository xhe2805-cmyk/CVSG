"""CVSG: training and evaluation with validation-based checkpoint selection."""
import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertConfig, BertTokenizer
import global_configs

ROOT = Path(__file__).resolve().parent
METHOD = 'CVSG'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    for key, default in {'max_seq_length': 50, 'train_batch_size': 32, 'dev_batch_size': 128,
                         'test_batch_size': 128, 'n_epochs': 48, 'gradient_accumulation_step': 1,
                         'seed': 0}.items():
        flags = list(dict.fromkeys(['--' + key, '--' + key.replace('_', '-')]))
        p.add_argument(*flags, dest=key, type=int, default=default)
    p.add_argument('--learning_rate', '--learning-rate', type=float, default=1e-5)
    p.add_argument('--beta_shift', type=float, default=1.)
    p.add_argument('--dropout_prob', type=float, default=.5)
    p.add_argument('--warmup_proportion', type=float, default=.1,
                   help='Accepted for CLI compatibility; unused by the optimizer')
    p.add_argument('--model', default='bert-base-uncased')
    p.add_argument('--bert_path', '--bert-path', default='')
    p.add_argument('--data_root', '--data-root', default='datasets')
    p.add_argument('--device', default='auto')
    p.add_argument('--gpu', type=int, default=None, help='Optional alias: GPU index; -1 for CPU')
    p.add_argument('--save_dir', '--save-dir', default='checkpoints')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--eval_only', '--eval-only', action='store_true')
    p.add_argument('--include_zero', '--include-zero', action='store_true')
    p.add_argument('--disable_coalition', action='store_true',
                   help='Ablation: no train-only coalition views or their utility losses')
    p.add_argument('--disable_value', action='store_true',
                   help='Ablation: uniform modality routing instead of learned marginal utility')
    p.add_argument('--disable_synergy', action='store_true',
                   help='Ablation: omit pair-coalition corrections and synergy supervision')
    p.add_argument('--disable_uncertainty', action='store_true',
                   help='Ablation: remove posterior-scale reliability descriptors')
    p.add_argument('--disable_synergy_supervision', action='store_true',
                   help='Ablation: train pair corrections without target synergy supervision')
    p.add_argument('--kl_weight', type=float, default=.001)
    p.add_argument('--coalition_weight', type=float, default=.08)
    p.add_argument('--value_weight', type=float, default=.04)
    p.add_argument('--synergy_weight', type=float, default=.02)
    p.add_argument('--utility_temperature', type=float, default=1.5)
    p.add_argument('--pair_strength', type=float, default=.65)
    p.add_argument('--select_metric', choices=['mse','mae'], default='mse',
                   help='Best checkpoint selected on VAL only. For MAE emphasis choose mae before training.')
    p.add_argument('--no_clip_eval', action='store_true',help='Disable [-3,3] prediction clipping in evaluation')
    p.add_argument('--stochastic_eval', action='store_true', help='Use posterior samples at evaluation')
    p.add_argument('--no_epoch_test', action='store_true', default=True,
                   help='Default: no TEST evaluation until training finishes')
    p.add_argument('--epoch_test', dest='no_epoch_test', action='store_false',
                   help='Diagnostic only: display TEST each epoch; never select by test')

    args = p.parse_args(argv)
    for key in ('max_seq_length', 'train_batch_size', 'dev_batch_size', 'test_batch_size',
                'n_epochs', 'gradient_accumulation_step'):
        if getattr(args, key) < (3 if key == 'max_seq_length' else 1):
            p.error(key + ' is too small')
    if min(args.kl_weight,args.coalition_weight,args.value_weight,args.synergy_weight,args.pair_strength) < 0:
        p.error('Variant coefficients must be nonnegative')
    if args.utility_temperature <= 0:
        p.error('utility_temperature must be positive')
    if not 0 <= args.seed < 2**32 or args.learning_rate <= 0:
        p.error('Invalid seed or learning rate')
    return args


def root_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def choose_device(args):
    name = ('cpu' if args.gpu < 0 else f'cuda:{args.gpu}') if args.gpu is not None else args.device
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu')
        if (device.index or 0) >= torch.cuda.device_count():
            raise ValueError('Requested GPU index exceeds visible devices')
        torch.cuda.set_device(device)
    return device


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def evaluation_rng(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


def read_data(path):
    if not path.is_file():
        raise FileNotFoundError('Dataset not found: ' + str(path))
    with path.open('rb') as handle:
        data = pickle.load(handle, encoding='latin1')
    if not isinstance(data, dict):
        raise ValueError('Expected word-aligned pickle with train/dev/test splits')
    return data


def get_split(data, name):
    key = 'valid' if name == 'dev' and 'dev' not in data else name
    if key not in data or not len(data[key]):
        raise ValueError('Missing or empty split: ' + name)
    if isinstance(data[key], dict):
        raise ValueError('Use word-aligned feature tuples; pre-encoded dictionary records are not supported')
    return data[key]


def make_dataset(examples, tokenizer, length, expected_dims=None):
    ids_all, visuals, acoustics, masks, types, labels = [], [], [], [], [], []
    dims = expected_dims
    for index, example in enumerate(examples):
        try:
            (words, visual, acoustic), label, _ = example
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Sample {index}: expected ((words, vision, audio), label, id)') from exc
        if isinstance(words, (str, bytes)):
            raise ValueError('Expected a word list, not a sentence string')
        visual, acoustic = np.asarray(visual, dtype=np.float32), np.asarray(acoustic, dtype=np.float32)
        if visual.ndim != 2 or acoustic.ndim != 2 or not (len(words) == len(visual) == len(acoustic)):
            raise ValueError(f'Sample {index}: require raw word-aligned feature rows. HME BERT-padded tuples are a different format')
        current_dims = (acoustic.shape[1], visual.shape[1])
        if min(current_dims) < 1:
            raise ValueError('Empty feature width')
        if dims is None:
            dims = current_dims
        if tuple(dims) != current_dims:
            raise ValueError(f'Feature dimensions {current_dims} do not match {dims}')
        visual = np.nan_to_num(visual, nan=0., posinf=0., neginf=0.)
        acoustic = np.nan_to_num(acoustic, nan=0., posinf=0., neginf=0.)
        tokens, inversions = [], []
        for word_index, word in enumerate(words):
            word = word.decode('utf-8') if isinstance(word, bytes) else str(word)
            pieces = tokenizer.tokenize(word) or [tokenizer.unk_token]
            tokens.extend(pieces)
            inversions.extend([word_index] * len(pieces))
        tokens, inversions = tokens[:length - 2], inversions[:length - 2]
        ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        ids_all.append(ids + [tokenizer.pad_token_id] * (length - len(ids)))
        masks.append([1] * len(ids) + [0] * (length - len(ids)))
        types.append([0] * length)
        v, a = np.zeros((length, dims[1]), np.float32), np.zeros((length, dims[0]), np.float32)
        v[1:1 + len(inversions)], a[1:1 + len(inversions)] = visual[inversions], acoustic[inversions]
        visuals.append(v)
        acoustics.append(a)
        value = np.asarray(label, dtype=np.float32).reshape(-1)
        if len(value) != 1 or not np.isfinite(value[0]):
            raise ValueError('Each regression target must be one finite scalar')
        labels.append(value[0])
    return TensorDataset(torch.tensor(ids_all, dtype=torch.long), torch.from_numpy(np.stack(visuals)),
                         torch.from_numpy(np.stack(acoustics)), torch.tensor(masks, dtype=torch.long),
                         torch.tensor(types, dtype=torch.long), torch.from_numpy(np.asarray(labels, dtype=np.float32))), tuple(dims)


def loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def metrics(pred, truth, include_zero=False):
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if not len(pred) or pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Predictions and targets must be finite, matching, nonempty vectors')
    selected = np.ones(len(truth), dtype=bool) if include_zero else truth != 0
    result = {'Acc_2': None, 'F1': None}
    if selected.any():
        result['Acc_2'] = float(accuracy_score(truth[selected] >= 0, pred[selected] >= 0))
        result['F1'] = float(f1_score(truth[selected] >= 0, pred[selected] >= 0, average='weighted', zero_division=0))
    result['Acc_7'] = float(np.mean(np.round(np.clip(pred, -3, 3)) == np.round(np.clip(truth, -3, 3))))
    result['MAE'] = float(np.mean(np.abs(pred - truth)))
    result['Corr'] = (float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 and pred.std() > 1e-12 and truth.std() > 1e-12 else None)
    result['MSE'] = float(np.mean((pred - truth)**2))
    return result


def format_metrics(result):
    return '  '.join(f'{name}=' + ('NA' if value is None else f'{value:.4f}') for name, value in result.items())


def format_epoch_line(epoch, split, result):
    """Fixed-width epoch rows: TRAIN / VAL / TEST / BEST VAL TEST."""
    result = result or {}
    fields = []
    for key in ('Acc_2', 'F1', 'Acc_7', 'MAE', 'Corr', 'MSE'):
        value = result.get(key)
        display = 'NA' if value is None else f'{value:.4f}'
        fields.append(f'{key}={display:>7}')
    objective = result.get('Objective')
    objective_display = '--' if objective is None else f'{objective:.4f}'
    fields.append(f'Objective={objective_display:>8}')
    return f'[CVSG][Epoch {epoch}][{split:<13}] ' + '  '.join(fields)


def train_epoch(model, data_loader, optimizers, args, device):
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    predictions, targets, total_loss, seen = [], [], 0., 0
    for step, batch in enumerate(tqdm(data_loader, desc='TRAIN', disable=not __import__('sys').stderr.isatty())):
        ids, v, a, mask, segments, y = [item.to(device) for item in batch]
        prediction, loss = model(ids, v, a, y, attention_mask=mask, token_type_ids=segments, return_loss=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at batch {step}')
        start = step // args.gradient_accumulation_step * args.gradient_accumulation_step
        end = min(start + args.gradient_accumulation_step, len(data_loader))
        group_samples = min(end * data_loader.batch_size, len(data_loader.dataset)) - start * data_loader.batch_size
        (loss * len(y) / group_samples).backward()
        if step + 1 == end:
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * len(y)
        seen += len(y)
        predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
        targets.extend(y.detach().cpu().reshape(-1).tolist())
    result = metrics(predictions, targets, args.include_zero)
    result['Objective'] = total_loss / seen
    return result


def evaluate(model, data_loader, args, device):
    model.eval()
    predictions, targets = [], []
    with evaluation_rng((args.seed + 9876) % 2**32, device), torch.no_grad():
        for batch in data_loader:
            ids, v, a, mask, segments, y = [item.to(device) for item in batch]
            prediction = model.test(ids, v, a, attention_mask=mask, token_type_ids=segments)
            predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
            targets.extend(y.detach().cpu().reshape(-1).tolist())
    return metrics(predictions, targets, args.include_zero)


def tokenizer_files(tokenizer):
    with tempfile.TemporaryDirectory(prefix='cvsg-tokenizer-') as directory:
        tokenizer.save_pretrained(directory)
        return {p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()}


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not found: ' + str(path))
    payload = torch.load(path, map_location='cpu')
    if payload.get('method') != METHOD:
        raise ValueError('CVSG checkpoint required; incompatible model configurations cannot be loaded')
    return payload


def configure_model(dims):
    global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM = dims
    module = importlib.import_module('model')
    if getattr(module, 'MODEL_ID', None) != METHOD:
        raise ValueError('main.py and model.py have different model identifiers')
    module.ACOUSTIC_DIM, module.VISUAL_DIM = dims
    return module.CVSGModel


def set_training_options(model, settings):
    model.fusion.configure_training_options(
        disable_coalition=getattr(settings,'disable_coalition',False),
        disable_value=getattr(settings,'disable_value',False),
        disable_synergy=getattr(settings,'disable_synergy',False),
        disable_uncertainty=getattr(settings,'disable_uncertainty',False),
        disable_synergy_supervision=getattr(settings,'disable_synergy_supervision',False),
        kl_weight=getattr(settings,'kl_weight',.001),
        coalition_weight=getattr(settings,'coalition_weight',.08),
        value_weight=getattr(settings,'value_weight',.04),
        synergy_weight=getattr(settings,'synergy_weight',.02),
        utility_temperature=getattr(settings,'utility_temperature',1.5),
        pair_strength=getattr(settings,'pair_strength',.65),
        deterministic_eval=not getattr(settings,'stochastic_eval',False),
        clip_eval=not getattr(settings,'no_clip_eval',False),
    )


def restore(payload, device):
    CVSGModel = configure_model(tuple(payload['feature_dims']))
    old_config = payload['experiment_config']
    multi = SimpleNamespace(beta_shift=old_config['beta_shift'], dropout_prob=old_config['dropout_prob'])
    model = CVSGModel(BertConfig.from_dict(payload['bert_config']), multi)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    saved_settings = payload['experiment_config']
    set_training_options(model, SimpleNamespace(**payload['experiment_config']))
    return model.to(device)


def restore_tokenizer(payload):
    with tempfile.TemporaryDirectory(prefix='cvsg-test-tokenizer-') as directory:
        for name, content in payload['tokenizer_files'].items():
            if Path(name).name != name:
                raise ValueError('Invalid embedded tokenizer file name')
            (Path(directory) / name).write_bytes(content)
        return BertTokenizer.from_pretrained(directory, local_files_only=True)


def fit(model, train_loader, valid_loader, args, tokenizer, dims, device, checkpoint_path, test_loader):
    bert_parameters = list(model.bert.parameters())
    bert_ids = {id(p) for p in bert_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in bert_ids]
    optimizers = (torch.optim.Adam(bert_parameters, lr=args.learning_rate),
                  torch.optim.Adam(other_parameters, lr=2 * args.learning_rate))
    files = tokenizer_files(tokenizer)
    best = float('inf')
    best_epoch = None
    best_test_result = None
    for epoch in range(1, args.n_epochs + 1):
        train_result = train_epoch(model, train_loader, optimizers, args, device)
        valid_result = evaluate(model, valid_loader, args, device)
                                                                              
        test_result = None if args.no_epoch_test else evaluate(model, test_loader, args, device)
        metric_key=args.select_metric.upper()
        new_best = valid_result[metric_key] < best
        if new_best:
                                                                                       
            best = valid_result[metric_key]
            best_epoch = epoch
            best_test_result = dict(test_result) if test_result is not None else None
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {'method': METHOD, 'dataset': args.dataset, 'seed': args.seed,
                       'epoch': epoch, 'best_validation_mse': valid_result['MSE'], 'best_validation_mae': valid_result['MAE'],
                       'best_validation_score': best,
                       'architecture_ablation': bool(args.disable_coalition or args.disable_value or args.disable_synergy or args.disable_uncertainty or args.disable_synergy_supervision),
                       'best_validation_f1': valid_result['F1'], 'selection_metric': 'VAL_'+metric_key, 'feature_dims': list(dims),
                       'best_validation_test_metrics': best_test_result,
                       'bert_config': model.config.to_dict(), 'experiment_config': vars(args).copy(),
                       'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       'tokenizer_files': files}
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
        lines = [format_epoch_line(epoch, 'TRAIN', train_result),
                 format_epoch_line(epoch, 'VAL', valid_result),
                 format_epoch_line(epoch, 'TEST', test_result),
                 format_epoch_line(epoch, 'BEST VAL TEST', best_test_result)
                 + (f'  BestEpoch={best_epoch}' if best_epoch is not None else '')]
        print('\n'.join(lines), flush=True)
        if new_best:
            print(f'New BEST VAL {metric_key}={best:.6f} -> {checkpoint_path}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args)
    run_tag = 'CVSG' + ''.join('_'+tag for flag,tag in (
        (args.disable_coalition, 'NOCOAL'),
        (args.disable_value, 'NOVALUE'),
        (args.disable_synergy, 'NOSYN'),
        (args.disable_uncertainty, 'NOUNC'),
        (args.disable_synergy_supervision, 'NOSYNSUP'),
        (args.select_metric=='mae','MAE'),
        (args.no_clip_eval,'NOCLIP'),
        (args.stochastic_eval, 'STOCH')) if flag)
    checkpoint_path = root_path(args.checkpoint or f'{args.save_dir}/{run_tag}_{args.dataset}_seed{args.seed}_best.pth')
    data_path = root_path(args.data_root) / (args.dataset + '.pkl')
    data = read_data(data_path)
    if args.eval_only:
        payload = load_checkpoint(checkpoint_path)
        if any(bool(payload['experiment_config'].get(k, False)) != bool(getattr(args, k)) for k in ('disable_coalition','disable_value','disable_synergy','disable_uncertainty','disable_synergy_supervision','stochastic_eval','no_clip_eval')):
            raise ValueError('Checkpoint ablation mode conflicts with CLI setting')
        if payload['dataset'] != args.dataset:
            raise ValueError('Checkpoint dataset does not match --dataset')
        saved = SimpleNamespace(**payload['experiment_config'])
        setup_seed(saved.seed)
        tokenizer = restore_tokenizer(payload)
        test_data, _ = make_dataset(get_split(data, 'test'), tokenizer, saved.max_seq_length, tuple(payload['feature_dims']))
        model = restore(payload, device)
    else:
        setup_seed(args.seed)
        source = args.bert_path or (str(ROOT / 'bert-base-uncased') if (ROOT / 'bert-base-uncased' / 'config.json').is_file() else args.model)
        if args.bert_path and root_path(args.bert_path).is_dir():
            source = str(root_path(args.bert_path))
        tokenizer = BertTokenizer.from_pretrained(source)
        config = BertConfig.from_pretrained(source)
        if args.max_seq_length > config.max_position_embeddings:
            raise ValueError('max_seq_length exceeds BERT position capacity')
        train_data, dims = make_dataset(get_split(data, 'train'), tokenizer, args.max_seq_length)
        valid_data, _ = make_dataset(get_split(data, 'dev'), tokenizer, args.max_seq_length, dims)
        epoch_test_data, _ = make_dataset(get_split(data, 'test'), tokenizer, args.max_seq_length, dims)
        CVSGModel = configure_model(dims)
        multimodal_config = SimpleNamespace(beta_shift=args.beta_shift, dropout_prob=args.dropout_prob)
                                                                                   
        model = CVSGModel.from_pretrained(source, config=config, multimodal_config=multimodal_config, _fast_init=False).to(device)
        set_training_options(model, args)
        print(f'CVSG | {args.dataset} | seed={args.seed} | device={device} | audio/vision={dims}', flush=True)
        print(f'BERT: {source}\nDataset: {data_path}\nBest checkpoint: {checkpoint_path}', flush=True)
        print(f'Selection: minimum VAL {args.select_metric.upper()}; {args.n_epochs} epochs; no added early stopping; learning rates={args.learning_rate:g}/{2*args.learning_rate:g}', flush=True)
        fit(model, loader(train_data, args.train_batch_size, True), loader(valid_data, args.dev_batch_size),
            args, tokenizer, dims, device, checkpoint_path, loader(epoch_test_data, args.test_batch_size))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        payload = load_checkpoint(checkpoint_path)
        saved = SimpleNamespace(**payload['experiment_config'])
        model = restore(payload, device)
        test_data, _ = make_dataset(get_split(data, 'test'), tokenizer, saved.max_seq_length, tuple(dims))
    result = evaluate(model, loader(test_data, saved.test_batch_size), saved, device)
    print(f'Loaded BEST epoch={payload["epoch"]} {payload["selection_metric"]}={payload["best_validation_score"]:.6f}', flush=True)
    print('[CVSG][TEST] ' + format_metrics(result), flush=True)
    print('Final TEST: ' + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
