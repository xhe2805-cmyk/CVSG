"""Train separate seeds and summarize held-out test metrics without cherry-picking."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mosi','mosei'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--bert_path', default=None)
    parser.add_argument('--data_root', default='datasets')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 54, 58, 101, 2334])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Use unique random seeds')
    output_dir = ROOT / 'results'
    output_dir.mkdir(exist_ok=True)
    runs=[]
    for seed in args.seeds:
        cmd = [sys.executable, str(ROOT / 'main.py'), '--dataset', args.dataset,
               '--gpu', str(args.gpu), '--n_epochs', str(args.n_epochs),
               '--seed', str(seed), '--data_root', args.data_root, '--no_epoch_test']
        if args.bert_path:
            cmd += ['--bert_path', args.bert_path]
        if args.train_batch_size is not None:
            cmd += ['--train_batch_size', str(args.train_batch_size)]
        log_path = output_dir / f'{args.dataset}_seed{seed}.log'
        print('Running:', ' '.join(cmd), flush=True)
        final = None
        with log_path.open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                if line.startswith('Final TEST: '):
                    final=json.loads(line.partition('Final TEST: ')[2].strip())
            code=proc.wait()
        if code or final is None:
            raise RuntimeError(f'Seed {seed} did not finish; inspect {log_path}')
        runs.append({'seed':seed, **final})
    summary={}
    for metric in ('Acc_2','F1','Acc_7','MAE','Corr','MSE'):
        values=[r[metric] for r in runs if r.get(metric) is not None]
        summary[metric]={'mean':statistics.mean(values) if values else None,
                         'std':statistics.stdev(values) if len(values)>1 else None}
    report={'dataset':args.dataset, 'per_seed':runs, 'summary':summary}
    path=output_dir / f'{args.dataset}_five_seed_summary.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved:',path)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
