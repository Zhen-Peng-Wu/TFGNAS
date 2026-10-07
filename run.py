#!/usr/bin/env python3
"""Train and evaluate the final TFGNAS model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from data import GeneData, get_data_file_path, prepare_fold_data
from model import TFGNASModel
from train import structural_features, tf_node_ids, _architecture_parameters, optimization_step, evaluate


def set_seed():
    np.random.seed(114514)
    torch.manual_seed(114514)
    torch.cuda.manual_seed_all(114514)
    torch.cuda.manual_seed(114514)
    random.seed(114514)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = '114514'


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-name', required=True, choices=['bonemarrow', 'mESC_1', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L'])
    parser.add_argument('--data-dir', type=Path, default=project/'data_evaluation')
    parser.add_argument('--output-dir', type=Path, default=project/'results'/'TFGNAS')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeat-id', type=int, default=1)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error('Dataset directory not found: %s. Extract data_evaluation in the code directory or pass --data-dir.' % args.data_dir)
    args.device = torch.device('cuda:%d' % args.device)
    args.store_path = str(args.data_dir)
    set_seed()
    args = get_data_file_path(args)
    data = GeneData(args.rpkm_path, args.label_path, args.divide_path, TF_num=args.TF_num,
                    gene_emb_path=args.gene_emb_path, cell_emb_path=args.cell_emb_path,
                    istime=args.is_time, gene_list_path=args.gene_list_path,
                    data_name=args.data_name, TF_random=args.TF_random, ish5=args.is_h5)
    data = prepare_fold_data(data, args)
    # Preserve the data loader's internal seed 42; do not reseed after data.
    structure = structural_features(data.graph, data.bipartite_graph).to(args.device)
    assert structure.shape[1] == 9
    tf_ids = tf_node_ids(data)
    folds = []
    for fold in range(3):
        model = TFGNASModel(data.gene_emb.shape[1], data.gene_emb.shape[0], tf_ids, structure).to(args.device)
        arch = list(_architecture_parameters(model))
        arch_ids = {id(p) for p in arch}
        weights = [p for p in model.parameters() if id(p) not in arch_ids]
        optimizer = torch.optim.Adam([{'params': weights, 'lr': 1e-3}, {'params': arch, 'lr': 3e-4}], weight_decay=1e-6)
        samples = {s: getattr(data, s+'_samples_all')[fold].to(args.device) for s in ('train', 'val', 'test')}
        labels = {s: getattr(data, s+'_labels_all')[fold] for s in samples}
        best, wait, best_epoch, checkpoint = -float('inf'), 0, -1, None
        for epoch in range(200):
            optimization_step(model, optimizer, data.graph, data.bipartite_graph, samples['train'], labels['train'])
            val = evaluate(model, data.graph, data.bipartite_graph, samples['val'], labels['val'])
            if val[2] > best:
                best, wait, best_epoch = val[2], 0, epoch
                checkpoint = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if epoch % 10 == 0:
                print('PROGRESS', args.data_name, fold+1, epoch, val[2], flush=True)
            if wait > 10:
                break
        model.load_state_dict(checkpoint)
        # Preserve the established evaluation sequence between folds.
        evaluate(model, data.graph, data.bipartite_graph, samples['train'], labels['train'])
        record = {'fold': fold+1, 'best_epoch': best_epoch, 'stop_epoch': epoch}
        for split in ('val', 'test'):
            metrics = evaluate(model, data.graph, data.bipartite_graph, samples[split], labels[split])
            record[split] = {'auroc': metrics[1], 'auprc': metrics[2], 'loss': metrics[3]}
        folds.append(record)
        print('FOLD_RESULT', json.dumps(record), flush=True)
    manifest = []
    for fold in range(3):
        record = {'fold': fold+1}
        for split in ('train', 'val', 'test'):
            samples = getattr(data, split+'_samples_all')[fold]
            labels = getattr(data, split+'_labels_all')[fold]
            record[split] = {'samples': len(samples), 'samples_sha256': tensor_hash(samples), 'labels_sha256': tensor_hash(labels)}
        manifest.append(record)
    result = {'method': 'TFGNAS', 'structure_dimension': 9, 'dataset': args.data_name,
              'repeat_id': args.repeat_id, 'seed': 114514, 'seed_protocol': 'tfgnas_fixed',
              'selection_protocol': 'auprc_best', 'evaluation': 'three_fold_arithmetic_mean',
              'folds': folds, 'split_manifest': manifest}
    for split in ('val', 'test'):
        result[split+'_mean'] = {k: float(np.mean([f[split][k] for f in folds])) for k in ('auroc', 'auprc')}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/(args.data_name+'.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)
    final_result = {
        'dataset': args.data_name,
        'test_auroc': result['test_mean']['auroc'],
        'test_auprc': result['test_mean']['auprc'],
    }
    print('FINAL_RESULT', json.dumps(final_result), flush=True)


if __name__ == '__main__':
    main()
