import argparse, json, random
from pathlib import Path
import numpy as np, torch, yaml
from .data import load_federated_data
from .model import FLyerLSTM
from .federated import FederatedTrainer

def main():
    ap=argparse.ArgumentParser(description='Reproduction of FLyer federated crop classifier')
    ap.add_argument('--data',required=True); ap.add_argument('--config',default='configs/default.yaml'); ap.add_argument('--out',default='runs/latest')
    ap.add_argument('--rounds',type=int); ap.add_argument('--epochs',type=int); ap.add_argument('--batch-size',type=int); ap.add_argument('--device',default='cpu')
    a=ap.parse_args(); cfg=yaml.safe_load(open(a.config)); cfg['rounds']=a.rounds or cfg['rounds']; cfg['local_epochs']=a.epochs or cfg['local_epochs']; cfg['batch_size']=a.batch_size or cfg['batch_size']
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    d=load_federated_data(a.data,seed,cfg['test_size'],cfg['cloud_holdout_size'],cfg['num_clients'])
    model=FLyerLSTM(len(d['feature_names']),len(d['classes']),cfg['lstm_units'],cfg['dense_units'])
    trainer=FederatedTrainer(model,cfg['learning_rate'],cfg['batch_size'],cfg['local_epochs'],a.device,seed)
    history=trainer.run(d['clients'],cfg['rounds'],cfg['client_fraction'])
    result={'config':cfg,'classes':d['classes'],'features':d['feature_names'],'global_test':trainer.evaluate(d['test']),'cloud_holdout':trainer.evaluate(d['cloud']),'history':history,'crypto_audit':trainer.audit}
    for i,c in enumerate(d['clients']):
        # Evaluate the final global model on each local partition, matching the paper's local-client reporting view.
        result[f'client_{i+1}']=trainer.evaluate(c)
    json.dump(result,open(out/'metrics.json','w'),indent=2); trainer.save(out/'global_model.pt')
    json.dump({'features':d['feature_names'],'classes':d['classes'],'scaler_mean':d['scaler'].mean_.tolist(),'scaler_scale':d['scaler'].scale_.tolist()},open(out/'preprocessing.json','w'),indent=2)
    print(json.dumps({'global_test':result['global_test'],'output':str(out)},indent=2))
if __name__=='__main__': main()
