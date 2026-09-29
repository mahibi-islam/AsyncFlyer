import copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .crypto import UpdateCipher, serialize_state, deserialize_state

class FederatedTrainer:
    def __init__(self, model, lr=.001, batch_size=32, local_epochs=5, device='cpu', seed=42):
        self.global_model = model.to(device); self.lr=lr; self.batch_size=batch_size; self.local_epochs=local_epochs; self.device=device; self.seed=seed
        self.cipher = UpdateCipher(); self.audit=[]
    def _train_one(self, state, X, y):
        model = copy.deepcopy(self.global_model).to(self.device); model.load_state_dict(state)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr); loss_fn=nn.CrossEntropyLoss()
        ds=TensorDataset(torch.tensor(X), torch.tensor(y)); loader=DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        model.train(); total=0.0
        for _ in range(self.local_epochs):
            for xb,yb in loader:
                xb,yb=xb.to(self.device),yb.to(self.device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total += float(loss.item())
        return model.state_dict(), len(ds), total/max(1,len(loader)*self.local_epochs)
    def _encrypt_state(self, state):
        start=time.perf_counter(); item=self.cipher.encrypt(serialize_state(state)); enc_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter(); plain=self.cipher.decrypt(item); dec_ms=(time.perf_counter()-start)*1000
        self.audit.append({'plaintext_bytes':len(plain),'ciphertext_bytes':len(item.ciphertext),'encrypt_ms':enc_ms,'decrypt_ms':dec_ms})
        return deserialize_state(plain, state)
    def run(self, clients, rounds=10, client_fraction=1.0):
        rng=np.random.default_rng(self.seed); history=[]
        for r in range(1, rounds+1):
            n=max(1, int(np.ceil(len(clients)*client_fraction))); selected=rng.choice(len(clients),size=n,replace=False)
            base=copy.deepcopy(self.global_model.state_dict()); updates=[]; total=0; losses=[]; t0=time.perf_counter()
            for i in selected:
                state,count,loss=self._train_one(base,*clients[i]); state=self._encrypt_state(state); updates.append((state,count)); total += count; losses.append(loss)
            avg={k:sum(state[k]*(count/total) for state,count in updates) for k in base}
            self.global_model.load_state_dict(avg); history.append({'round':r,'clients':selected.tolist(),'mean_local_loss':float(np.mean(losses)),'round_seconds':time.perf_counter()-t0})
        return history
    @torch.no_grad()
    def evaluate(self, data):
        X,y=data; self.global_model.eval(); pred=torch.argmax(self.global_model(torch.tensor(X).to(self.device)),dim=1).cpu().numpy()
        p,r,f,_=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
        return {'accuracy':float(accuracy_score(y,pred)),'precision':float(p),'recall':float(r),'f1':float(f),'confusion_matrix':confusion_matrix(y,pred).tolist()}
    def save(self, path): torch.save(self.global_model.state_dict(),path)
