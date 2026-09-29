import numpy as np, torch
from flyer.model import FLyerLSTM
from flyer.crypto import UpdateCipher, LocalStorageCipher

def test_model_shape():
    m=FLyerLSTM(6,4); out=m(torch.zeros(3,1,6)); assert out.shape==(3,4)

def test_crypto_roundtrip():
    c=UpdateCipher(); msg=b'fl-update'; item=c.encrypt(msg); assert c.decrypt(item)==msg
    s=LocalStorageCipher(); item=s.encrypt(b'local-data'); assert s.decrypt(item)==b'local-data'

def test_single_fed_round():
    from flyer.federated import FederatedTrainer
    clients=[(np.random.randn(8,1,3).astype('float32'),np.random.randint(0,2,8)) for _ in range(2)]
    t=FederatedTrainer(FLyerLSTM(3,2),local_epochs=1,batch_size=4)
    h=t.run(clients,rounds=1); assert len(h)==1 and len(t.audit)==2
