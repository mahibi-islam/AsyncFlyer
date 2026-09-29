import base64, json, os
from dataclasses import dataclass
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

@dataclass
class EncryptedUpdate:
    nonce: bytes
    ciphertext: bytes
    def as_dict(self): return {'nonce': base64.b64encode(self.nonce).decode(), 'ciphertext': base64.b64encode(self.ciphertext).decode()}

class UpdateCipher:
    def __init__(self, key=None): self.key = key or AESGCM.generate_key(bit_length=256)
    def encrypt(self, payload: bytes) -> EncryptedUpdate:
        nonce = os.urandom(12); return EncryptedUpdate(nonce, AESGCM(self.key).encrypt(nonce, payload, None))
    def decrypt(self, item: EncryptedUpdate) -> bytes: return AESGCM(self.key).decrypt(item.nonce, item.ciphertext, None)

class LocalStorageCipher:
    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
    def encrypt(self, data: bytes):
        aes_key = AESGCM.generate_key(bit_length=256); nonce = os.urandom(12)
        wrapped = self.public_key.encrypt(aes_key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
        return {'wrapped_key': wrapped, 'nonce': nonce, 'ciphertext': AESGCM(aes_key).encrypt(nonce, data, None)}
    def decrypt(self, item):
        key = self.private_key.decrypt(item['wrapped_key'], padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
        return AESGCM(key).decrypt(item['nonce'], item['ciphertext'], None)

def serialize_state(state):
    arrays = {k: v.detach().cpu().numpy().astype(np.float32).tolist() for k,v in state.items()}
    return json.dumps(arrays, separators=(',', ':')).encode()

def deserialize_state(payload, reference):
    raw = json.loads(payload.decode()); return {k: __import__('torch').tensor(raw[k], dtype=v.dtype) for k,v in reference.items()}
