import os
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64

def decrypt_password(password_name):
    """Decrypt a password encrypted with AES-GCM"""
    try:
        # 1. Load the 256-bit key
        key_file_path = f"FileReaderv2/PasswordKey/encrypted_{password_name}_password_key.key"
        with open(key_file_path, "rb") as key_file:
            key = key_file.read()
        
        # 2. Load the encrypted data
        encrypted_password_path = f"FileReaderv2/PasswordKey/encrypted_{password_name}_password.parquet"
        df = pd.read_parquet(encrypted_password_path)
        encrypted_data = base64.urlsafe_b64decode(df["Encrypted_Password"][0])
        
        # 3. Extract nonce (first 12 bytes) and ciphertext
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        # 4. Create AES-GCM cipher and decrypt
        aesgcm = AESGCM(key)
        password = aesgcm.decrypt(nonce, ciphertext, None)
        
        return password.decode()
    
    except Exception as e:
        raise ValueError(f"Decryption failed: {str(e)}")
