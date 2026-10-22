import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import getpass
import pandas as pd

def encrypt_password():
    # Step 1: Generate a true 256-bit (32-byte) key
    key = AESGCM.generate_key(bit_length=256)  # 256-bit key for AES-GCM
    password_name = input('Creating encrypted password for: ')
    plaintext_pw = getpass.getpass(f"Enter {password_name} Password: ").encode()

    # Step 2: Generate a 96-bit (12-byte) nonce for GCM
    nonce = os.urandom(12)

    # Step 3: Create AES-GCM cipher
    aesgcm = AESGCM(key)

    # Step 4: Encrypt and authenticate the data
    # No padding needed with GCM mode
    ciphertext = aesgcm.encrypt(nonce, plaintext_pw, None)

    # Step 5: Combine nonce and ciphertext for storage
    encrypted_data = nonce + ciphertext

    # Step 6: Base64 encode for safe storage
    encrypted_password = base64.urlsafe_b64encode(encrypted_data)

    # create directory if not exist
    os.makedirs("FileReaderv2/PasswordKey/", exist_ok=True)
    # Save the encrypted password
    encrypted_password_path = f"FileReaderv2/PasswordKey/encrypted_{password_name}_password.parquet"
    df = pd.DataFrame({"Encrypted_Password": [encrypted_password]})
    df.to_parquet(encrypted_password_path)

    # Save the key (in a real application, protect this carefully!)
    key_file_path = f"FileReaderv2/PasswordKey/encrypted_{password_name}_password_key.key"
    with open(key_file_path, "wb") as key_file:
        key_file.write(key)

    print(f"Password encrypted to {encrypted_password_path}")
    print(f"256-bit key saved to {key_file_path}")

if __name__ == "__main__":
    encrypt_password()
