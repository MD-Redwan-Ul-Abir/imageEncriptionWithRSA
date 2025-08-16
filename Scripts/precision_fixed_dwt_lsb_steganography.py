import numpy as np
import cv2
import pywt
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import logging
import datetime
import os
import sys


def setup_structured_directories():
    """Create structured directory layout for organized file management"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"
    
    # Define directory structure
    base_dirs = {
        'output': 'Output',
        'logs': 'Output/Logs',
        'keys': 'Output/Keys', 
        'stego_images': 'Output/Steganographic_Images',
        'session': f'Output/Sessions/{session_id}',
        'session_logs': f'Output/Sessions/{session_id}/logs',
        'session_keys': f'Output/Sessions/{session_id}/keys',
        'session_images': f'Output/Sessions/{session_id}/images',
        'session_metadata': f'Output/Sessions/{session_id}/metadata'
    }
    
    # Create directories
    for dir_name, dir_path in base_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dirs, session_id, timestamp


def setup_logging(base_dirs, session_id, timestamp):
    """Setup comprehensive logging with structured file organization"""
    # Log files with descriptive names
    session_log = os.path.join(base_dirs['session_logs'], f"execution_log_{timestamp}.txt")
    main_log = os.path.join(base_dirs['logs'], f"steganography_log_{timestamp}.txt")
    
    # Create logger
    logger = logging.getLogger('steganography')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handlers (both session and main logs)
    session_handler = logging.FileHandler(session_log, encoding='utf-8')
    session_handler.setLevel(logging.DEBUG)
    
    main_handler = logging.FileHandler(main_log, encoding='utf-8')
    main_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Set formatters
    session_handler.setFormatter(detailed_formatter)
    main_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(session_handler)
    logger.addHandler(main_handler)
    logger.addHandler(console_handler)
    
    # Log session startup information
    logger.info("="*80)
    logger.info("STEGANOGRAPHY SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Timestamp: {datetime.datetime.now()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info(f"Session Log: {session_log}")
    logger.info(f"Main Log: {main_log}")
    logger.info("="*80)
    
    return logger, session_log, main_log

class PrecisionFixedDWTLSBSteganography:
    def __init__(self, logger=None, base_dirs=None, session_id=None, timestamp=None):
        self.logger = logger or logging.getLogger('steganography')
        self.base_dirs = base_dirs or {}
        self.session_id = session_id or "default"
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.public_key = None
        self.private_key = None
        self.key_size = 2048
        self.ciphertext_length = self.key_size // 8  # 256 bytes
        self.session_metadata = {}
        
        self.logger.info("Initializing PrecisionFixedDWTLSBSteganography system")
        self.generate_rsa_keys()


    def generate_rsa_keys(self):
        """Generate RSA public and private key pair"""
        self.logger.info(f"Starting RSA-{self.key_size} key generation...")
        start_time = datetime.datetime.now()
        
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        end_time = datetime.datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        self.logger.info(f"RSA-{self.key_size} Keys generated successfully")
        self.logger.debug(f"Key generation time: {generation_time:.3f} seconds")
        self.logger.debug(f"Public exponent: 65537")
        self.logger.debug(f"Key size: {self.key_size} bits")
        print(f"RSA-{self.key_size} Keys generated successfully")


    def rsa_encrypt(self, message):
        """Encrypt message using RSA public key"""
        try:
            self.logger.info("Starting RSA encryption...")
            start_time = datetime.datetime.now()
            
            if isinstance(message, str):
                original_message = message
                message = message.encode('utf-8')
                self.logger.debug(f"Original message: '{original_message}'")
                self.logger.debug(f"Message length (characters): {len(original_message)}")
                self.logger.debug(f"Message length (bytes): {len(message)}")

            encrypted_message = self.public_key.encrypt(
                message,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            end_time = datetime.datetime.now()
            encryption_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"RSA encryption successful: {len(encrypted_message)} bytes")
            self.logger.debug(f"Encryption time: {encryption_time:.3f} seconds")
            self.logger.debug(f"Encrypted data size: {len(encrypted_message)} bytes")
            print(f"* RSA encryption: {len(encrypted_message)} bytes")
            return encrypted_message
        except Exception as e:
            self.logger.error(f"RSA Encryption Error: {e}")
            print(f"RSA Encryption Error: {e}")
            return None


    def rsa_decrypt(self, encrypted_message):
        """Decrypt message using RSA private key"""
        try:
            if len(encrypted_message) != self.ciphertext_length:
                print(f"Error: Wrong ciphertext length: {len(encrypted_message)} != {self.ciphertext_length}")
                return None


            decrypted_message = self.private_key.decrypt(
                encrypted_message,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            print(f"* RSA decryption successful")
            return decrypted_message.decode('utf-8')
        except Exception as e:
            print(f"RSA Decryption Error: {e}")
            return None


    def rgb_to_ycbcr(self, image):
        """Convert RGB image to YCbCr color space"""
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        return ycbcr


    def ycbcr_to_rgb(self, ycbcr_image):
        """Convert YCbCr image back to RGB"""
        rgb = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)
        return rgb


    def apply_dwt(self, channel):
        """Apply 2D Discrete Wavelet Transform"""
        # Ensure channel is float32 for DWT
        if channel.dtype != np.float32:
            channel = channel.astype(np.float32)
        coeffs = pywt.dwt2(channel, 'haar')
        return coeffs


    def inverse_dwt(self, coeffs):
        """Apply inverse 2D Discrete Wavelet Transform"""
        reconstructed = pywt.idwt2(coeffs, 'haar')
        return reconstructed


    def message_to_binary(self, message):
        """Convert message to binary string"""
        if isinstance(message, str):
            binary = ''.join(format(ord(char), '08b') for char in message)
        else:
            binary = ''.join(format(byte, '08b') for byte in message)
        return binary


    def embed_lsb_in_dwt_precise(self, coeffs, binary_message):
        """
        Embed LSB in DWT coefficients following exact pseudocode:
        Only use LH and HH subbands as specified in pseudocode
        """
        cA, (cH, cV, cD) = coeffs

        # Following pseudocode: only use LH and HH subbands
        # LH = cH (horizontal details), HH = cD (diagonal details)
        target_coeffs = np.concatenate([cH.flatten(), cD.flatten()])
        
        print(f"  Available coefficient space (LH+HH only): {len(target_coeffs)} bits")
        print(f"  Message length: {len(binary_message)} bits")

        if len(binary_message) > len(target_coeffs):
            print("Error: Message too long for available space")
            return None

        # ROBUST EMBEDDING: Use significant modifications that will survive DWT/IDWT
        # We'll modify coefficients by at least ±8.0 to ensure the change persists
        
        embedded_count = 0
        target_coeffs_modified = target_coeffs.copy()
        
        for i in range(len(binary_message)):
            if i < len(target_coeffs_modified):
                bit = int(binary_message[i])
                current_val = target_coeffs_modified[i]
                
                # Use a large modification that will definitely survive DWT roundtrip
                if bit == 1:
                    # Force coefficient to be significantly positive and > 8
                    if current_val >= 0:
                        target_coeffs_modified[i] = max(abs(current_val), 8.0) + 8.0
                    else:
                        target_coeffs_modified[i] = -max(abs(current_val), 8.0) - 8.0
                else:
                    # Force coefficient to have no fractional part and be < 8 in magnitude
                    int_part = int(current_val)
                    if abs(int_part) >= 8:
                        # Keep sign but reduce magnitude
                        target_coeffs_modified[i] = float(7 if current_val >= 0 else -7)
                    else:
                        target_coeffs_modified[i] = float(int_part)
                
                embedded_count += 1

        print(f"  Embedded {embedded_count} bits successfully with robust modifications")
        
        # Split back into LH (cH) and HH (cD) components
        cH_size = cH.size
        cH_new = target_coeffs_modified[:cH_size].reshape(cH.shape)
        cD_new = target_coeffs_modified[cH_size:].reshape(cD.shape)

        # Keep cA and cV unchanged as per pseudocode
        return cA, (cH_new, cV, cD_new)


    def extract_lsb_from_dwt_precise(self, coeffs):
        """
        Extract LSB from DWT coefficients following exact pseudocode:
        Only use LH and HH subbands as specified in pseudocode
        """
        cA, (cH, cV, cD) = coeffs

        # Following pseudocode: only use LH and HH subbands
        # LH = cH (horizontal details), HH = cD (diagonal details)
        target_coeffs = np.concatenate([cH.flatten(), cD.flatten()])

        # Use robust extraction matching the embedding approach
        
        required_bits = self.ciphertext_length * 8  # 2048 bits

        binary_message = ""
        for i in range(min(required_bits, len(target_coeffs))):
            coeff = target_coeffs[i]
            
            # Extract bit based on robust embedding pattern
            # If absolute value is >= 8, it's a "1" bit
            if abs(coeff) >= 8.0:
                binary_message += "1"
            else:
                binary_message += "0"

        print(f"  Extracted {len(binary_message)} bits from LH+HH subbands")
        return binary_message


    def embed_message(self, cover_image, secret_message):
        """Main embedding function"""
        print("Starting embedding process...")


        # Step 1: RSA Encrypt
        print("1. RSA Encrypting message...")
        encrypted_message = self.rsa_encrypt(secret_message)
        if encrypted_message is None:
            return None, 0


        # Step 2: Convert to YCbCr and extract Y channel (following pseudocode)
        print("2. Converting RGB to YCbCr...")
        ycbcr_image = self.rgb_to_ycbcr(cover_image)
        Y, Cb, Cr = cv2.split(ycbcr_image)


        # Step 3: Apply DWT to Y channel (following exact pseudocode)
        print("3. Applying DWT to Y channel...")
        y_coeffs = self.apply_dwt(Y)


        # Step 4: Convert encrypted message to binary
        binary_message = self.message_to_binary(encrypted_message)
        print(f"   Binary message length: {len(binary_message)} bits")


        # Step 5: Embed using precision-preserving LSB in LH and HH subbands
        print("4. Embedding message in LH and HH subbands...")
        modified_coeffs = self.embed_lsb_in_dwt_precise(y_coeffs, binary_message)


        if modified_coeffs is None:
            return None, 0


        # Step 6: Apply inverse DWT to reconstruct Y channel
        print("5. Applying inverse DWT to reconstruct Y channel...")
        modified_y = self.inverse_dwt(modified_coeffs)


        # Ensure values are in valid range
        modified_y = np.clip(modified_y, 0, 255).astype(np.uint8)


        # Step 7: Reconstruct image with modified Y channel
        print("6. Converting YCbCr back to RGB...")
        stego_ycbcr = cv2.merge([modified_y, Cb, Cr])
        stego_rgb = self.ycbcr_to_rgb(stego_ycbcr)


        print("* Embedding completed successfully!")
        
        # Update metadata
        self.session_metadata.update({
            'embedding_completed': True,
            'original_message_length': len(secret_message),
            'encrypted_message_length': len(encrypted_message),
            'binary_message_length': len(binary_message),
            'embedding_timestamp': datetime.datetime.now().isoformat()
        })
        
        return stego_rgb, len(binary_message)


    def extract_message(self, stego_image):
        """Main extraction function"""
        print("Starting extraction process...")


        # Step 1: Convert to YCbCr and extract Y channel (following pseudocode)
        print("1. Converting RGB to YCbCr...")
        ycbcr_image = self.rgb_to_ycbcr(stego_image)
        Y, Cb, Cr = cv2.split(ycbcr_image)


        # Step 2: Apply DWT to Y channel (following exact pseudocode)
        print("2. Applying DWT to Y channel...")
        y_coeffs = self.apply_dwt(Y)


        # Step 3: Extract from LH and HH subbands only
        print("3. Extracting message from LH and HH subbands...")
        binary_message = self.extract_lsb_from_dwt_precise(y_coeffs)


        if binary_message is None:
            return None


        # Step 4: Convert binary to encrypted message
        print("4. Converting binary to encrypted message...")
        try:
            encrypted_message = bytes([int(binary_message[i:i+8], 2)
                                      for i in range(0, len(binary_message), 8)])
            print(f"   * Reconstructed encrypted message: {len(encrypted_message)} bytes")
        except Exception as e:
            print(f"   Error converting binary to encrypted message: {e}")
            return None


        # Step 5: RSA Decrypt
        print("5. RSA Decrypting message...")
        decrypted_message = self.rsa_decrypt(encrypted_message)


        print("* Extraction completed!")
        return decrypted_message


    def save_keys(self, private_key_path='private_key.pem', public_key_path='public_key.pem'):
        """Save RSA keys to files"""
        try:
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )


            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )


            with open(private_key_path, 'wb') as f:
                f.write(private_pem)


            with open(public_key_path, 'wb') as f:
                f.write(public_pem)


            print(f"* RSA keys saved to '{private_key_path}' and '{public_key_path}'")
            return True
        except Exception as e:
            print(f"Error saving keys: {e}")
            return False

    def save_keys_structured(self):
        """Save RSA keys to structured directories"""
        try:
            # Save to both session and main directories
            session_private = os.path.join(self.base_dirs['session_keys'], f'private_key_{self.timestamp}.pem')
            session_public = os.path.join(self.base_dirs['session_keys'], f'public_key_{self.timestamp}.pem')
            main_private = os.path.join(self.base_dirs['keys'], 'private_key.pem')
            main_public = os.path.join(self.base_dirs['keys'], 'public_key.pem')
            
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            # Save to session directory
            with open(session_private, 'wb') as f:
                f.write(private_pem)
            with open(session_public, 'wb') as f:
                f.write(public_pem)
                
            # Save to main directory (latest keys)
            with open(main_private, 'wb') as f:
                f.write(private_pem)
            with open(main_public, 'wb') as f:
                f.write(public_pem)

            self.logger.info(f"RSA keys saved to session: {session_private}, {session_public}")
            self.logger.info(f"RSA keys saved to main: {main_private}, {main_public}")
            print(f"* RSA keys saved to session and main directories")
            
            # Update metadata
            self.session_metadata.update({
                'private_key_path': session_private,
                'public_key_path': session_public,
                'keys_saved_timestamp': datetime.datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving keys: {e}")
            print(f"Error saving keys: {e}")
            return False
    
    def save_stego_image_structured(self, stego_image, original_image_name="unknown"):
        """Save steganographic image to structured directories"""
        try:
            # Generate descriptive filename
            clean_name = os.path.splitext(os.path.basename(original_image_name))[0]
            session_filename = f"{clean_name}_stego_{self.timestamp}.png"
            main_filename = f"precision_fixed_stego_{self.timestamp}.png"
            
            session_path = os.path.join(self.base_dirs['session_images'], session_filename)
            main_path = os.path.join(self.base_dirs['stego_images'], main_filename)
            
            # Convert RGB to BGR for OpenCV
            stego_bgr = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
            
            # Save to both locations
            cv2.imwrite(session_path, stego_bgr)
            cv2.imwrite(main_path, stego_bgr)
            
            self.logger.info(f"Stego image saved to session: {session_path}")
            self.logger.info(f"Stego image saved to main: {main_path}")
            print(f"* Stego image saved to session and main directories")
            
            # Update metadata
            self.session_metadata.update({
                'stego_image_session_path': session_path,
                'stego_image_main_path': main_path,
                'original_image_name': original_image_name,
                'stego_image_saved_timestamp': datetime.datetime.now().isoformat()
            })
            
            return session_path, main_path
        except Exception as e:
            self.logger.error(f"Error saving stego image: {e}")
            print(f"Error saving stego image: {e}")
            return None, None
    
    def save_session_metadata(self, additional_metadata=None):
        """Save session metadata to JSON file"""
        try:
            import json
            
            if additional_metadata:
                self.session_metadata.update(additional_metadata)
            
            # Add system information
            self.session_metadata.update({
                'session_id': self.session_id,
                'timestamp': self.timestamp,
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'session_end_timestamp': datetime.datetime.now().isoformat()
            })
            
            metadata_file = os.path.join(self.base_dirs['session_metadata'], f'session_metadata_{self.timestamp}.json')
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Session metadata saved to: {metadata_file}")
            print(f"* Session metadata saved to: {metadata_file}")
            
            return metadata_file
        except Exception as e:
            self.logger.error(f"Error saving session metadata: {e}")
            print(f"Error saving session metadata: {e}")
            return None


    def load_keys(self, private_key_path='private_key.pem', public_key_path='public_key.pem'):
        """Load RSA keys from files"""
        try:
            with open(private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )


            with open(public_key_path, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )


            print(f"* RSA keys loaded from '{private_key_path}' and '{public_key_path}'")
            return True
        except Exception as e:
            print(f"Error loading keys: {e}")
            return False


def test_precision_fixed_structured(base_dirs, session_id, timestamp, logger=None):
    """Test the precision-fixed version with structured file management"""
    if logger is None:
        logger = logging.getLogger('steganography')
        
    print("=== PRECISION-FIXED DWT-LSB STEGANOGRAPHY TEST ===")
    logger.info("Starting precision-fixed steganography test with structured organization")

    # Get inputs
    image_path = input("Enter image path: ").strip()
    logger.info(f"Image path provided: {image_path}")
    if not image_path:
        logger.error("No image path provided")
        print("Error: Image path required")
        return

    secret_message = input("Enter secret message: ").strip() or "gixermonoton"
    logger.info(f"Secret message length: {len(secret_message)} characters")
    logger.debug(f"Secret message content: '{secret_message}'")

    # Load image
    logger.info(f"Attempting to load image: {image_path}")
    cover_image = cv2.imread(image_path)
    if cover_image is None:
        logger.error(f"Could not load image from '{image_path}'")
        print(f"Error: Could not load image from '{image_path}'")
        return

    original_shape = cover_image.shape
    logger.debug(f"Original image shape: {original_shape}")
    cover_image = cv2.resize(cover_image, (512, 512))
    logger.info(f"Image loaded and resized from {original_shape[:2]} to 512x512")
    print(f"* Image loaded and resized to 512x512")

    # Initialize system with structured directories
    logger.info("Initializing steganography system with structured file management")
    stego_system = PrecisionFixedDWTLSBSteganography(logger, base_dirs, session_id, timestamp)
    
    # Record initial metadata
    stego_system.session_metadata.update({
        'input_image_path': image_path,
        'input_image_shape': original_shape,
        'secret_message_length': len(secret_message),
        'test_start_timestamp': datetime.datetime.now().isoformat()
    })

    # Embed message
    print("\n" + "="*50)
    stego_image, message_length = stego_system.embed_message(cover_image, secret_message)

    if stego_image is not None:
        # Save stego image using structured approach
        session_path, main_path = stego_system.save_stego_image_structured(stego_image, image_path)
        
        # Save keys using structured approach
        stego_system.save_keys_structured()

        # Extract to verify
        print("\n" + "="*50)
        print("VERIFICATION PHASE")
        extracted_message = stego_system.extract_message(stego_image)

        # Record results in metadata
        test_results = {
            'original_message': secret_message,
            'extracted_message': extracted_message,
            'success': extracted_message == secret_message,
            'test_end_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save session metadata
        stego_system.save_session_metadata(test_results)

        print(f"\n=== FINAL RESULTS ===")
        print(f"Original message:  '{secret_message}'")
        print(f"Extracted message: '{extracted_message}'")
        print(f"Success: {extracted_message == secret_message}")
        print(f"Session files saved to: {base_dirs['session']}")

        if extracted_message == secret_message:
            print("SUCCESS: Precision-fixed method works!")
            logger.info("Test completed successfully")
        else:
            print("ERROR: Still failing - may need alternative approach")
            logger.error("Test failed - extraction did not match original")
    else:
        print("ERROR: Embedding failed!")
        logger.error("Embedding failed")
        stego_system.save_session_metadata({'error': 'Embedding failed'})

def test_precision_fixed(logger=None):
    """Test the precision-fixed version (legacy function for compatibility)"""
    if logger is None:
        logger = logging.getLogger('steganography')
        
    print("=== PRECISION-FIXED DWT-LSB STEGANOGRAPHY TEST ===")
    logger.info("Starting precision-fixed steganography test")

    # Get inputs
    image_path = input("Enter image path: ").strip()
    logger.info(f"Image path provided: {image_path}")
    if not image_path:
        logger.error("No image path provided")
        print("Error: Image path required")
        return

    secret_message = input("Enter secret message: ").strip() or "gixermonoton"
    logger.info(f"Secret message length: {len(secret_message)} characters")
    logger.debug(f"Secret message content: '{secret_message}'")


    # Load image
    logger.info(f"Attempting to load image: {image_path}")
    cover_image = cv2.imread(image_path)
    if cover_image is None:
        logger.error(f"Could not load image from '{image_path}'")
        print(f"Error: Could not load image from '{image_path}'")
        return

    original_shape = cover_image.shape
    logger.debug(f"Original image shape: {original_shape}")
    cover_image = cv2.resize(cover_image, (512, 512))
    logger.info(f"Image loaded and resized from {original_shape[:2]} to 512x512")
    print(f"* Image loaded and resized to 512x512")

    # Initialize system
    logger.info("Initializing steganography system")
    stego_system = PrecisionFixedDWTLSBSteganography(logger)


    # Embed message
    print("\n" + "="*50)
    stego_image, message_length = stego_system.embed_message(cover_image, secret_message)


    if stego_image is not None:
        # Save stego image
        cv2.imwrite('precision_fixed_stego.png', cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        print(f"* Stego image saved as 'precision_fixed_stego.png'")


        # Save keys
        stego_system.save_keys()


        # Extract to verify
        print("\n" + "="*50)
        print("VERIFICATION PHASE")
        extracted_message = stego_system.extract_message(stego_image)


        print(f"\n=== FINAL RESULTS ===")
        print(f"Original message:  '{secret_message}'")
        print(f"Extracted message: '{extracted_message}'")
        print(f"Success: {extracted_message == secret_message}")


        if extracted_message == secret_message:
            print("SUCCESS: Precision-fixed method works!")
        else:
            print("ERROR: Still failing - may need alternative approach")
    else:
        print("ERROR: Embedding failed!")


def interactive_mode(logger=None):
    """Interactive mode"""
    if logger is None:
        logger = logging.getLogger('steganography')
        
    print("=== PRECISION-FIXED DWT-LSB STEGANOGRAPHY ===")
    logger.info("Starting interactive mode")

    image_path = input("Enter image path: ").strip()
    secret_message = input("Enter secret message: ").strip()
    
    logger.info(f"Image path: {image_path}")
    logger.info(f"Message length: {len(secret_message)} characters")

    cover_image = cv2.imread(image_path)
    if cover_image is None:
        logger.error(f"Could not load image from '{image_path}'")
        print("Error loading image")
        return

    original_shape = cover_image.shape
    cover_image = cv2.resize(cover_image, (512, 512))
    logger.info(f"Image resized from {original_shape[:2]} to 512x512")

    stego_system = PrecisionFixedDWTLSBSteganography(logger)


    # Embed
    stego_image, _ = stego_system.embed_message(cover_image, secret_message)


    if stego_image is not None:
        output_path = input("Enter output filename: ").strip() or "precision_stego.png"
        cv2.imwrite(output_path, cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        print(f"* Stego image saved as '{output_path}'")


        stego_system.save_keys()


        # Extract
        extracted = stego_system.extract_message(stego_image)
        print(f"\nOriginal:  '{secret_message}'")
        print(f"Extracted: '{extracted}'")
        print(f"Success:   {extracted == secret_message}")


def interactive_mode_structured(base_dirs, session_id, timestamp, logger=None):
    """Interactive mode with structured file management"""
    if logger is None:
        logger = logging.getLogger('steganography')
        
    print("=== PRECISION-FIXED DWT-LSB STEGANOGRAPHY (Interactive) ===")
    logger.info("Starting interactive mode with structured organization")

    image_path = input("Enter image path: ").strip()
    secret_message = input("Enter secret message: ").strip()
    
    logger.info(f"Image path: {image_path}")
    logger.info(f"Message length: {len(secret_message)} characters")

    cover_image = cv2.imread(image_path)
    if cover_image is None:
        logger.error(f"Could not load image from '{image_path}'")
        print("Error loading image")
        return

    original_shape = cover_image.shape
    cover_image = cv2.resize(cover_image, (512, 512))
    logger.info(f"Image resized from {original_shape[:2]} to 512x512")

    # Initialize system with structured directories
    stego_system = PrecisionFixedDWTLSBSteganography(logger, base_dirs, session_id, timestamp)
    
    # Record metadata
    stego_system.session_metadata.update({
        'mode': 'interactive',
        'input_image_path': image_path,
        'input_image_shape': original_shape,
        'secret_message_length': len(secret_message),
        'session_start_timestamp': datetime.datetime.now().isoformat()
    })

    # Embed
    stego_image, _ = stego_system.embed_message(cover_image, secret_message)

    if stego_image is not None:
        output_filename = input("Enter output filename (or press Enter for default): ").strip()
        if not output_filename:
            output_filename = f"interactive_stego_{timestamp}.png"
        
        # Save using structured approach
        session_path, main_path = stego_system.save_stego_image_structured(stego_image, image_path)
        stego_system.save_keys_structured()

        # Extract
        extracted = stego_system.extract_message(stego_image)
        
        # Save final metadata
        results = {
            'original_message': secret_message,
            'extracted_message': extracted,
            'success': extracted == secret_message,
            'output_filename': output_filename,
            'session_end_timestamp': datetime.datetime.now().isoformat()
        }
        stego_system.save_session_metadata(results)
        
        print(f"\nOriginal:  '{secret_message}'")
        print(f"Extracted: '{extracted}'")
        print(f"Success:   {extracted == secret_message}")
        print(f"Session files saved to: {base_dirs['session']}")


if __name__ == "__main__":
    # Setup structured directories and logging
    base_dirs, session_id, timestamp = setup_structured_directories()
    logger, session_log, main_log = setup_logging(base_dirs, session_id, timestamp)
    
    print("=== PRECISION-FIXED STEGANOGRAPHY ===")
    print("1. Test precision-fixed method (Structured)")
    print("2. Interactive mode (Structured)")
    print("3. Legacy test method")
    print(f"Session ID: {session_id}")
    print(f"Session Log: {session_log}")
    print(f"Main Log: {main_log}")
    print(f"Session Directory: {base_dirs['session']}")

    choice = input("Choose (1-3): ").strip()
    logger.info(f"User selected option: {choice}")

    try:
        if choice == "1":
            test_precision_fixed_structured(base_dirs, session_id, timestamp, logger)
        elif choice == "2":
            interactive_mode_structured(base_dirs, session_id, timestamp, logger)
        else:
            print("Using legacy mode...")
            test_precision_fixed(logger)
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}")
        print(f"Error: {e}")
    finally:
        logger.info("="*80)
        logger.info("STEGANOGRAPHY SESSION COMPLETED")
        logger.info("="*80)