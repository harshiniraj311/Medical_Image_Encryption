from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import hashlib
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store encryption state
class EncryptionState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.original_image = None
        self.encrypted_image = None
        self.decrypted_image = None
        self.layers = {}
        self.bit_planes = {}
        self.keys = {}
        self.sboxes = {}
        self.round_keys = {}
        self.diff_seq = None
        self.stream = None
        self.stream_L3 = None
        self.h = None
        self.w = None
        self.K_L1 = None
        self.K_L2 = None
        self.K_L3 = None
        self.K_GLOBAL = None
        self.round_keys_L1 = None
        self.round_keys_L2 = None
        self.diff_seq_L1 = None
        self.stream_L1 = None
        self.stream_L3_data = None

state = EncryptionState()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# AES S-box (fixed base)
# -------------------------------
AES_SBOX = np.array([
    99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
    202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
    183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
    4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
    9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
    83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
    208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
    81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
    205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
    96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
    224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
    231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
    186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
    112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
    225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
    140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22
], dtype=np.uint8)

# -------------------------------
# Key Derivation Function (KDF)
# -------------------------------
def kdf(master_key, context, out_len=32):
    data = (master_key + context).encode()
    output = bytearray()
    current = hashlib.sha256(data).digest()

    while len(output) < out_len:
        current = hashlib.sha256(current + data).digest()
        output.extend(current)

    return bytes(output[:out_len])

# -------------------------------
# SHA-256 based keystream generator
# -------------------------------
def sha256_stream(key_bytes, length):
    out = bytearray()
    counter = 0

    while len(out) < length:
        data = key_bytes + counter.to_bytes(4, 'big')
        out.extend(hashlib.sha256(data).digest())
        counter += 1

    return np.frombuffer(out[:length], dtype=np.uint8)

# -------------------------------
# Dynamic S-box generation
# -------------------------------
def generate_dynamic_sbox(key_bytes):
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    sbox = AES_SBOX.copy()
    rng.shuffle(sbox)
    return sbox

def generate_sbox_L2(key_bytes):
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    sbox = np.arange(256, dtype=np.uint8)
    rng.shuffle(sbox)
    return sbox

# -------------------------------
# Global diffusion encryption
# -------------------------------
def global_diffusion_encrypt(img, key_bytes):
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.uint8)
    
    img_hash = hashlib.sha256(img.tobytes()).digest()
    seed_material = key_bytes + img_hash
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    
    rng = np.random.default_rng(seed)
    seq = rng.integers(0, 256, size=h * w, dtype=np.uint8).reshape(h, w)
    
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                out[i, j] = img[i, j] ^ seq[i, j]
            elif j == 0:
                out[i, j] = img[i, j] ^ out[i-1, w-1] ^ seq[i, j]
            else:
                out[i, j] = img[i, j] ^ out[i, j-1] ^ seq[i, j]
    
    return out, seq

# -------------------------------
# Global diffusion decryption
# -------------------------------
def global_diffusion_decrypt(cipher, key_bytes, seq):
    h, w = cipher.shape
    out = np.zeros_like(cipher, dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                out[i, j] = cipher[i, j] ^ seq[i, j]
            elif j == 0:
                out[i, j] = cipher[i, j] ^ cipher[i-1, w-1] ^ seq[i, j]
            else:
                out[i, j] = cipher[i, j] ^ cipher[i, j-1] ^ seq[i, j]
    
    return out

# -------------------------------
# Extract bit planes
# -------------------------------
def extract_bit_plane(image, bit):
    return ((image >> bit) & 1).astype(np.uint8) << bit

# -------------------------------
# Process image (encryption)
# -------------------------------
def process_image(image_path, master_key="PatientID_StudyID_PrivateSecret"):
    # Load and resize image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.uint8)
    
    h, w = img.shape
    
    # Generate keys
    K_L1 = kdf(master_key, "LAYER1_KEY", 32)
    K_L2 = kdf(master_key, "LAYER2_KEY", 32)
    K_L3 = kdf(master_key, "LAYER3_KEY", 32)
    K_GLOBAL = kdf(master_key, "GLOBAL_DIFFUSION", 32)
    
    # Extract bit planes
    bit_planes = {}
    for b in range(8):
        bit_planes[b] = extract_bit_plane(img, b)
    
    # Semantic layers
    L1_true = bit_planes[7] | bit_planes[6]
    L2_true = bit_planes[5] | bit_planes[4] | bit_planes[3]
    L3_true = bit_planes[2] | bit_planes[1] | bit_planes[0]
    
    # Generate S-boxes
    SBOX_L1 = generate_dynamic_sbox(K_L1)
    SBOX_L2 = generate_sbox_L2(K_L2)
    
    INV_SBOX_L1 = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        INV_SBOX_L1[SBOX_L1[i]] = i
    
    INV_SBOX_L2 = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        INV_SBOX_L2[SBOX_L2[i]] = i
    
    # Round keys
    rng = np.random.default_rng(int.from_bytes(K_L1[:4], "big"))
    round_keys_L1 = rng.integers(0, 256, size=h * w, dtype=np.uint8)
    
    rng = np.random.default_rng(int.from_bytes(K_L2[:4], "big"))
    round_keys_L2 = rng.integers(0, 256, size=h * w, dtype=np.uint8)
    
    # Diffusion sequence
    rng = np.random.default_rng(int.from_bytes(K_L1[4:8], "big"))
    diff_seq = rng.integers(0, 256, size=h * w, dtype=np.uint8).reshape(h, w)
    
    # Streams
    stream_L1 = sha256_stream(K_L1, h * w).reshape(h, w)
    stream_L3_data = sha256_stream(K_L3, h * w).reshape(h, w)
    
    # ==================== LAYER 1 ENCRYPTION ====================
    L1_sub = SBOX_L1[L1_true]
    
    L1_blk = np.zeros_like(L1_sub, dtype=np.uint8)
    idx = 0
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = L1_sub[i:i+4, j:j+4].copy()
            if block.shape != (4, 4):
                continue
            
            rk = round_keys_L1[idx:idx+16].reshape(4, 4)
            idx += 16
            block ^= rk
            
            for r in range(4):
                block[r,1] ^= block[r,0]
                block[r,2] ^= block[r,1]
                block[r,3] ^= block[r,2]
            
            for c in range(4):
                block[1,c] ^= block[0,c]
                block[2,c] ^= block[1,c]
                block[3,c] ^= block[2,c]
            
            L1_blk[i:i+4, j:j+4] = block
    
    L1_diff = np.zeros_like(L1_blk, dtype=np.uint8)
    prev = 0
    for i in range(h):
        for j in range(w):
            L1_diff[i, j] = L1_blk[i, j] ^ prev ^ diff_seq[i, j]
            prev = L1_diff[i, j]
    
    L1_enc = L1_diff ^ stream_L1
    
    # ==================== LAYER 2 ENCRYPTION ====================
    L2_enc = np.zeros_like(L2_true, dtype=np.uint8)
    idx = 0
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = L2_true[i:i+4, j:j+4].copy()
            if block.shape != (4, 4):
                continue
            
            block = SBOX_L2[block]
            block ^= round_keys_L2[idx:idx+16].reshape(4, 4)
            idx += 16
            
            for r in range(4):
                block[r,1] ^= block[r,0]
                block[r,2] ^= block[r,1]
                block[r,3] ^= block[r,2]
            
            block = np.roll(block, 1, axis=1)
            L2_enc[i:i+4, j:j+4] = block
    
    # ==================== LAYER 3 ENCRYPTION ====================
    L3_enc = L3_true ^ stream_L3_data
    
    # ==================== FINAL ENCRYPTION ====================
    img_enc_layers = L1_enc ^ L2_enc ^ L3_enc
    img_cipher, global_seq = global_diffusion_encrypt(img_enc_layers, K_GLOBAL)
    
    # ==================== LAYER 1 DECRYPTION ====================
    L1_diff_dec = L1_enc ^ stream_L1
    
    L1_blk_dec = np.zeros_like(L1_diff_dec, dtype=np.uint8)
    prev = 0
    for i in range(h):
        for j in range(w):
            L1_blk_dec[i, j] = L1_diff_dec[i, j] ^ prev ^ diff_seq[i, j]
            prev = L1_diff_dec[i, j]
    
    L1_sub_dec = np.zeros_like(L1_blk_dec, dtype=np.uint8)
    idx = 0
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = L1_blk_dec[i:i+4, j:j+4].copy()
            if block.shape != (4, 4):
                continue
            
            for c in range(4):
                block[3,c] ^= block[2,c]
                block[2,c] ^= block[1,c]
                block[1,c] ^= block[0,c]
            
            for r in range(4):
                block[r,3] ^= block[r,2]
                block[r,2] ^= block[r,1]
                block[r,1] ^= block[r,0]
            
            rk = round_keys_L1[idx:idx+16].reshape(4, 4)
            idx += 16
            block ^= rk
            
            L1_sub_dec[i:i+4, j:j+4] = block
    
    L1_dec = INV_SBOX_L1[L1_sub_dec]
    
    # ==================== LAYER 2 DECRYPTION ====================
    L2_dec = np.zeros_like(L2_enc, dtype=np.uint8)
    idx = 0
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = L2_enc[i:i+4, j:j+4].copy()
            if block.shape != (4, 4):
                continue
            
            block = np.roll(block, -1, axis=1)
            
            for r in range(4):
                block[r,3] ^= block[r,2]
                block[r,2] ^= block[r,1]
                block[r,1] ^= block[r,0]
            
            block ^= round_keys_L2[idx:idx+16].reshape(4, 4)
            idx += 16
            
            block = INV_SBOX_L2[block]
            L2_dec[i:i+4, j:j+4] = block
    
    # ==================== LAYER 3 DECRYPTION ====================
    L3_dec = L3_enc ^ stream_L3_data
    
    # ==================== FULL DECRYPTION ====================
    img_enc_layers_dec = global_diffusion_decrypt(img_cipher, K_GLOBAL, global_seq)
    L1_enc_rec = img_enc_layers_dec ^ L2_enc ^ L3_enc
    L2_enc_rec = img_enc_layers_dec ^ L1_enc ^ L3_enc
    L3_enc_rec = img_enc_layers_dec ^ L1_enc ^ L2_enc
    img_reconstructed = (L1_dec | L2_dec | L3_dec).astype(np.uint8)
    
    # Store all data in state
    state.original_image = img
    state.encrypted_image = img_cipher
    state.decrypted_image = img_reconstructed
    state.layers = {
        'L1_true': L1_true, 'L1_enc': L1_enc, 'L1_dec': L1_dec,
        'L2_true': L2_true, 'L2_enc': L2_enc, 'L2_dec': L2_dec,
        'L3_true': L3_true, 'L3_enc': L3_enc, 'L3_dec': L3_dec,
    }
    state.bit_planes = bit_planes
    state.keys = {'K_L1': K_L1, 'K_L2': K_L2, 'K_L3': K_L3, 'K_GLOBAL': K_GLOBAL}
    state.sboxes = {'SBOX_L1': SBOX_L1, 'SBOX_L2': SBOX_L2, 
                   'INV_SBOX_L1': INV_SBOX_L1, 'INV_SBOX_L2': INV_SBOX_L2}
    state.round_keys = {'L1': round_keys_L1, 'L2': round_keys_L2}
    state.diff_seq = diff_seq
    state.stream = stream_L1
    state.stream_L3 = stream_L3_data
    state.h = h
    state.w = w
    state.global_seq = global_seq
    
    return {
        'success': True,
        'message': 'Image processed successfully'
    }

# -------------------------------
# Convert numpy array to base64 image
# -------------------------------
def array_to_base64(img_array):
    if img_array is None:
        return None
    
    # Ensure uint8
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Convert to PIL Image
    from PIL import Image
    img_pil = Image.fromarray(img_array)
    
    # Save to bytes
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'})
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + '_' + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Process image
    master_key = request.form.get('master_key', 'PatientID_StudyID_PrivateSecret')
    result = process_image(filepath, master_key)
    
    if result['success']:
        # Clean up uploaded file
        os.remove(filepath)
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Processing failed'})

def normalize_for_display(img):
    img = img.astype(np.float32)
    minv, maxv = img.min(), img.max()
    if maxv > minv:
        img = (img - minv) / (maxv - minv) * 255.0
    return img.astype(np.uint8)

@app.route('/get_image/<image_type>')
def get_image(image_type):
    """Return image as base64"""
    if image_type == 'original':
        img = state.original_image
    elif image_type == 'encrypted':
        img = state.encrypted_image
    elif image_type == 'decrypted':
        img = state.decrypted_image
    elif image_type == 'bitplane':
        bit = int(request.args.get('bit', 7))
        img = state.bit_planes.get(bit)
    elif image_type == 'layer1_true':
        img = state.layers.get('L1_true')
    elif image_type == 'layer1_enc':
        img = state.layers.get('L1_enc')
    elif image_type == 'layer1_dec':
        img = state.layers.get('L1_dec')
    elif image_type == 'layer2_true':
        img = state.layers.get('L2_true')
    elif image_type == 'layer2_enc':
        img = state.layers.get('L2_enc')
    elif image_type == 'layer2_dec':
        img = state.layers.get('L2_dec')
    elif image_type == 'layer3_true':
        img = state.layers.get('L3_true')
    elif image_type == 'layer3_enc':
        img = state.layers.get('L3_enc')
    elif image_type == 'layer3_dec':
        img = state.layers.get('L3_dec')
    else:
        return jsonify({'success': False, 'error': 'Invalid image type'})
    
    if img is None:
        return jsonify({'success': False, 'error': 'Image not found'})
    
    img_disp = normalize_for_display(img)
    img_base64 = array_to_base64(img_disp)
    return jsonify({'success': True, 'image': img_base64})

@app.route('/get_metrics')
def get_metrics():
    """Calculate and return encryption metrics"""
    if state.original_image is None or state.encrypted_image is None:
        return jsonify({'success': False, 'error': 'No image processed'})
    
    def image_entropy(img):
        hist = np.bincount(img.flatten(), minlength=256)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))
    
    def correlation(img, mode='horizontal'):
        h, w = img.shape
        if mode == 'horizontal':
            x = img[:, :-1].flatten()
            y = img[:, 1:].flatten()
        elif mode == 'vertical':
            x = img[:-1, :].flatten()
            y = img[1:, :].flatten()
        elif mode == 'diagonal':
            x = img[:-1, :-1].flatten()
            y = img[1:, 1:].flatten()
        if len(x) < 2 or len(y) < 2:
            return 0
        return np.corrcoef(x, y)[0, 1]
    
    def npcr(img1, img2):
        return float(np.sum(img1 != img2) / img1.size * 100)
    
    def uaci(img1, img2):
        return float(np.mean(np.abs(img1.astype(np.int16) - img2.astype(np.int16)) / 255) * 100)
    
    # Create modified image for NPCR/UACI
    img_mod = state.original_image.copy()
    img_mod[0, 0] ^= 1
    
    # Re-encrypt modified image
    h, w = state.h, state.w
    bp_mod = {}
    for b in range(8):
        bp_mod[b] = ((img_mod >> b) & 1).astype(np.uint8) << b
    
    L1m = bp_mod[7] | bp_mod[6]
    L2m = bp_mod[5] | bp_mod[4] | bp_mod[3]
    L3m = bp_mod[2] | bp_mod[1] | bp_mod[0]
    
    # Layer 1 re-encryption
    L1m_sub = state.sboxes['SBOX_L1'][L1m]
    L1m_blk = np.zeros_like(L1m_sub)
    idx = 0
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = L1m_sub[i:i+4, j:j+4].copy()
            if block.shape != (4,4):
                continue
            rk = state.round_keys['L1'][idx:idx+16].reshape(4,4)
            idx += 16
            block ^= rk
            for r in range(4):
                block[r,1] ^= block[r,0]
                block[r,2] ^= block[r,1]
                block[r,3] ^= block[r,2]
            for c in range(4):
                block[1,c] ^= block[0,c]
                block[2,c] ^= block[1,c]
                block[3,c] ^= block[2,c]
            L1m_blk[i:i+4, j:j+4] = block
    
    L1m_diff = np.zeros_like(L1m_blk)
    prev = 0
    for i in range(h):
        for j in range(w):
            L1m_diff[i,j] = L1m_blk[i,j] ^ prev ^ state.diff_seq[i,j]
            prev = L1m_diff[i,j]
    L1m_enc = L1m_diff ^ state.stream
    
    # Layer 2 re-encryption
    L2m_enc = np.zeros_like(L2m)
    idx = 0
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = L2m[i:i+4, j:j+4].copy()
            if block.shape != (4,4):
                continue
            block = state.sboxes['SBOX_L2'][block]
            block ^= state.round_keys['L2'][idx:idx+16].reshape(4,4)
            idx += 16
            for r in range(4):
                block[r,1] ^= block[r,0]
                block[r,2] ^= block[r,1]
                block[r,3] ^= block[r,2]
            block = np.roll(block, 1, axis=1)
            L2m_enc[i:i+4, j:j+4] = block
    
    # Layer 3 re-encryption
    L3m_enc = L3m ^ state.stream_L3
    
    # Final encryption
    img_enc_mod = L1m_enc ^ L2m_enc ^ L3m_enc
    img_enc_mod, _ = global_diffusion_encrypt(img_enc_mod, state.keys['K_GLOBAL'])
    
    # Calculate metrics
    layers = {
        "Layer-1 (Coarse)": (state.layers['L1_true'], state.layers['L1_enc']),
        "Layer-2 (Detail)": (state.layers['L2_true'], state.layers['L2_enc']),
        "Layer-3 (Fine)": (state.layers['L3_true'], state.layers['L3_enc']),
        "Final Image": (state.encrypted_image, img_enc_mod)
    }
    
    results = {}
    for name, (orig, enc) in layers.items():
        results[name] = {
            'entropy': round(image_entropy(enc), 4),
            'corr_h': round(correlation(enc, 'horizontal'), 4),
            'corr_v': round(correlation(enc, 'vertical'), 4),
            'corr_d': round(correlation(enc, 'diagonal'), 4),
            'npcr': round(npcr(orig, enc), 2),
            'uaci': round(uaci(orig, enc), 2)
        }
    
    return jsonify({'success': True, 'metrics': results})

@app.route('/reset')
def reset_state():
    state.reset()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)