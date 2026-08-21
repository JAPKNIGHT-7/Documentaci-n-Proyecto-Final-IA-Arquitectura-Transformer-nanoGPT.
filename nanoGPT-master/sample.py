"""
Sample from a trained nanoGPT model (char-level, ej. Quijote de la Mancha).

VERSION MODIFICADA: en cada paso de generacion, muestra en pantalla los
candidatos mas probables para el siguiente caracter (logits -> softmax),
antes de que el modelo elija uno.

Este archivo esta basado en el sample.py original de nanoGPT
(https://github.com/karpathy/nanoGPT). NO SE MODIFICO model.py:
el bucle de generacion se reescribe aqui mismo, llamando al modelo
como una "caja negra" (model(idx_cond)) igual que lo hace generate()
internamente.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'          # 'resume' (desde out_dir) o una variante de gpt2
out_dir = 'out-quijote-char'  # carpeta donde train.py guardo el checkpoint
start = "Don quijote"  # texto inicial (prompt)
num_samples = 1                # cuantas muestras generar
max_new_tokens = 200            # cuantos caracteres generar
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# --- opciones nuevas para el modo visual ---
show_probs = True     # True = muestra los candidatos en cada paso (para el video)
top_k_display = 4     # cuantos candidatos mostrar por paso (ej: 4)
# -----------------------------------------------------------------------------
exec(open('configurator.py').read())  # permite sobreescribir estas variables desde la terminal
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- cargar el modelo entrenado (esto no cambia respecto al original) ---
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# --- cargar el vocabulario de caracteres (meta.pkl generado por prepare.py) ---
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Cargando vocabulario desde {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


def generate_visual(model, idx, max_new_tokens, temperature=1.0, top_k=None,
                     show_probs=True, top_k_display=4):
    """
    @brief Genera texto mostrando las probabilidades de los candidatos.

    @details
    Reescribe el proceso de generación de model.py para mostrar los candidatos
    más probables y sus probabilidades antes de seleccionar el siguiente carácter.

    @param model Modelo GPT entrenado.
    @param idx Secuencia inicial de tokens.
    @param max_new_tokens Número de tokens a generar.
    @param temperature Temperatura de generación.
    @param top_k Candidatos considerados durante el muestreo.
    @param show_probs Muestra u oculta las probabilidades.
    @param top_k_display Número de candidatos mostrados.

    @return Secuencia original junto con los tokens generados.
    """
    print(f"\nContexto inicial: {decode(idx[0].tolist())!r}\n")

    for paso in range(max_new_tokens):
        # recortar el contexto al tamano maximo que soporta el modelo (block_size)
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # --- pedirle al modelo los logits (esto es lo mismo que hace generate() internamente) ---
        with torch.no_grad():
            with ctx:
                logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        # top_k opcional para el muestreo real (igual que en el generate() original)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        # --- aqui esta la parte visual ---
        if show_probs:
            top_probs, top_ids = torch.topk(probs, k=top_k_display)
            candidatos = []
            for p, tid in zip(top_probs[0], top_ids[0]):
                caracter = decode([tid.item()])
                candidatos.append(f"{caracter!r} {p.item()*100:.1f}%")
            print(f"Paso {paso+1:3d} | candidatos: " + "  |  ".join(candidatos))

        # elegir el siguiente caracter (muestreo real, con las probs completas)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # --- mostrar el caracter elegido, para que el texto se vea "escribirse" en vivo ---
        if show_probs:
            elegido = decode([idx_next.item()])
            print(f"          -> elegido: {elegido!r}    (texto hasta ahora: {decode(idx[0].tolist())!r})\n")

    return idx


with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print("=" * 60)
            print(f"MUESTRA {k+1}/{num_samples}")
            print("=" * 60)

            y = generate_visual(
                model, x, max_new_tokens,
                temperature=temperature, top_k=top_k,
                show_probs=show_probs, top_k_display=top_k_display
            )

            print("\nTexto generado completo:")
            print(decode(y[0].tolist()))
            print('---------------')

