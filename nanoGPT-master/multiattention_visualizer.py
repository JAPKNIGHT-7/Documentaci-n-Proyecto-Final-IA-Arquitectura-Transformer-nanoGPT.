"""
Visualizador de las 6 cabezas de auto-atencion causal de nanoGPT (shakespeare_char / Quijote).
No modifica model.py ni sample.py. Usa un forward_hook (no toca el comportamiento del
modelo) para capturar la entrada de la capa de atencion, recalcula la matriz de atencion
completa por fuera, y guarda un GIF con las 6 cabezas actualizandose token por token.
"""
import os, pickle, torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import GPTConfig, GPT

# ===== CONFIGURACION (ajustable) =====
OUT_DIR = "out-quijote-char"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START = "En un lugar de la Mancha"
NEW_TOKENS = 30
LAYER = 0          # 0 a n_layer-1
# ========================================================

## cargar el checkpoint entrenado
ckpt = torch.load(os.path.join(OUT_DIR, "ckpt.pt"), map_location=DEVICE)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)

## cargar el vocabulario de caracteres
with open(f"data/{ckpt['config']['dataset']}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

## hook: captura la entrada de la capa de atencion sin alterar el modelo
capturado = {}
def hacer_hook(nombre):
    """
    @brief Crea un hook para capturar la entrada de una capa de atención.

    @author Andrej Karpathy
    **Documentación en español:** *Josué Aybar Paulino*

    @details

    El hook captura la entrada de la capa de atención durante el
    forward pass sin modificar el comportamiento del modelo.

    @param nombre Nombre utilizado para identificar la capa capturada.
    @return Función hook que almacena la entrada de la capa.
    """

    def guardar(module, entrada, salida):
        capturado[nombre] = entrada[0].detach()
    return guardar

for i, block in enumerate(model.transformer.h):
    block.attn.register_forward_hook(hacer_hook(i))

## recalcula la atencion completa (T x T) para las 6 cabezas, por fuera del modelo
def calcular_atencion(x, layer):
    """
    @brief Calcula la matriz de atención causal de una capa.

    @author Andrej Karpathy
    **Documentación en español:** *Josué Aybar Paulino*

    @details

    Recalcula externamente las puntuaciones de atención para todas
    las cabezas y aplica la máscara causal y softmax.

    @param x Entrada de la capa de atención.
    @param layer Índice de la capa Transformer.

    @return Matriz de atención de todas las cabezas en CPU.
    """
    attn = model.transformer.h[layer].attn
    B, T, C = x.shape
    q, k, _ = attn.c_attn(x).split(C, dim=2)
    hs = C // attn.n_head
    q = q.view(B, T, attn.n_head, hs).transpose(1, 2)
    k = k.view(B, T, attn.n_head, hs).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (hs ** 0.5)
    mascara = torch.tril(torch.ones(T, T, device=x.device))
    scores = scores.masked_fill(mascara == 0, float("-inf"))
    return torch.softmax(scores, dim=-1)[0].cpu()

## generar texto, guardando la atencion de cada paso (una sola llamada al modelo por paso)
idx = torch.tensor([encode(START)], dtype=torch.long, device=DEVICE)
historial_atencion, historial_texto = [], []

with torch.no_grad():
    for paso in range(NEW_TOKENS):
        logits, _perdida = model(idx)                # el hook captura la entrada aqui mismo
        historial_atencion.append(calcular_atencion(capturado[LAYER], LAYER))
        historial_texto.append(decode(idx[0].tolist()))

        probs = torch.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)

        ## imprimir logits y probabilidades top-5 de este paso (para ver el softmax en vivo)
        top_p, top_i = torch.topk(probs[0], 5)
        top_l = logits[0, -1, top_i]
        candidatos = [f"'{itos[i.item()]}'={l.item():.2f}/{p.item():.2%}" for i, l, p in zip(top_i, top_l, top_p)]
        print(f"paso {paso+1:>2} -> siguiente: '{itos[idx[0,-1].item()]}' | top5 (logit/prob): {', '.join(candidatos)}")

print("\nTexto generado:\n" + decode(idx[0].tolist()))

## animar las 6 cabezas (heatmap completo) y guardar como GIF
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

def actualizar(frame):
    """
    @brief Actualiza la visualización de las cabezas de atención.

    @author Andrej Karpathy
    **Documentación en español:** *Josué Aybar Paulino*

    @details

    Actualiza los seis mapas de calor con la matriz de atención
    correspondiente al paso de generación indicado.

    @param frame Índice del paso de generación que se está visualizando.
    """
    atencion = historial_atencion[frame]
    for h, ax in enumerate(axes.flat):
        ax.clear()
        ax.imshow(atencion[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Cabeza {h + 1}", fontsize=9)
        ax.set_xlabel("Token al que mira", fontsize=7)
        ax.set_ylabel("Token actual", fontsize=7)
    fig.suptitle(f"Capa {LAYER + 1} — Autoatencion causal\nTexto: {historial_texto[frame][-60:]}")
    fig.tight_layout()

anim = animation.FuncAnimation(fig, actualizar, frames=len(historial_atencion), interval=400)

os.makedirs("visualizaciones", exist_ok=True)
anim.save("visualizaciones/atencion_multicabeza.gif", writer="pillow", fps=2)
print("Listo. GIF guardado en visualizaciones/atencion_multicabeza.gif")