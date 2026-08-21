# train a miniature character-level quijote model
# good for debugging and playing on macbooks and such

"""
# Configuración utilizada para ejecutar el modelo en su versión más simple.

**Autor:** Andrej Karpathy
**Documentación en español:** *Josué Aybar Paulino*

# Descripción

 Esta es la configuración de base que se utiliza para ejecutar el dataset quijote_char.py con el propósito
de mostrar la generación de texto de este modelo. Existen otras configuraciones donde el bloque de transformers
es de 12 y las cabezas de atención por igual, con una dimensión mayor; pero ello es para entrenar el modelo
usando GPT2 u OpenWebtext, lo que conlleva más recursos y tiempo.
"""
out_dir = 'out-quijote-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'quijote-char'
wandb_run_name = 'mini-gpt'

dataset = 'quijote_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

compile = False # torch.compile can be problematic on Windows

# on macbook also add
# device = 'cpu'  # run on cpu only
