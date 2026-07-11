# Losses de treinamento — guia de leitura

Referência: `training/losses.py` (implementação) e `training/lightning_module.py`
(`_step`, onde as losses são combinadas e logadas). Segue a receita de
arXiv:2509.06926, Apêndice A (Lagrangian Self-Distillation).

## As três losses

### 1. Flow matching (`loss_fm` / `mse_fm`)

Treina a cabeça de fluxo `flow_net` (`SimpleMLPAdaLN`) a prever a velocidade
que leva do ruído (`t=0`) ao latent alvo (`t=1`), na diagonal `s == t`:

```
x_t = (1 - t) * eps + t * x1        # interpolação linear ruído -> dado
v_target = x1 - eps                  # velocidade constante ao longo do caminho
pred = flow_net(cond, t, t, x_t)
mse_fm = mean((pred - v_target)^2)
```

É a loss principal — a que ensina o modelo a reconstruir o timbre/prosódia
correto a partir do contexto (`cond`, saída do backbone transformer).
Aplicada a 75% das amostras do "head batch" (`fm_fraction`, ver abaixo).

### 2. LSD — Lagrangian Self-Distillation (`loss_lsd` / `mse_lsd`)

Aplicada aos 25% restantes. Em vez de treinar só na diagonal, força
consistência do *mapa de fluxo* entre dois instantes `a -> b` quaisquer:

```
x_a = (1 - a) * eps + a * x1
f(x_a, a->b) = x_a + (b - a) * flow_net(cond, a, b, x_a)   # flow map
df/db  via torch.func.jvp (derivada direcional, forward-mode AD)
teacher = flow_net(cond, b, b, f(x_a, a->b))                # stop-gradient
mse_lsd = mean((df/db - teacher)^2)
```

É o que permite decodificar em **poucos passos** na inferência (o modelo
público usa `lsd_decode_steps=1`) sem perder qualidade — sem essa loss, o
flow matching sozinho exigiria dezenas de passos de integração numérica.
Mais cara de calcular (usa `jvp`), por isso só 25% do batch a recebe.

### 3. EOS (`loss_eos`)

BCE mascarado e ponderado (`pos_weight`) na cabeça `out_eos`, que prevê se
cada frame de 80 ms (12,5 Hz) é o último frame de fala:

```
loss_eos = BCEWithLogits(logits, labels, pos_weight) mascarado por frame válido
```

Só entra na loss total se `eos_weight > 0` (ver seção "EOS" abaixo) —
é a única das três que pode ficar completamente fora do treino.

### Combinação final

```python
loss = loss_fm + loss_lsd
if eos_weight > 0:
    loss = loss + eos_weight * loss_eos
```

`loss_fm` e `loss_lsd` sempre entram com peso 1 (a paridade entre elas é
controlada por `fm_fraction`, não por um multiplicador extra).

## Ponderação adaptativa `w_psi(s, t)` — por que `loss_fm` ≠ `mse_fm`

Cada loss (FM e LSD) tem sua própria rede pequena `AdaptiveWeight` (MLP sobre
features de Fourier de `(s, t)`) que aprende a reponderar o erro por
dificuldade do par de tempos:

```
loss_ponderada = exp(-w(s,t)) * erro_por_amostra + w(s,t)
```

Isso é o valor logado como `loss_fm`/`loss_lsd`. `w` começa em zero
(última camada do MLP inicializada em zero), então no início do treino
`loss_fm ≈ mse_fm`. Conforme `w` aprende, os dois **divergem**:

- Se `w` fica negativo, `exp(-w) > 1` — o erro é amplificado (região que a
  rede considera "difícil e informativa").
- Se `w` fica positivo, `exp(-w) < 1` — o erro é atenuado.

**Armadilha importante:** o mínimo teórico de `exp(-w)*erro + w` em função
de `w` é `1 + ln(erro)` — ou seja, **`loss_fm`/`loss_lsd` podem continuar
caindo só porque `w` está se ajustando ao erro, mesmo que o erro real
(`mse_fm`/`mse_lsd`) tenha parado de melhorar ou esteja piorando.** Foi
exatamente esse efeito que mascarou uma divergência em um run real (ver
"Casos observados" abaixo): `loss_total` caía suavemente enquanto o modelo
colapsava.

Por isso `_step` loga separadamente:
- `mse_fm` / `mse_lsd` — erro cru, não ponderado. **É o que decide se o
  modelo está melhorando de verdade.**
- `w_fm` / `w_lsd` — média de `w` no batch. Útil para saber se a queda de
  `loss_fm` é erro caindo ou `w` migrando.

`w` é *clampado* em `[-4, 4]` (`_weighted` em `losses.py`) — perto da
convergência, `w` tende a rastrear `ln(erro)`, e sem o clamp um batch difícil
tardio no treino teria seu gradiente amplificado por `exp(-w) ≈ 1/erro`,
o que gerou uma espiral de instabilidade num run observado. Se `w_fm`/`w_lsd`
encostam no limite (±4) de forma persistente, é sinal de que a ponderação
está saturando — considere baixar o LR ou investigar outliers no batch.

Desligar (`adaptive_weighting: false`) faz `loss_fm == mse_fm` e
`loss_lsd == mse_lsd` (MSE simples), mas perde a reponderação por
dificuldade que o paper usa.

## EOS — calibração e as métricas de diagnóstico

A cabeça de EOS tem uma particularidade: ela é usada na **inferência** com
um threshold fixo (`DEFAULT_EOS_THRESHOLD = -4.0` no CLI padrão), então
"a loss está baixa" não implica "a calibração está certa nesse threshold".
Por isso `_step` também loga, calculados sem gradiente:

- **`eos_logit_true`** — logit médio da cabeça **exatamente no frame
  verdadeiro de fim de fala**. Precisa ficar **acima** do threshold de
  deploy para a geração parar no lugar certo.
  - Muito abaixo do threshold → o modelo exportado não vai saber parar
    (fala arrastada / continua até o limite de frames).
  - Bem acima (positivo, ex. +4) → parada segura, com folga.

- **`eos_false_trigger`** — fração de frames **no meio da fala** cujo logit
  já ultrapassa o threshold de deploy. É o risco oposto: parar cedo demais.
  Idealmente perto de zero.

Três regimes de `eos_weight`:

| `eos_weight` | Comportamento |
|---|---|
| `0.0` | Cabeça **congelada de verdade** (`requires_grad_(False)` em `configure_model` — só zerar o peso da loss não bastava, o weight decay do AdamW ainda corroía os pesos). Preserva a calibração pré-treinada, mas o *backbone* muda com o fine-tuning e a representação que chega na cabeça desloca — em um run observado, `eos_logit_true` caiu de ~−1 para −4,3 (abaixo do threshold) em ~650 steps. Use só em runs curtos ou quando não há orçamento para monitorar. |
| pequeno (ex. `0.05`) | A cabeça acompanha a deriva do backbone sem dominar a loss total. Recomendado para fine-tunes de mais de ~500 steps. **Não restaurar** os pesos pré-treinados de `out_eos.*` no export quando usar isso — a cabeça treinada é a correta. |
| alto (`>= 1.0`, com `eos_pos_weight` também alto) | Regime do paper para treino do zero. Em fine-tuning agressivo (`pos_weight` alto) já causou o modelo exportado cortar a fala cedo demais — evite a menos que esteja monitorando `eos_false_trigger` de perto. |

Ao ler os áudios de validação no TensorBoard (`_synthesize_and_log`), a
síntese usa o mesmo threshold de deploy (`DEFAULT_EOS_THRESHOLD`) de
propósito — usar um threshold mais frouxo esconderia problemas de
truncamento que o modelo exportado teria de verdade.

## Como ler os gráficos no TensorBoard

Por prioridade de diagnóstico:

1. **`val/mse_fm`** — a métrica que importa para decidir "o modelo está
   melhorando?". É o monitor do `ModelCheckpoint` e do `EarlyStopping` nas
   configs (não `val/loss_total` — esse é dominado pelo termo `w` e pode
   cair mesmo com o modelo piorando).
2. **`train/grad_norm`** — norma L2 do gradiente antes do clip
   (`on_before_optimizer_step`). Estável (ex. 1,5–2) é saudável. Picos
   grandes e crescentes precedem divergência — é o alerta mais precoce.
3. **`*/eos_logit_true`** vs. o threshold de deploy (−4.0 por padrão) —
   decide se o modelo exportado vai saber parar de falar.
4. **`*/w_fm`, `*/w_lsd`** — só para checar se a ponderação adaptativa está
   saturando (perto de ±4) ou deriva excessivamente.
5. **`*/loss_fm`, `*/loss_lsd`, `*/loss_total`** — tendência geral, mas
   nunca confiar neles isoladamente para decidir se o treino está indo bem;
   sempre cruzar com `mse_fm`/`mse_lsd`.
6. **`*/eos_false_trigger`** — sobe = risco de corte prematuro em fala
   longa; olhar junto com os áudios de `val_samples`.

Gap `train/mse_fm` vs `val/mse_fm` crescendo com o tempo = overfitting
(esperado em fine-tunes de poucas horas de áudio de um único locutor).

## Parâmetros que afetam o resultado

Todos em `PocketTTSTraining.__init__` (`training/lightning_module.py`),
setados via o bloco `model:` do config YAML.

| Parâmetro | Efeito |
|---|---|
| `fm_fraction` (default 0.75) | Fração do "head batch" que recebe FM em vez de LSD. Mais FM = mais fiel ao alvo diagonal; mais LSD = melhor decodificação em poucos passos. |
| `head_batch_multiplier` (default 8) | Cada saída do backbone é reutilizada para várias amostras independentes de `(t, eps)` na cabeça de fluxo — mais sinal de treino por passo do transformer, sem custo adicional de atenção. |
| `adaptive_weighting` (default `true`) | Liga/desliga o `w_psi(s,t)` aprendido descrito acima. |
| `eos_weight` / `eos_pos_weight` | Ver seção EOS. |
| `text_dropout` (default 0.15) | Fração de amostras de treino que têm o texto zerado — necessário para o *classifier-free guidance* (CFG) funcionar na inferência. Não afeta a loss diretamente, mas afeta a distribuição de `cond` que `flow_fm`/`loss_lsd` veem. |
| `voice_prompt_prob`, `voice_prompt_min/max_frames` | Controlam com que frequência e tamanho o modelo recebe um recorte de voz — de uma *outra* utterance do mesmo locutor no batch (`_sample_voice_prompt`), nunca da própria utterance-alvo. Requer `speaker_id` por amostra; sem par do mesmo locutor no batch (comum no TAGARELA, que não tem esse campo), a amostra treina sem prompt. |
| `lr`, `warmup_steps`, `lr_min_ratio` | Schedule cosine com warmup linear (`configure_optimizers`). LR alto demais é a causa mais comum de `grad_norm` instável e `mse_fm`/`mse_lsd` divergindo tarde no treino. |
| `weight_decay`, `betas` | AdamW padrão. `weight_decay` é a razão de a cabeça de EOS precisar ser explicitamente congelada (`requires_grad_(False)`) quando `eos_weight == 0` — só zerar o peso da loss não impede o decay de corroer os pesos. |
| `gradient_clip_val` (config do trainer, não do módulo) | Se `train/grad_norm` fica consistentemente bem acima desse valor, todo update está sendo cortado — considere baixar o LR em vez de confiar só no clip. |
| `pretrained` | Se `false`, reinicializa o FlowLM (mantendo Mimi e `emb_mean`/`emb_std`) — treino do zero, receita do paper (8×H100, 400k steps), inviável em uma GPU única. |

## Casos observados (histórico, para contexto)

- **Divergência mascarada pela ponderação adaptativa**: um run com batch
  efetivo 480 e `max_steps: 1_000_000` (cosine efetivamente não decaindo)
  divergiu por volta do step ~2.200 — `mse_fm` saltou de ~0,03 para
  dezenas/centenas — mas `loss_total` (ponderado) continuou parecendo cair
  em alguns pontos porque `w` compensava. O `ModelCheckpoint` antigo
  (monitorando `loss_total`) chegou a salvar como "melhor" um checkpoint de
  dentro da região divergida. Motivou: logar `mse_fm`/`mse_lsd` cru, mudar o
  monitor para `mse_fm`, adicionar `EarlyStopping`, clampar `w`, e logar
  `grad_norm`.
- **Deriva do EOS com a cabeça congelada**: com `eos_weight: 0.0`,
  `eos_logit_true` caiu monotonicamente de ~−1,0 para −4,3 em ~650 steps,
  cruzando o threshold de deploy — o modelo exportado teria parado de
  funcionar (nunca disparando EOS). Motivou o uso de `eos_weight` pequeno
  (ex. 0.05) em vez de congelamento total para fine-tunes acima de algumas
  centenas de steps.
- **Self-prompt corrompendo o alinhamento texto-áudio**: `_split_prompt`
  recortava o voice prompt da própria utterance-alvo, mas o texto continuava
  sendo a transcrição inteira — o modelo aprendia que um prefixo do texto já
  estava "coberto" pelo prompt, algo que não existe na inferência real
  (prompt é áudio não relacionado). Um checkpoint treinado assim (LJSpeech,
  ~1000 steps) sintetizou áudio ininteligível e **instável entre contagens de
  decode steps** (duração 0,32–1,68 s e RMS 0,0008–0,12 para o mesmo
  texto/voz, contra 2,96–3,28 s e RMS 0,08–0,10 estáveis no modelo base sem
  fine-tuning) — sinal de que a autoconsistência da LSD havia sido quebrada.
  Motivou trocar para `_sample_voice_prompt`: recorte de uma *outra*
  utterance do mesmo locutor no batch, mantendo sempre o alvo completo.
