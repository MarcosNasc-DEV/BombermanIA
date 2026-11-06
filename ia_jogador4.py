"""
IA do Jogador 4 - Bomberman Arena  (rede neural + camada de segurança)

- Camada de segurança (regras + BFS com previsão do TEMPO das explosões): decide a fuga
  e filtra as ações perigosas. A rede NUNCA consegue escolher uma ação que a camada
  considera insegura.
- Rede neural (MLP em numpy, treinada por imitação do "professor" de regras, estilo DAgger):
  escolhe entre as ações seguras. Os pesos são salvos em rede_jogador4.npz e os exemplos
  de treino em dados_jogador4.npz, na mesma pasta deste arquivo.
- A IA nunca altera os objetos recebidos (a main detecta isso como trapaça).
"""
import os
import math
import atexit
import random
import traceback
from collections import deque

import numpy as np

DIRS = {"cima": (0, -1), "baixo": (0, 1), "esquerda": (-1, 0), "direita": (1, 0)}
PASSO = 0.12    # tempo estimado por passo (o jogo usa 0.1s, aqui com folga)
MARGEM = 0.06   # folga de segurança contra o fogo

PRESSAO = False       # (testado: piora os resultados) solta bomba quando o inimigo está perto e há fuga garantida
DIST_PRESSAO = 3

_memoria = {}   # estado por jogador: id(player) -> {"n": decisões seguidas paradas}


# ------------------------------------------------------------------------------
# Utilidades do mapa
# ------------------------------------------------------------------------------
def _andavel(mapa, x, y):
    # 0 = chão, 3 e 4 = power-ups. 1 (quebrável) e 2 (indestrutível) bloqueiam.
    return 0 <= y < len(mapa) and 0 <= x < len(mapa[0]) and mapa[y][x] in (0, 3, 4)


def _celulas_explosao(mapa, x, y, nivel):
    """Mesma regra da main: alcance 1 + (nivel-1)*2; para no bloco 2; inclui o bloco 1 e para."""
    cels = [(x, y)]
    alcance = 1 + (nivel - 1) * 2
    for dx, dy in DIRS.values():
        for i in range(1, alcance + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= ny < len(mapa) and 0 <= nx < len(mapa[0])):
                break
            if mapa[ny][nx] == 2:
                break
            cels.append((nx, ny))
            if mapa[ny][nx] == 1:
                break
    return cels


# ------------------------------------------------------------------------------
# Mapa de perigo com tempo (inclui explosões em cadeia)
# ------------------------------------------------------------------------------
def _calcular_perigo(mapa, bombas, virtual, tempo_fogo):
    """
    Retorna {celula: [(inicio, fim), ...]} com os intervalos (em segundos a partir de agora)
    em que há fogo naquela célula. 'virtual' = (x, y, nivel, tempo) simula uma bomba nova.
    """
    perigo = {}
    ativas = []  # [x, y, tempo_para_explodir, set_de_celulas]

    for b in bombas:
        if getattr(b, "explodida", False):
            fim = max(getattr(b, "tempo_fogo", 0.0), 0.0)
            for c in getattr(b, "fogo", []):
                perigo.setdefault(tuple(c), []).append((0.0, fim))
        else:
            cels = set(_celulas_explosao(mapa, b.x, b.y, b.nivel))
            ativas.append([b.x, b.y, b.tempo_explosao, cels])

    if virtual is not None:
        vx, vy, vnivel, vtempo = virtual
        ativas.append([vx, vy, vtempo, set(_celulas_explosao(mapa, vx, vy, vnivel))])

    # Explosão em cadeia: uma bomba dentro do fogo de outra explode junto com ela
    mudou = True
    while mudou:
        mudou = False
        for a in ativas:
            for o in ativas:
                if o is not a and (a[0], a[1]) in o[3] and o[2] < a[2]:
                    a[2] = o[2]
                    mudou = True

    for _, _, t, cels in ativas:
        for c in cels:
            perigo.setdefault(c, []).append((t, t + tempo_fogo))
    return perigo


def _seguro_em(perigo, cel, chegada, passo=PASSO, margem=MARGEM):
    """True se ficar em 'cel' durante [chegada, chegada+passo] não cruza com fogo."""
    for ini, fim in perigo.get(cel, ()):
        if chegada - margem < fim and chegada + passo + margem > ini:
            return False
    return True


def _livre_apos(perigo, cel, chegada, margem=MARGEM):
    """True se, chegando em 'cel' no tempo 'chegada', nunca mais haverá fogo ali."""
    return all(fim < chegada - margem for _, fim in perigo.get(cel, ()))


# ------------------------------------------------------------------------------
# BFS: acha o caminho mais curto cujos passos são todos seguros no tempo certo
# ------------------------------------------------------------------------------
def _buscar(mapa, inicio, bloqueados, perigo, aceita, max_passos=30, t0=0.0, passo=PASSO, margem=MARGEM):
    """Retorna (primeira_acao, destino, passos) ou None. aceita(celula, tempo_chegada)."""
    fila = deque([(inicio, 0, None)])
    visto = {inicio}
    while fila:
        cel, n, primeira = fila.popleft()
        if n >= 1 and aceita(cel, t0 + (n - 1) * passo):
            return primeira, cel, n
        if n >= max_passos:
            continue
        for nome, (dx, dy) in DIRS.items():
            viz = (cel[0] + dx, cel[1] + dy)
            if viz in visto or viz in bloqueados or not _andavel(mapa, viz[0], viz[1]):
                continue
            if not _seguro_em(perigo, viz, t0 + n * passo, passo, margem):
                continue
            visto.add(viz)
            fila.append((viz, n + 1, primeira or nome))
    return None


def _preso(mapa, bombas, bloqueados, cel, nivel, t_bomba, t_fogo, inimigo):
    """Se eu soltar uma bomba em 'cel', o inimigo NÃO consegue escapar a tempo? (supõe inimigo rápido)"""
    pv = _calcular_perigo(mapa, bombas, (cel[0], cel[1], nivel, t_bomba), t_fogo)
    if _livre_apos(pv, inimigo, 0.0, 0.0):
        return False                                   # nem está no alcance
    return _buscar(mapa, inimigo, bloqueados | {cel}, pv,
                   lambda c, t: _livre_apos(pv, c, t, 0.0),
                   max_passos=30, passo=0.10, margem=0.0) is None


def _bomba_util(mapa, cel, nivel, inimigos):
    """A bomba nessa célula atingiria um bloco quebrável ou um inimigo?"""
    for c in _celulas_explosao(mapa, cel[0], cel[1], nivel)[1:]:
        if mapa[c[1]][c[0]] == 1 or c in inimigos:
            return True
    return False


# ------------------------------------------------------------------------------
# Decisão
# ------------------------------------------------------------------------------
def _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state):
    eu = (self_state["grid_x"], self_state["grid_y"])
    nivel = self_state["bomba_nivel"]
    t_fogo = hud_info.get("tempo_fogo", 0.5)
    t_bomba = hud_info.get("tempo_explosao", 4)

    bloqueados = {(b.x, b.y) for b in bombas if not b.explodida}   # bomba bloqueia passagem
    perigo = _calcular_perigo(mapa, bombas, None, t_fogo)
    inimigos = {(p.grid_x, p.grid_y) for p in players if p is not player and p.ativo}
    est = _memoria.setdefault(id(player), {"n": 0})

    # 1) FUGA: se a minha casa vai pegar fogo, fugir tem prioridade total
    if not _livre_apos(perigo, eu, 0.0):
        r = _buscar(mapa, eu, bloqueados, perigo, lambda c, t: _livre_apos(perigo, c, t))
        if r:
            return r[0]
        # sem rota completa: dá o passo que adia mais o fogo
        melhor, melhor_t = None, -1
        for nome, (dx, dy) in DIRS.items():
            viz = (eu[0] + dx, eu[1] + dy)
            if not _andavel(mapa, viz[0], viz[1]) or viz in bloqueados:
                continue
            if not _seguro_em(perigo, viz, 0.0):
                continue
            prox = min((ini for ini, _ in perigo.get(viz, ())), default=99)
            if prox > melhor_t:
                melhor, melhor_t = nome, prox
        return melhor or "parado"

    pode_bomba = self_state["bombas_ativas"] < self_state["max_bombas"] and eu not in bloqueados

    def escape_ok(cel):
        # simula soltar a bomba em 'cel' e verifica se existe rota de fuga a tempo
        pv = _calcular_perigo(mapa, bombas, (cel[0], cel[1], nivel, t_bomba), t_fogo)
        # só aceita rotas curtas (até ~1.7s): sobra tempo para replanejar se algo mudar
        return _buscar(mapa, cel, bloqueados, pv, lambda c, t: _livre_apos(pv, c, t),
                       max_passos=14) is not None

    # 2) Soltar bomba aqui (só se for útil E houver rota de fuga garantida)
    if pode_bomba and _bomba_util(mapa, eu, nivel, inimigos) and escape_ok(eu):
        est["n"] = 0
        return "bomba"

    # 2a) Pressão: inimigo bem perto e eu tenho fuga garantida -> solta bomba para encurralá-lo
    if PRESSAO and pode_bomba and inimigos:
        dmin = min(abs(ex - eu[0]) + abs(ey - eu[1]) for ex, ey in inimigos)
        if dmin <= DIST_PRESSAO and escape_ok(eu):
            est["n"] = 0
            return "bomba"

    # 2b) Caçar: ir até uma casa de onde a MINHA bomba prende um inimigo (sem eu me arriscar)
    if inimigos and pode_bomba:
        linhas = {e: set(_celulas_explosao(mapa, e[0], e[1], nivel)) for e in inimigos}
        alvos = set().union(*linhas.values())

        def meta_cacar(c, t):
            if c not in alvos or not _livre_apos(perigo, c, t):
                return False
            if not any(_preso(mapa, bombas, bloqueados, c, nivel, t_bomba, t_fogo, e)
                       for e in inimigos if c in linhas[e]):
                return False
            return escape_ok(c)
        r = _buscar(mapa, eu, bloqueados, perigo, meta_cacar, max_passos=12)
        if r:
            est["n"] = 0
            return r[0]

    # 3) Pegar power-up próximo
    r = _buscar(mapa, eu, bloqueados, perigo,
                lambda c, t: mapa[c[1]][c[0]] in (3, 4) and _livre_apos(perigo, c, t),
                max_passos=14)
    if r:
        est["n"] = 0
        return r[0]

    # 4) Ir até a casa mais próxima onde uma bomba seria útil e segura
    def meta(c, t):
        return (_livre_apos(perigo, c, t)
                and _bomba_util(mapa, c, nivel, inimigos)
                and escape_ok(c))
    r = _buscar(mapa, eu, bloqueados, perigo, meta)
    if r:
        est["n"] = 0
        return r[0]

    # 5) Perseguir o inimigo mais próximo
    if inimigos:
        r = _buscar(mapa, eu, bloqueados, perigo, lambda c, t: c in inimigos)
        if r:
            est["n"] = 0
            return r[0]

    # 6) Nada a fazer: espera, mas não fica parada para sempre
    est["n"] += 1
    if est["n"] > 20:
        opcoes = []
        for nome, (dx, dy) in DIRS.items():
            viz = (eu[0] + dx, eu[1] + dy)
            if (_andavel(mapa, viz[0], viz[1]) and viz not in bloqueados
                    and _livre_apos(perigo, viz, 0.0)):
                opcoes.append(nome)
        if opcoes:
            est["n"] = 0
            return random.choice(opcoes)
    return "parado"



# ==============================================================================
# REDE NEURAL (numpy) - 5 ações: 0 cima, 1 baixo, 2 esquerda, 3 direita, 4 bomba
# ==============================================================================
_DIR = os.path.dirname(os.path.abspath(__file__))
ARQ_REDE = os.path.join(_DIR, "rede_jogador4.npz")
ARQ_DADOS = os.path.join(_DIR, "dados_jogador4.npz")
ARQ_PTH = os.path.join(_DIR, "rede_jogador4.pth")   # cópia dos pesos em formato torch (se o torch existir)

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba"]
N_IN, N_H, N_OUT = 103, 64, 5
LR = 1e-3
BATCH = 64
CAP_DADOS = 30000

USAR_REDE = True         # False = joga só com as regras
TREINAR = True           # True = a rede continua aprendendo (consulta o professor a cada decisão
                         #        e salva pesos/dados ao fechar o jogo). False = só joga.
EPS_TREINO = 0.10        # chance de explorar uma ação segura aleatória enquanto treina


class Rede:
    NOMES = ("W1", "b1", "W2", "b2", "W3", "b3")

    def __init__(self):
        rng = np.random.default_rng()
        self.p = {
            "W1": (rng.standard_normal((N_IN, N_H)) * math.sqrt(2.0 / N_IN)).astype(np.float32),
            "b1": np.zeros(N_H, np.float32),
            "W2": (rng.standard_normal((N_H, N_H)) * math.sqrt(2.0 / N_H)).astype(np.float32),
            "b2": np.zeros(N_H, np.float32),
            "W3": (rng.standard_normal((N_H, N_OUT)) * 0.1).astype(np.float32),
            "b3": np.zeros(N_OUT, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def frente(self, x):
        z1 = x @ self.p["W1"] + self.p["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.p["W2"] + self.p["b2"]
        h2 = np.maximum(z2, 0)
        return z1, h1, z2, h2, h2 @ self.p["W3"] + self.p["b3"]

    def prever(self, x):
        return self.frente(x)[4]

    def adam(self, g):
        self.t += 1
        for k, gr in g.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= (LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


class Dados:
    """Exemplos (estado, ação do professor) para treinar a rede."""

    def __init__(self):
        self.S = np.zeros((CAP_DADOS, N_IN), np.float32)
        self.L = np.zeros(CAP_DADOS, np.int64)
        self.n = 0
        self.ptr = 0

    def add(self, s, label):
        self.S[self.ptr], self.L[self.ptr] = s, label
        self.ptr = (self.ptr + 1) % CAP_DADOS
        self.n = min(self.n + 1, CAP_DADOS)

    def amostra(self, k):
        i = np.random.randint(0, self.n, k)
        return self.S[i], self.L[i]

    def salvar(self, caminho):
        tmp = caminho + ".tmp.npz"
        np.savez(tmp, S=self.S[:self.n], L=self.L[:self.n], ptr=self.ptr)
        os.replace(tmp, caminho)

    def carregar(self, caminho):
        d = np.load(caminho)
        if d["S"].ndim != 2 or d["S"].shape[1] != N_IN:
            return False
        n = min(len(d["S"]), CAP_DADOS)
        self.S[:n], self.L[:n] = d["S"][:n], d["L"][:n]
        self.n, self.ptr = n, int(d["ptr"]) % CAP_DADOS
        return True


_rede = Rede()
_dados = Dados()
_passos = 0          # total de exemplos de treino já vistos (guardado junto com os pesos)
_partidas = 0        # quantas partidas a rede já treinou (guardado junto com os pesos)
_novos = 0
_stats = {"rede": 0, "seguranca": 0}   # quem decidiu: a rede ou a camada de segurança (fuga)


def _carregar():
    global _passos, _partidas, _rede
    try:
        if os.path.exists(ARQ_REDE):
            d = np.load(ARQ_REDE)
            if d["W1"].shape == (N_IN, N_H) and d["W3"].shape == (N_H, N_OUT):
                for k in Rede.NOMES:
                    _rede.p[k] = d[k].astype(np.float32)
                _passos = int(d["passos"])
                _partidas = int(d["partidas"]) if "partidas" in d.files else 0
                print(f"[IA Jogador 4] REDE NEURAL CARREGADA de {ARQ_REDE} "
                      f"({_partidas} partidas, {_passos} exemplos de treino).")
            else:
                print("[IA Jogador 4] Arquivo de rede incompatível: começando uma rede nova.")
        else:
            print("[IA Jogador 4] Nenhuma rede salva: começando uma rede nova.")
        if os.path.exists(ARQ_DADOS):
            if _dados.carregar(ARQ_DADOS):
                print(f"[IA Jogador 4] Dados de treino carregados ({_dados.n} exemplos).")
    except Exception as e:
        print(f"[IA Jogador 4] Não consegui carregar a rede ({e}): começando uma rede nova.")
        _rede = Rede()


def _salvar(com_dados=True):
    try:
        tmp = ARQ_REDE + ".tmp.npz"
        np.savez(tmp, passos=_passos, partidas=_partidas, **_rede.p)
        os.replace(tmp, ARQ_REDE)
        msg = (f"[IA Jogador 4] REDE NEURAL SALVA em {ARQ_REDE} ({os.path.getsize(ARQ_REDE)} bytes) | "
               f"partidas treinadas: {_partidas} | exemplos: {_passos} | "
               f"decisões da rede: {_stats['rede']} | fugas pela camada de segurança: {_stats['seguranca']}")
        print(msg)
        if com_dados and _dados.n > 0:
            _dados.salvar(ARQ_DADOS)
        try:                                   # cópia em formato torch, só se o torch estiver instalado
            import torch
            torch.save({k: torch.from_numpy(v.copy()) for k, v in _rede.p.items()}, ARQ_PTH)
            print(f"[IA Jogador 4] Cópia em formato torch salva em {ARQ_PTH}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[IA Jogador 4] Não consegui salvar o .pth: {e}")
    except Exception as e:
        print(f"[IA Jogador 4] ERRO ao salvar a rede: {e}")


def _treinar_um_lote():
    """Entropia cruzada entre a saída da rede (só ações seguras) e a ação do professor."""
    S, L = _dados.amostra(BATCH)
    B = len(S)
    z1, h1, z2, h2, q = _rede.frente(S)
    mask = (S[:, -5:] > 0)
    lg = np.where(mask, q, -1e9)
    lg = lg - lg.max(axis=1, keepdims=True)
    e = np.exp(lg) * mask
    p = e / e.sum(axis=1, keepdims=True)
    dq = p.copy()
    dq[np.arange(B), L] -= 1.0
    dq = (dq * mask / B).astype(np.float32)
    g = {"W3": h2.T @ dq, "b3": dq.sum(axis=0)}
    dz2 = (dq @ _rede.p["W3"].T) * (z2 > 0)
    g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(axis=0)
    dz1 = (dz2 @ _rede.p["W2"].T) * (z1 > 0)
    g["W1"], g["b1"] = S.T @ dz1, dz1.sum(axis=0)
    _rede.adam(g)


def _fim_de_jogo(salvar=True):
    global _partidas
    if TREINAR:
        _partidas += 1
        if salvar:
            _salvar(True)


atexit.register(_fim_de_jogo)


# ------------------------------------------------------------------------------
# Entradas da rede
# ------------------------------------------------------------------------------
def _escape_ok(mapa, cel, nivel, bombas, bloqueados, t_bomba, t_fogo):
    pv = _calcular_perigo(mapa, bombas, (cel[0], cel[1], nivel, t_bomba), t_fogo)
    return _buscar(mapa, cel, bloqueados, pv, lambda c, t: _livre_apos(pv, c, t),
                   max_passos=14) is not None


def _distancias(mapa, eu, nivel, bloqueados, perigo, inimigos, bombas, t_bomba, t_fogo):
    """
    Uma BFS só. Para cada primeira ação (cima/baixo/esq/dir) e cada tipo de alvo
    (power-up, casa boa para bomba, inimigo, casa que PRENDE um inimigo), retorna
    1/distância (0 = inalcançável).
    """
    saida = np.zeros((4, 4), np.float32)
    linhas = {e: set(_celulas_explosao(mapa, e[0], e[1], nivel)) for e in inimigos}
    alvos = set().union(*linhas.values()) if linhas else set()
    avaliados = 0
    fila = deque()
    visto = {eu}
    for i, (dx, dy) in enumerate(DIRS.values()):
        viz = (eu[0] + dx, eu[1] + dy)
        if (_andavel(mapa, viz[0], viz[1]) and viz not in bloqueados
                and viz not in visto and _seguro_em(perigo, viz, 0.0)):
            visto.add(viz)
            fila.append((viz, 1, i))
    while fila:
        cel, n, i = fila.popleft()
        if mapa[cel[1]][cel[0]] in (3, 4) and _livre_apos(perigo, cel, (n - 1) * PASSO):
            saida[0, i] = max(saida[0, i], 1.0 / n)
        if _livre_apos(perigo, cel, (n - 1) * PASSO) and _bomba_util(mapa, cel, nivel, inimigos):
            saida[1, i] = max(saida[1, i], 1.0 / n)
        if cel in inimigos:
            saida[2, i] = max(saida[2, i], 1.0 / n)
        if cel in alvos and avaliados < 12 and _livre_apos(perigo, cel, (n - 1) * PASSO):
            avaliados += 1
            if any(_preso(mapa, bombas, bloqueados, cel, nivel, t_bomba, t_fogo, e)
                   for e in inimigos if cel in linhas[e]):
                saida[3, i] = max(saida[3, i], 1.0 / n)
        if n >= 30:
            continue
        for dx, dy in DIRS.values():
            viz = (cel[0] + dx, cel[1] + dy)
            if (viz in visto or viz in bloqueados or not _andavel(mapa, viz[0], viz[1])
                    or not _seguro_em(perigo, viz, n * PASSO)):
                continue
            visto.add(viz)
            fila.append((viz, n + 1, i))
    return saida.reshape(-1)


def _features(mapa, eu, nivel, perigo, bloqueados, inimigos, pode_bomba, util, esc_ok, mask,
              bombas, t_bomba, t_fogo):
    rows, cols = len(mapa), len(mapa[0])
    jan = [(eu[0] + dx, eu[1] + dy) for dy in range(-2, 3) for dx in range(-2, 3)]
    f = []
    for x, y in jan:                                          # 25: tipo de cada casa
        f.append(mapa[y][x] / 4.0 if 0 <= x < cols and 0 <= y < rows else 0.5)
    for x, y in jan:                                          # 25: fogo previsto
        v = 0.0
        iv = perigo.get((x, y))
        if iv:
            v = 1.0 if min(i for i, _ in iv) <= 1.5 else 0.5
        f.append(v)
    for c in jan:                                             # 25: bombas
        f.append(1.0 if c in bloqueados else 0.0)
    f += [eu[0] / cols, eu[1] / rows, float(pode_bomba), float(util), float(esc_ok)]   # 5
    if inimigos:
        ex, ey = min(inimigos, key=lambda c: abs(c[0] - eu[0]) + abs(c[1] - eu[1]))
        f += [(ex - eu[0]) / cols, (ey - eu[1]) / rows]                                  # 2
    else:
        f += [0.0, 0.0]
    f += list(_distancias(mapa, eu, nivel, bloqueados, perigo, inimigos, bombas, t_bomba, t_fogo))  # 16
    f += [float(m) for m in mask]                                                        # 5: ações seguras
    return np.array(f, dtype=np.float32)


def _decidir_rede(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state):
    global _passos, _novos
    eu = (self_state["grid_x"], self_state["grid_y"])
    nivel = self_state["bomba_nivel"]
    t_fogo = hud_info.get("tempo_fogo", 0.5)
    t_bomba = hud_info.get("tempo_explosao", 4)

    bloqueados = {(b.x, b.y) for b in bombas if not b.explodida}
    perigo = _calcular_perigo(mapa, bombas, None, t_fogo)
    inimigos = {(p.grid_x, p.grid_y) for p in players if p is not player and p.ativo}

    # Em perigo, a camada de segurança foge sozinha (a rede não tem permissão para arriscar)
    if not _livre_apos(perigo, eu, 0.0):
        _stats["seguranca"] += 1
        return _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)

    _stats["rede"] += 1
    pode_bomba = self_state["bombas_ativas"] < self_state["max_bombas"] and eu not in bloqueados
    util = _bomba_util(mapa, eu, nivel, inimigos)
    esc_ok = (pode_bomba and util
              and _escape_ok(mapa, eu, nivel, bombas, bloqueados, t_bomba, t_fogo))

    # ações que a camada de segurança aprova
    mask = [0, 0, 0, 0, int(bool(esc_ok))]
    for i, (dx, dy) in enumerate(DIRS.values()):
        viz = (eu[0] + dx, eu[1] + dy)
        if not _andavel(mapa, viz[0], viz[1]) or viz in bloqueados or not _seguro_em(perigo, viz, 0.0):
            continue
        # casa sem fogo previsto, OU casa de onde ainda dá para sair a tempo
        if _livre_apos(perigo, viz, 0.0) or _buscar(
                mapa, viz, bloqueados, perigo, lambda c, t: _livre_apos(perigo, c, t),
                max_passos=14, t0=PASSO) is not None:
            mask[i] = 1
    validas = [i for i in range(5) if mask[i]]
    if not validas:
        return "parado"

    s = _features(mapa, eu, nivel, perigo, bloqueados, inimigos, pode_bomba, util, esc_ok, mask,
                  bombas, t_bomba, t_fogo)
    q = np.where(np.array(mask) > 0, _rede.prever(s), -1e9)
    acao_idx = int(np.argmax(q))

    if TREINAR:
        # professor: o que as regras fariam neste estado?
        prof = _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
        if prof in ACOES and mask[ACOES.index(prof)]:
            _dados.add(s, ACOES.index(prof))
            _passos += 1
            _novos += 1
            if _dados.n >= 500 and _novos % 4 == 0:
                _treinar_um_lote()
            if _passos % 5000 == 0:
                _salvar(False)
        if random.random() < EPS_TREINO:
            acao_idx = random.choice(validas)               # exploração segura (DAgger)
    return ACOES[acao_idx]


def decidir_acao(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state):
    try:
        if USAR_REDE:
            return _decidir_rede(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
        return _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
    except Exception:
        traceback.print_exc()   # mostra o erro em vez de engolir em silêncio
        try:
            return _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
        except Exception:
            return "parado"


_carregar()