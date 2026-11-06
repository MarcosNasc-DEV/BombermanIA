"""
IA TchkBum - Bomberman Arena  (rede neural + camada de segurança + camada tática)
==================================================================================
VERSÃO UNIFICADA DO GRUPO. Em cada trecho, o comentário ">>> [NOME]" mostra de quem
é a ideia/código e o "O que faz" explica de forma simples.

  [MARCOS]       base: camada de segurança + rede neural (MLP em NumPy) + DAgger + salvar/carregar
  [HYTALO]       previsão de perigo mais esperta (bloco quebrado por uma bomba libera o fogo da próxima)
  [CARLOS]       modo "pressão": solta bomba / persegue o inimigo quando ele tem pouca chance de fugir
  [RYAN]         não ficar indo e voltando (histórico de posições) e pausa entre uma bomba e outra
  [MAICON]       ideia de soltar bomba ao lado de bloco quebrável (aqui só com fuga garantida)
  [JOSÉ PEDRO]  camada tática: modos grupo/caça/duelo, evita emboscada, posto de espera seguro,
                 armadilha garantida, não destrói power-up alcançável, +10 entradas na rede
  [NOVO]         relatório final das decisões + arquivos TchkBum (.npz sempre, .pth se tiver torch)

Como funciona (resumo):
- Camada de segurança (regras + BFS com previsão do TEMPO das explosões): decide a fuga
  e filtra as ações perigosas. A rede NUNCA consegue escolher uma ação insegura.
- Camada tática (regras): escolhe o que fazer quando está seguro (bomba, caçar, power-up, esperar).
  Funciona como o "professor" que a rede imita.
- Rede neural (MLP em numpy, treinada por imitação do "professor", estilo DAgger):
  escolhe entre as ações seguras.
- Arquivos salvos na mesma pasta deste arquivo, ao fechar o jogo:
      rede_TchkBum.npz   -> pesos da rede
      dados_TchkBum.npz  -> exemplos de treino
      rede_TchkBum.pth   -> cópia dos pesos em formato torch (só se o torch estiver instalado)
- No final do jogo é impresso um RELATÓRIO com as tomadas de decisão.
- A IA nunca altera os objetos recebidos (a main detecta isso como trapaça).
"""
import os
import math
import atexit
import random
import traceback
from collections import deque

import numpy as np

NOME = "TchkBum"   # >>> [NOVO] nome da equipe: aparece nos prints e nos nomes dos arquivos

# ==============================================================================
# CONFIGURAÇÃO (chaves liga/desliga)
# ==============================================================================
# >>> [JOSÉ PEDRO] TATICA, MODO_FORCADO e o aviso sobre TREINAR são dele.
TATICA = True            # False = comportamento antigo, sem a camada tática (só para comparar)
MODO_FORCADO = None      # None = automático. "duelo" / "grupo" / "caca" forçam o modo
USAR_REDE = True         # False = joga só com as regras
TREINAR = True           # True = a rede aprende e SALVA ao fechar o jogo (use nos treinos).
                         # No CAMPEONATO deixe False: só joga, não gasta tempo treinando.
EPS_TREINO = 0.10        # chance de explorar uma ação segura aleatória enquanto treina

# >>> [CARLOS] Configurações do modo "pressão"
# O que faz: liga/desliga o ataque ao inimigo que está com pouca chance de fugir.
#   FUGA_MIN     = se a fuga do inimigo for de 3 passos ou mais, vale soltar a bomba
#   DEST_MAX     = se o inimigo tiver 3 saídas ou menos, vale soltar a bomba
#   CACA_PRESSAO = até quantos passos eu ando para achar um bom ponto de pressão
PRESSAO = True
FUGA_MIN = 3
DEST_MAX = 3
CACA_PRESSAO = 6
# >>> [NOVO] Junção Carlos + José Pedro: a pressão só liga nos modos agressivos da camada tática.
# No modo "grupo" (3-4 vivos) vale a filosofia do José Pedro: só ataca com armadilha garantida.
PRESSAO_MODOS = ("duelo", "caca")

# >>> [RYAN] Configurações anti-vaivém e pausa entre bombas
# O que faz: COOLDOWN_BOMBA = espera (em segundos) antes de soltar outra bomba;
#   HIST_TAM = quantas posições passadas a IA lembra; LIMITE_VAIVEM = quantas vezes
#   ela pode repetir a mesma casa antes de a IA tentar outro caminho.
COOLDOWN_BOMBA = 0.6
HIST_TAM = 12
LIMITE_VAIVEM = 3

DIRS = {"cima": (0, -1), "baixo": (0, 1), "esquerda": (-1, 0), "direita": (1, 0)}
PASSO = 0.12    # tempo estimado por passo (o jogo usa 0.1s, aqui com folga)
MARGEM = 0.06   # folga de segurança contra o fogo

_memoria = {}   # estado por jogador: id(player) -> contador de parada, histórico e hora da última bomba


def _estado(player):
    """>>> [MARCOS + RYAN] Guarda a memória curta de cada jogador (contador de parada,
    últimas posições e quando soltou a última bomba)."""
    return _memoria.setdefault(id(player), {"n": 0, "hist": deque(maxlen=HIST_TAM), "bomba_t": None})


def _cooldown_ok(est, tempo_restante):
    """>>> [RYAN] O que faz: só deixa soltar nova bomba se já passou um tempinho
    (COOLDOWN_BOMBA) desde a última. Evita encher o mapa de bombas seguidas."""
    t = est.get("bomba_t")
    if t is None or not isinstance(tempo_restante, (int, float)):
        return True
    return abs(t - tempo_restante) >= COOLDOWN_BOMBA


# ------------------------------------------------------------------------------
# Utilidades do mapa
# ------------------------------------------------------------------------------
def _andavel(mapa, x, y):
    # 0 = chão, 3 e 4 = power-ups. 1 (quebrável) e 2 (indestrutível) bloqueiam.
    return 0 <= y < len(mapa) and 0 <= x < len(mapa[0]) and mapa[y][x] in (0, 3, 4)


def _celulas_explosao(mapa, x, y, nivel, quebrados=frozenset()):
    """
    >>> [HYTALO] (versão dele; a base é do MARCOS)
    O que faz: calcula quais casas a explosão atinge. Mesma regra da main (alcance
    1 + (nivel-1)*2, para no bloco 2 e no primeiro bloco 1). A novidade: 'quebrados'
    são blocos que JÁ terão sumido quando essa bomba explodir, então o fogo passa por eles.
    """
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
            if mapa[ny][nx] == 1 and (nx, ny) not in quebrados:
                break
    return cels


# ------------------------------------------------------------------------------
# Mapa de perigo com tempo (inclui explosões em cadeia)
# ------------------------------------------------------------------------------
def _calcular_perigo(mapa, bombas, virtual, tempo_fogo):
    """
    >>> [HYTALO] (versão dele; a ideia do mapa de perigo é do MARCOS)
    O que faz: devolve {casa: [(inicio, fim), ...]} = em que momentos (em segundos a partir
    de agora) haverá fogo em cada casa. 'virtual' = (x, y, nivel, tempo) simula uma bomba nova.
    A novidade: as explosões são processadas da mais cedo para a mais tarde, e se uma
    explosão quebra um bloco, a explosão seguinte já passa por ali (cadeia mais fiel ao jogo).
    """
    perigo = {}
    quebra = {}      # bloco -> instante em que ele some (fim do fogo que o atinge)
    pendentes = []   # [x, y, nivel, tempo_para_explodir]

    for b in bombas:
        if getattr(b, "explodida", False):
            fim = max(getattr(b, "tempo_fogo", 0.0), 0.0)
            for c in getattr(b, "fogo", []):
                c = tuple(c)
                perigo.setdefault(c, []).append((0.0, fim))
                if mapa[c[1]][c[0]] == 1:
                    quebra[c] = min(quebra.get(c, fim), fim)
        else:
            pendentes.append([b.x, b.y, b.nivel, b.tempo_explosao])

    if virtual is not None:
        vx, vy, vnivel, vtempo = virtual
        pendentes.append([vx, vy, vnivel, vtempo])

    while pendentes:                       # processa sempre a explosão mais cedo
        pendentes.sort(key=lambda p: p[3])
        x, y, nivel, t = pendentes.pop(0)
        quebrados = {c for c, tq in quebra.items() if tq <= t}
        cels = _celulas_explosao(mapa, x, y, nivel, quebrados)
        for c in cels:
            perigo.setdefault(c, []).append((t, t + tempo_fogo))
            if mapa[c[1]][c[0]] == 1:
                quebra[c] = min(quebra.get(c, t + tempo_fogo), t + tempo_fogo)
        atingidas = set(cels)
        for p in pendentes:                # explosão em cadeia: adota o menor tempo
            if (p[0], p[1]) in atingidas and p[3] > t:
                p[3] = t
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


# ------------------------------------------------------------------------------
# >>> [CARLOS] Pressão sobre o inimigo
# ------------------------------------------------------------------------------
def _fuga_inimigo(mapa, bombas, bloqueados, cel, nivel, t_bomba, t_fogo, inimigo):
    """
    >>> [CARLOS]
    O que faz: imagina que eu soltei uma bomba em 'cel' e olha pelo lado do INIMIGO:
    ele consegue fugir? Devolve (menor distância de fuga, quantas saídas ele tem).
    (99, 0) = não tem como fugir. None = ele nem está no alcance do fogo.
    """
    pv = _calcular_perigo(mapa, bombas, (cel[0], cel[1], nivel, t_bomba), t_fogo)
    if _livre_apos(pv, inimigo, 0.0, 0.0):
        return None
    fila = deque([(inimigo, 0)])
    visto = {inimigo}
    destinos = []
    bloq = bloqueados | {cel}
    while fila:
        c, n = fila.popleft()
        if n >= 1 and _livre_apos(pv, c, (n - 1) * 0.10, 0.0):
            destinos.append(n)
            continue
        if n >= 30:
            continue
        for dx, dy in DIRS.values():
            v = (c[0] + dx, c[1] + dy)
            if v in visto or v in bloq or not _andavel(mapa, v[0], v[1]):
                continue
            if not _seguro_em(pv, v, n * 0.10, 0.10, 0.0):
                continue
            visto.add(v)
            fila.append((v, n + 1))
    if not destinos:
        return (99, 0)
    return (min(destinos), len(destinos))


def _pressiona(mapa, bombas, bloqueados, cel, nivel, t_bomba, t_fogo, inimigos):
    """
    >>> [CARLOS]
    O que faz: responde "vale a pena soltar bomba em 'cel' para pressionar algum inimigo?".
    Sim quando o inimigo não tem fuga, ou a fuga dele é longa (>= FUGA_MIN passos),
    ou ele tem poucas saídas (<= DEST_MAX).
    >>> [NOVO] Otimização: só analisa o inimigo que está na linha da minha bomba
    (evita fazer a conta pesada para inimigos que a bomba nem alcança).
    """
    linha = set(_celulas_explosao(mapa, cel[0], cel[1], nivel))
    for e in inimigos:
        if e not in linha:
            continue
        f = _fuga_inimigo(mapa, bombas, bloqueados, cel, nivel, t_bomba, t_fogo, e)
        if f is None:
            continue
        if f[1] == 0 or f[0] >= FUGA_MIN or f[1] <= DEST_MAX:
            return True
    return False


def _bomba_util(mapa, cel, nivel, inimigos):
    """A bomba nessa célula atingiria um bloco quebrável ou um inimigo?"""
    for c in _celulas_explosao(mapa, cel[0], cel[1], nivel)[1:]:
        if mapa[c[1]][c[0]] == 1 or c in inimigos:
            return True
    return False


# ------------------------------------------------------------------------------
# >>> [JOSÉ PEDRO] CAMADA TÁTICA
# O que faz (resumo): olha o que cada inimigo pode fazer, escolhe o "modo" de jogo
# (grupo / caça / duelo), evita casas onde um inimigo poderia me encurralar, não
# destrói power-up que ainda dá para pegar, espera em "postos" seguros e só ataca
# quando a armadilha é garantida.
# ------------------------------------------------------------------------------
def _nivel_de(p):
    """>>> [NOVO] Nível da bomba de um jogador. Se a main não tiver esse dado, assume 1 (não trava)."""
    return getattr(p, "bomba_nivel", 1)


def _maxb_de(p):
    """>>> [NOVO] Máximo de bombas de um jogador. Se a main não tiver esse dado, assume 1."""
    return getattr(p, "max_bombas", 1)


def _bombas_de(p):
    """>>> [NOVO] Quantas bombas o jogador tem no mapa agora. Se a main não tiver esse dado, assume 0."""
    return len(getattr(p, "bombas", None) or [])


def _ameaca(p):
    """>>> [JOSÉ PEDRO] Quanto um inimigo é perigoso: mais bombas e mais alcance = mais perigoso."""
    return _maxb_de(p) + 1.5 * _nivel_de(p)


def _modo(mapa, players, tempo_restante):
    """
    >>> [JOSÉ PEDRO]
    O que faz: escolhe o jeito de jogar conforme a situação.
      "duelo" = só 2 vivos (mais agressivo); "caça" = sem blocos ou pouco tempo (monta armadilhas);
      "grupo" = 3 ou 4 vivos (coleta power-ups e sobrevive).
    """
    if MODO_FORCADO in ("grupo", "caca", "duelo"):
        return MODO_FORCADO
    vivos = sum(1 for p in players if p.ativo)
    if vivos <= 2:
        return "duelo"
    blocos = sum(linha.count(1) for linha in mapa)
    if blocos == 0 or tempo_restante < 60:
        return "caca"
    return "grupo"


def _risco_emboscada(mapa, cel, inimigos_p, bombas, bloqueados, t_bomba, t_fogo):
    """
    >>> [JOSÉ PEDRO]
    O que faz: True se algum inimigo puder soltar uma bomba (onde está, ou depois de dar 1 passo)
    que me deixe SEM rota de fuga se eu estiver em 'cel'. É o contrário da armadilha: aqui o
    encurralado seria eu.
    """
    for p in inimigos_p:
        if _bombas_de(p) >= _maxb_de(p):
            continue
        nv = _nivel_de(p)
        alc = 1 + (nv - 1) * 2
        ex, ey = p.grid_x, p.grid_y
        if abs(ex - cel[0]) + abs(ey - cel[1]) > alc + 2:
            continue
        origens = [(ex, ey)] + [(ex + dx, ey + dy) for dx, dy in DIRS.values()
                                if _andavel(mapa, ex + dx, ey + dy)]
        for o in origens:
            if o in bloqueados:
                continue
            if cel not in _celulas_explosao(mapa, o[0], o[1], nv):
                continue
            pv = _calcular_perigo(mapa, bombas, (o[0], o[1], nv, t_bomba), t_fogo)
            if _buscar(mapa, cel, bloqueados | {o}, pv, lambda c, t: _livre_apos(pv, c, t),
                       max_passos=14) is None:
                return True
    return False


class _BombaVirtual:
    """>>> [JOSÉ PEDRO] Bomba imaginária (só para simular); nunca é entregue ao jogo."""
    def __init__(self, x, y, nivel, t):
        self.x, self.y, self.nivel, self.tempo_explosao = x, y, nivel, t
        self.explodida, self.tempo_fogo, self.fogo, self.dono = False, 0.0, [], None


def _escape_tatico(mapa, cel, nivel, bombas, bloqueados, t_bomba, t_fogo, inimigos_p):
    """
    >>> [JOSÉ PEDRO]
    O que faz: responde "é seguro soltar bomba em 'cel'?". Além de existir rota de fuga, a rota
    não pode passar por casas onde um inimigo poderia me trancar soltando a bomba dele
    (o erro clássico: fugir para um corredor e o inimigo fechar a saída).
    """
    pv = _calcular_perigo(mapa, bombas, (cel[0], cel[1], nivel, t_bomba), t_fogo)
    bl = set(bloqueados)
    if inimigos_p:
        com_minha = list(bombas) + [_BombaVirtual(cel[0], cel[1], nivel, t_bomba)]
        for dx, dy in DIRS.values():
            v = (cel[0] + dx, cel[1] + dy)
            if (_andavel(mapa, v[0], v[1]) and v not in bl
                    and _risco_emboscada(mapa, v, inimigos_p, com_minha, bl | {cel}, t_bomba, t_fogo)):
                bl.add(v)
    return _buscar(mapa, cel, bl, pv, lambda c, t: _livre_apos(pv, c, t), max_passos=14) is not None


def _posto(mapa, eu, bloqueados, perigo, inimigos_p, bombas, modo, t_bomba, t_fogo):
    """
    >>> [JOSÉ PEDRO]
    O que faz: escolhe um bom lugar para ESPERAR (em vez de correr atrás do inimigo): longe das
    bombas, com várias saídas, fora da linha de tiro dos inimigos e (conforme o modo) a uma
    distância boa deles. Devolve a primeira ação para chegar lá, ou None para ficar parado.
    """
    bombas_vivas = [(b.x, b.y) for b in bombas if not b.explodida]
    ini = [(p.grid_x, p.grid_y) for p in inimigos_p]
    linhas = set()
    for p in inimigos_p:
        linhas |= set(_celulas_explosao(mapa, p.grid_x, p.grid_y, _nivel_de(p)))

    def pont(c, n):
        d_b = min([abs(c[0] - bx) + abs(c[1] - by) for bx, by in bombas_vivas] + [8])
        saidas = sum(1 for dx, dy in DIRS.values()
                     if _andavel(mapa, c[0] + dx, c[1] + dy) and (c[0] + dx, c[1] + dy) not in bloqueados)
        s = 1.0 * min(d_b, 8) + 1.5 * saidas - 0.15 * n
        if c in linhas:
            s -= 4.0
        if ini:
            d_i = min(min(abs(c[0] - x) + abs(c[1] - y) for x, y in ini), 10)
            if modo == "grupo":
                s += 0.4 * min(d_i, 7)
            elif modo == "caca":
                s -= 0.8 * abs(d_i - 3)
            else:
                s -= 1.0 * abs(d_i - 2)
        return s

    cands = [(pont(eu, 0), eu, None)]
    fila = deque([(eu, 0, None)])
    visto = {eu}
    while fila:
        cel, n, primeira = fila.popleft()
        if n >= 10:
            continue
        for nome, (dx, dy) in DIRS.items():
            viz = (cel[0] + dx, cel[1] + dy)
            if viz in visto or viz in bloqueados or not _andavel(mapa, viz[0], viz[1]):
                continue
            if not _seguro_em(perigo, viz, n * PASSO):
                continue
            visto.add(viz)
            pr = primeira or nome
            fila.append((viz, n + 1, pr))
            if _livre_apos(perigo, viz, n * PASSO):
                cands.append((pont(viz, n + 1), viz, pr))
    atual = cands[0][0]
    cands.sort(key=lambda t: -t[0])
    for s, c, pr in cands[:4]:
        if pr is None:
            return None                     # ficar onde estou é o melhor
        if s < atual + 0.7:
            return None                     # não compensa trocar (evita ficar vai-e-vem)
        if not (inimigos_p and _risco_emboscada(mapa, c, inimigos_p, bombas, bloqueados, t_bomba, t_fogo)):
            return pr
    return None


def _escolher_tatico(player, mapa, players, bombas, tempo_restante, eu, nivel, t_fogo, t_bomba,
                     bloqueados, perigo, inimigos, est, self_state):
    """
    >>> [JOSÉ PEDRO] (com encaixes do CARLOS, RYAN e MAICON, marcados abaixo)
    O que faz: decide o que fazer quando a IA está segura. Ordem: bomba aqui -> (pressão)
    -> caçar com armadilha garantida -> power-up -> aproximar -> casa boa para bomba -> posto de espera.
    """
    inimigos_p = [p for p in players if p is not player and p.ativo]
    modo = _modo(mapa, players, tempo_restante)
    # >>> [RYAN] também respeita a pausa entre bombas (cooldown)
    pode_bomba = (self_state["bombas_ativas"] < self_state["max_bombas"]
                  and eu not in bloqueados and _cooldown_ok(est, tempo_restante))
    # >>> [CARLOS + JOSÉ PEDRO] pressão só nos modos agressivos
    pressao_on = PRESSAO and modo in PRESSAO_MODOS

    def passo_ok(nome):
        if not inimigos_p:
            return True
        dx, dy = DIRS[nome]
        return not _risco_emboscada(mapa, (eu[0] + dx, eu[1] + dy), inimigos_p, bombas, bloqueados,
                                    t_bomba, t_fogo)

    def escape_ok(cel):
        return _escape_tatico(mapa, cel, nivel, bombas, bloqueados, t_bomba, t_fogo, inimigos_p)

    # power-up alcançável? (se sim, não vale explodir um bloco que o destruiria)
    rpu = _buscar(mapa, eu, bloqueados, perigo,
                  lambda c, t: mapa[c[1]][c[0]] in (3, 4) and _livre_apos(perigo, c, t), max_passos=14)

    def bomba_boa(cel):
        cels = _celulas_explosao(mapa, cel[0], cel[1], nivel)[1:]
        if any(c in inimigos for c in cels):
            return True
        bloco = any(mapa[y][x] == 1 for x, y in cels)
        pu = any(mapa[y][x] in (3, 4) for x, y in cels)
        return bloco and not (pu and rpu is not None)

    # 1) soltar bomba aqui (útil + fuga garantida)
    # >>> [MAICON] a ideia dele (soltar bomba ao lado de bloco quebrável) é este passo,
    # mas aqui só solta se a fuga for garantida e sem destruir power-up que dá para pegar.
    if pode_bomba and bomba_boa(eu) and escape_ok(eu):
        est["n"] = 0
        return "bomba"

    # 1b) >>> [CARLOS] pressão: inimigo com pouca chance de fuga e eu tenho fuga -> solta bomba
    if pressao_on and pode_bomba and inimigos:
        if (_pressiona(mapa, bombas, bloqueados, eu, nivel, t_bomba, t_fogo, inimigos)
                and escape_ok(eu)):
            est["n"] = 0
            return "bomba"

    # 2) caçar com armadilha garantida, começando pelo inimigo MAIS PERIGOSO
    if inimigos_p and pode_bomba:
        for e in sorted(inimigos_p, key=lambda p: (-_ameaca(p), abs(p.grid_x - eu[0]) + abs(p.grid_y - eu[1]))):
            ec = (e.grid_x, e.grid_y)
            linha_e = set(_celulas_explosao(mapa, ec[0], ec[1], nivel))

            def meta_cacar(c, t, ec=ec, linha_e=linha_e):
                if c not in linha_e or not _livre_apos(perigo, c, t):
                    return False
                if not _preso(mapa, bombas, bloqueados, c, nivel, t_bomba, t_fogo, ec):
                    return False
                return escape_ok(c)
            r = _buscar(mapa, eu, bloqueados, perigo, meta_cacar, max_passos=12)
            if r and passo_ok(r[0]):
                est["n"] = 0
                return r[0]

    # 2b) >>> [CARLOS] caçar com pressão: anda até um ponto onde a bomba pressiona o inimigo
    if pressao_on and CACA_PRESSAO and inimigos and pode_bomba:
        def meta_pressao(c, t):
            return (_livre_apos(perigo, c, t)
                    and _pressiona(mapa, bombas, bloqueados, c, nivel, t_bomba, t_fogo, inimigos)
                    and escape_ok(c))
        r = _buscar(mapa, eu, bloqueados, perigo, meta_pressao, max_passos=CACA_PRESSAO)
        if r and passo_ok(r[0]):
            est["n"] = 0
            return r[0]

    # 3) power-up próximo
    if rpu and passo_ok(rpu[0]):
        est["n"] = 0
        return rpu[0]

    def aproximar():
        if not inimigos or not (modo in ("caca", "duelo") or r4 is None):
            return None
        alvo_d = 2 if modo == "duelo" else 3
        if min(abs(ex - eu[0]) + abs(ey - eu[1]) for ex, ey in inimigos) <= alvo_d:
            return None
        r = _buscar(mapa, eu, bloqueados, perigo,
                    lambda c, t: _livre_apos(perigo, c, t)
                    and min(abs(ex - c[0]) + abs(ey - c[1]) for ex, ey in inimigos) <= alvo_d)
        return r if (r and passo_ok(r[0])) else None

    # 4) (duelo: primeiro chegar perto do inimigo) / casa útil para bomba
    r4 = None
    if modo == "duelo":
        r = aproximar()
        if r:
            est["n"] = 0
            return r[0]
    r4 = _buscar(mapa, eu, bloqueados, perigo,
                 lambda c, t: _livre_apos(perigo, c, t) and bomba_boa(c) and escape_ok(c))
    if r4 and passo_ok(r4[0]):
        est["n"] = 0
        return r4[0]
    if modo != "duelo":
        r = aproximar()
        if r:
            est["n"] = 0
            return r[0]

    # 5) posto de espera seguro (não fica correndo atrás de ninguém)
    a = _posto(mapa, eu, bloqueados, perigo, inimigos_p, bombas, modo, t_bomba, t_fogo)
    return a or "parado"


# ------------------------------------------------------------------------------
# Decisão do "professor" (regras). A rede imita o que ele faz.
# ------------------------------------------------------------------------------
def _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state):
    eu = (self_state["grid_x"], self_state["grid_y"])
    nivel = self_state["bomba_nivel"]
    t_fogo = hud_info.get("tempo_fogo", 0.5)
    t_bomba = hud_info.get("tempo_explosao", 4)

    bloqueados = {(b.x, b.y) for b in bombas if not b.explodida}   # bomba bloqueia passagem
    perigo = _calcular_perigo(mapa, bombas, None, t_fogo)
    inimigos = {(p.grid_x, p.grid_y) for p in players if p is not player and p.ativo}
    est = _estado(player)

    # 1) FUGA: se a minha casa vai pegar fogo, fugir tem prioridade total
    if not _livre_apos(perigo, eu, 0.0):
        if TATICA:
            # >>> [JOSÉ PEDRO] prefere rotas cujo primeiro passo NÃO possa ser fechado por uma bomba inimiga
            ip = [p for p in players if p is not player and p.ativo]
            if ip:
                bl2 = set(bloqueados)
                for dx, dy in DIRS.values():
                    v = (eu[0] + dx, eu[1] + dy)
                    if (_andavel(mapa, v[0], v[1]) and v not in bl2
                            and _risco_emboscada(mapa, v, ip, bombas, bloqueados, t_bomba, t_fogo)):
                        bl2.add(v)
                if len(bl2) > len(bloqueados):
                    r = _buscar(mapa, eu, bl2, perigo, lambda c, t: _livre_apos(perigo, c, t))
                    if r:
                        return r[0]
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

    # >>> [JOSÉ PEDRO] com a camada tática ligada, ela decide tudo daqui para baixo
    if TATICA:
        return _escolher_tatico(player, mapa, players, bombas, tempo_restante, eu, nivel, t_fogo, t_bomba,
                                bloqueados, perigo, inimigos, est, self_state)

    # ---- Caminho ANTIGO (TATICA = False): Marcos + Carlos + Ryan + Maicon, para comparar ----
    # >>> [RYAN] 'pode_bomba' também exige que a pausa entre bombas (cooldown) tenha passado
    pode_bomba = (self_state["bombas_ativas"] < self_state["max_bombas"]
                  and eu not in bloqueados and _cooldown_ok(est, tempo_restante))

    def escape_ok(cel):
        # simula soltar a bomba em 'cel' e verifica se existe rota de fuga a tempo
        pv = _calcular_perigo(mapa, bombas, (cel[0], cel[1], nivel, t_bomba), t_fogo)
        # só aceita rotas curtas (até ~1.7s): sobra tempo para replanejar se algo mudar
        return _buscar(mapa, cel, bloqueados, pv, lambda c, t: _livre_apos(pv, c, t),
                       max_passos=14) is not None

    # 2) Soltar bomba aqui (só se for útil E houver rota de fuga garantida)
    # >>> [MAICON] ideia dele: bomba ao lado de bloco quebrável (aqui com fuga garantida)
    if pode_bomba and _bomba_util(mapa, eu, nivel, inimigos) and escape_ok(eu):
        est["n"] = 0
        return "bomba"

    # 2a) >>> [CARLOS] Pressão: inimigo com pouca chance de fuga e eu tenho fuga garantida -> bomba
    if PRESSAO and pode_bomba and inimigos:
        if (_pressiona(mapa, bombas, bloqueados, eu, nivel, t_bomba, t_fogo, inimigos)
                and escape_ok(eu)):
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

    # 2c) >>> [CARLOS] Caçar com pressão
    if PRESSAO and CACA_PRESSAO and inimigos and pode_bomba:
        def meta_pressao(c, t):
            return (_livre_apos(perigo, c, t)
                    and _pressiona(mapa, bombas, bloqueados, c, nivel, t_bomba, t_fogo, inimigos)
                    and escape_ok(c))
        r = _buscar(mapa, eu, bloqueados, perigo, meta_pressao, max_passos=CACA_PRESSAO)
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
# >>> [MARCOS] Toda esta parte (rede MLP, dados, treino DAgger, salvar/carregar)
# ==============================================================================
_DIR = os.path.dirname(os.path.abspath(__file__))

# >>> [NOVO] Nomes dos arquivos com o nome da equipe.
ARQ_REDE = os.path.join(_DIR, f"rede_{NOME}.npz")     # pesos da rede (NumPy)
ARQ_DADOS = os.path.join(_DIR, f"dados_{NOME}.npz")   # exemplos de treino
ARQ_PTH = os.path.join(_DIR, f"rede_{NOME}.pth")      # cópia em formato torch (se o torch existir)
# Se ainda não existir arquivo TchkBum, aproveita o treino antigo (jogador4), caso exista:
ARQ_REDE_ANTIGO = os.path.join(_DIR, "rede_TchakaBum.npz")
ARQ_DADOS_ANTIGO = os.path.join(_DIR, "dados_TchakaBum.npz")

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba"]

# >>> [JOSÉ PEDRO] A rede ganhou 10 entradas novas (modo, risco de emboscada, ameaça...):
# 103 -> 113. As 5 últimas entradas continuam sendo a máscara de ações seguras.
# >>> [NOVO] Para NÃO perder o treino já feito, redes/dados antigos de 103 entradas são
# convertidos sozinhos (as 10 entradas novas começam em zero). Veja _migrar_pesos/_migrar_dados.
N_IN_ANTIGO = 103
N_EXTRA = 10
POS_EXTRA = 98          # as entradas novas entram logo antes das 5 da máscara
N_IN = N_IN_ANTIGO + N_EXTRA
N_H, N_OUT = 64, 5
LR = 1e-3
BATCH = 64
CAP_DADOS = 30000


def _migrar_pesos(w1):
    """>>> [NOVO] Converte a primeira camada de 103 para 113 entradas (linhas novas = 0)."""
    return np.concatenate([w1[:POS_EXTRA], np.zeros((N_EXTRA, w1.shape[1]), np.float32),
                           w1[POS_EXTRA:]], axis=0).astype(np.float32)


def _migrar_dados(s):
    """>>> [NOVO] Converte exemplos de treino de 103 para 113 entradas (colunas novas = 0)."""
    return np.concatenate([s[:, :POS_EXTRA], np.zeros((len(s), N_EXTRA), np.float32),
                           s[:, POS_EXTRA:]], axis=1).astype(np.float32)


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
        self.migrado = False

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
        with np.load(caminho) as d:
            s = d["S"]
            if s.ndim == 2 and s.shape[1] == N_IN_ANTIGO:      # >>> [NOVO] dados antigos (103 entradas)
                s = _migrar_dados(s)
                self.migrado = True
            if s.ndim != 2 or s.shape[1] != N_IN:
                return False
            n = min(len(s), CAP_DADOS)
            self.S[:n], self.L[:n] = s[:n], d["L"][:n]
            self.n, self.ptr = n, int(d["ptr"]) % CAP_DADOS
        return True


_rede = Rede()
_dados = Dados()
_rede_ok = False     # >>> [JOSÉ PEDRO] True só se uma rede treinada foi carregada do arquivo
_passos = 0          # total de exemplos de treino já vistos (guardado junto com os pesos)
_partidas = 0        # quantas execuções do jogo a rede já treinou (guardado junto com os pesos)
_novos = 0
_avisou_torch = False

# >>> [NOVO] Contadores para o relatório final das decisões.
_stats = {"rede": 0,          # decididas pela rede neural
          "regras": 0,        # decididas só pelas regras (sem rede treinada ou USAR_REDE = False)
          "seguranca": 0,     # fugas feitas pela camada de segurança
          "emboscada": 0,     # vezes em que evitou uma casa de risco de emboscada (camada tática)
          "exploracao": 0,    # ações aleatórias seguras (só no treino)
          "vaivem": 0,        # vezes que o anti-vaivém trocou a ação da rede
          "sem_acao": 0,      # nenhuma ação segura: ficou parada
          "comparadas": 0,    # vezes em que comparou a rede com o professor (só no treino)
          "concorda": 0}      # vezes em que a rede escolheu o mesmo que o professor
_contagem = {a: 0 for a in ACOES + ["parado"]}   # quantas vezes cada ação foi escolhida
_modos = {"grupo": 0, "caca": 0, "duelo": 0}      # em que modo tático a IA estava
_ultimas = deque(maxlen=15)                       # as últimas decisões (para mostrar no final)


def _carregar(verbose=True):
    global _passos, _partidas, _rede, _rede_ok
    try:
        arq_rede = ARQ_REDE if os.path.exists(ARQ_REDE) else (
            ARQ_REDE_ANTIGO if os.path.exists(ARQ_REDE_ANTIGO) else None)
        arq_dados = ARQ_DADOS if os.path.exists(ARQ_DADOS) else (
            ARQ_DADOS_ANTIGO if os.path.exists(ARQ_DADOS_ANTIGO) else None)
        sem_rede = ("começando uma rede nova (treino)." if TREINAR
                    else "usando as regras (sem rede).")
        if arq_rede:
            with np.load(arq_rede) as d:
                w1 = d["W1"]
                migrar = (w1.shape == (N_IN_ANTIGO, N_H))
                if (w1.shape == (N_IN, N_H) or migrar) and d["W3"].shape == (N_H, N_OUT):
                    for k in Rede.NOMES:
                        _rede.p[k] = d[k].astype(np.float32)
                    if migrar:
                        _rede.p["W1"] = _migrar_pesos(_rede.p["W1"])
                    _passos = int(d["passos"])
                    _partidas = int(d["partidas"]) if "partidas" in d.files else 0
                    _rede_ok = True
                    if verbose:
                        print(f"[IA {NOME}] REDE NEURAL CARREGADA de {arq_rede} "
                              f"({_partidas} partidas, {_passos} exemplos de treino).")
                        if migrar:
                            print(f"[IA {NOME}] Rede antiga (103 entradas) convertida para {N_IN} entradas "
                                  f"sem perder o treino; ao salvar ela fica no formato novo.")
                elif verbose:
                    print(f"[IA {NOME}] Arquivo de rede incompatível: {sem_rede}")
        elif verbose:
            print(f"[IA {NOME}] Nenhuma rede salva: {sem_rede}")
        # >>> [JOSÉ PEDRO] os exemplos de treino só são carregados quando vai treinar
        if TREINAR and arq_dados:
            if _dados.carregar(arq_dados) and verbose:
                print(f"[IA {NOME}] Dados de treino carregados ({_dados.n} exemplos"
                      f"{', convertidos para o formato novo' if _dados.migrado else ''}).")
    except Exception as e:
        print(f"[IA {NOME}] Não consegui carregar a rede ({e}): "
              + ("começando uma rede nova." if TREINAR else "usando as regras (sem rede)."))
        _rede = Rede()
        _rede_ok = False


def _salvar(com_dados=True):
    """Salva a rede (.npz), os dados de treino (.npz) e, se houver torch, a cópia .pth."""
    global _avisou_torch
    try:
        tmp = ARQ_REDE + ".tmp.npz"
        np.savez(tmp, passos=_passos, partidas=_partidas, **_rede.p)
        os.replace(tmp, ARQ_REDE)
        print(f"[IA {NOME}] REDE NEURAL SALVA em {ARQ_REDE} ({os.path.getsize(ARQ_REDE)} bytes) | "
              f"partidas treinadas: {_partidas} | exemplos: {_passos} | "
              f"decisões da rede: {_stats['rede']} | fugas pela camada de segurança: {_stats['seguranca']}")
        if com_dados and _dados.n > 0:
            _dados.salvar(ARQ_DADOS)
            print(f"[IA {NOME}] DADOS DE TREINO SALVOS em {ARQ_DADOS} ({_dados.n} exemplos)")
        try:                                   # cópia em formato torch, só se o torch estiver instalado
            import torch
            torch.save({k: torch.from_numpy(v.copy()) for k, v in _rede.p.items()}, ARQ_PTH)
            print(f"[IA {NOME}] Cópia em formato torch salva em {ARQ_PTH}")
        except ImportError:
            if not _avisou_torch:               # >>> [NOVO] aviso claro de por que não tem .pth
                _avisou_torch = True
                print(f"[IA {NOME}] torch NÃO instalado: o arquivo .pth não foi gerado "
                      f"(para gerar, instale com: pip install torch). O .npz está salvo normalmente.")
        except Exception as e:
            print(f"[IA {NOME}] Não consegui salvar o .pth: {e}")
    except Exception as e:
        print(f"[IA {NOME}] ERRO ao salvar a rede: {e}")


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


# ------------------------------------------------------------------------------
# >>> [NOVO] Relatório final: mostra as tomadas de decisão da IA
# ------------------------------------------------------------------------------
def _fmt_t(t):
    try:
        return f"{float(t):7.1f}s"
    except (TypeError, ValueError):
        return "      ?"


def _relatorio():
    """
    >>> [NOVO]
    O que faz: ao fechar o jogo, imprime um resumo de como a IA decidiu: quantas decisões,
    quem decidiu (rede, regras ou segurança), quantas vezes foi cada ação, em que modo tático
    estava, o quanto a rede concordou com o professor e as últimas decisões.
    """
    total = sum(_contagem.values())
    if total == 0:
        return

    def pct(v):
        return 100.0 * v / total

    print("=" * 70)
    print(f"[IA {NOME}] RELATÓRIO FINAL DE DECISÕES")
    print("=" * 70)
    print(f"Total de decisões: {total}")
    print(f"  decididas pela rede neural ........ {_stats['rede']} ({pct(_stats['rede']):.1f}%)")
    print(f"      dentre elas: {_stats['exploracao']} por exploração segura (treino) e "
          f"{_stats['vaivem']} desvios anti-vaivém")
    if _stats["regras"]:
        print(f"  decididas só pelas regras ......... {_stats['regras']} ({pct(_stats['regras']):.1f}%)")
    print(f"  fugas pela camada de segurança .... {_stats['seguranca']} ({pct(_stats['seguranca']):.1f}%)")
    print(f"  sem ação segura (ficou parada) .... {_stats['sem_acao']} ({pct(_stats['sem_acao']):.1f}%)")
    print("Ações escolhidas: " + " | ".join(f"{a}={_contagem.get(a, 0)}" for a in ACOES + ["parado"]))
    if sum(_modos.values()):
        print(f"Modo tático (decisões fora de fuga): grupo={_modos['grupo']} | "
              f"caça={_modos['caca']} | duelo={_modos['duelo']}")
        print(f"Casas de risco de emboscada evitadas: {_stats['emboscada']} vezes")
    if _stats["comparadas"]:
        c, n = _stats["concorda"], _stats["comparadas"]
        print(f"Rede x professor: a rede escolheu o mesmo que as regras em {c} de {n} "
              f"decisões ({100.0 * c / n:.1f}%)")
    print(f"Últimas {len(_ultimas)} decisões (tempo | posição | ação | quem decidiu):")
    for t, pos, acao, origem in _ultimas:
        print(f"   {_fmt_t(t)} | {str(pos):>8} | {acao:<8} | {origem}")
    print(f"Arquivos: {ARQ_REDE}")
    print(f"          {ARQ_DADOS}")
    print(f"          {ARQ_PTH} (só existe se o torch estiver instalado)")
    print("=" * 70)


def _fim_de_jogo(salvar=True):
    """Roda sozinha quando o jogo fecha: salva a rede (se estiver treinando) e mostra o relatório."""
    global _partidas
    try:
        if sum(_contagem.values()) == 0:
            return            # >>> [NOVO] esta IA nem jogou nesta execução: não salva nem conta partida
        if TREINAR:
            _partidas += 1
            if salvar:
                _salvar(True)
        else:
            print(f"[IA {NOME}] TREINAR = False (modo só jogar): a rede NÃO foi salva nesta execução.")
        _relatorio()
    except Exception as e:
        print(f"[IA {NOME}] Erro no fim de jogo: {e}")


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
              bombas, t_bomba, t_fogo, extra):
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
    f += list(extra)             # 10 >>> [JOSÉ PEDRO]: modo, risco de emboscada, ameaça, vivos, blocos
    f += [float(m) for m in mask]                                                        # 5: ações seguras (SEMPRE por último)
    return np.array(f, dtype=np.float32)


def _evita_vaivem(acao_idx, q, validas, eu, hist):
    """
    >>> [RYAN] (ideia do histórico de posições dele, adaptada ao código do Marcos)
    O que faz: se a rede quer andar para uma casa que a IA já visitou várias vezes nas
    últimas posições (ficar indo e voltando), troca por outra ação SEGURA menos visitada.
    Só troca entre ações que a camada de segurança já aprovou (nunca arrisca).
    Devolve (ação, True se trocou).
    """
    if acao_idx >= 4 or len(hist) < 6:
        return acao_idx, False
    vetores = list(DIRS.values())

    def visitas(i):
        dx, dy = vetores[i]
        return hist.count((eu[0] + dx, eu[1] + dy))

    atual = visitas(acao_idx)
    if atual < LIMITE_VAIVEM:
        return acao_idx, False
    alternativas = [i for i in validas if i < 4 and visitas(i) < atual]
    if not alternativas:
        return acao_idx, False
    novo = max(alternativas, key=lambda i: (-visitas(i), float(q[i])))
    return novo, True


def _decidir_rede(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state):
    """Devolve (ação, quem_decidiu). 'quem_decidiu' alimenta o relatório final."""
    global _passos, _novos
    eu = (self_state["grid_x"], self_state["grid_y"])
    nivel = self_state["bomba_nivel"]
    t_fogo = hud_info.get("tempo_fogo", 0.5)
    t_bomba = hud_info.get("tempo_explosao", 4)
    est = _estado(player)

    bloqueados = {(b.x, b.y) for b in bombas if not b.explodida}
    perigo = _calcular_perigo(mapa, bombas, None, t_fogo)
    inimigos = {(p.grid_x, p.grid_y) for p in players if p is not player and p.ativo}
    inimigos_p = [p for p in players if p is not player and p.ativo]    # >>> [JOSÉ PEDRO]

    # Em perigo, a camada de segurança foge sozinha (a rede não tem permissão para arriscar)
    if not _livre_apos(perigo, eu, 0.0):
        _stats["seguranca"] += 1
        acao = _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
        return acao, "fuga (segurança)"

    # >>> [RYAN] também respeita a pausa entre bombas (cooldown)
    pode_bomba = (self_state["bombas_ativas"] < self_state["max_bombas"]
                  and eu not in bloqueados and _cooldown_ok(est, tempo_restante))
    util = _bomba_util(mapa, eu, nivel, inimigos)
    if TATICA:   # >>> [JOSÉ PEDRO] a fuga da bomba também considera emboscada de inimigos
        esc_ok = pode_bomba and util and _escape_tatico(mapa, eu, nivel, bombas, bloqueados, t_bomba,
                                                        t_fogo, inimigos_p)
    else:
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

    # >>> [JOSÉ PEDRO] camada tática: tira da lista as casas onde um inimigo poderia me
    # encurralar (só se sobrar alguma outra opção segura)
    risco = [0, 0, 0, 0]
    if TATICA and inimigos_p:
        for i, (dx, dy) in enumerate(DIRS.values()):
            if mask[i] and _risco_emboscada(mapa, (eu[0] + dx, eu[1] + dy), inimigos_p, bombas,
                                            bloqueados, t_bomba, t_fogo):
                risco[i] = 1
        livres = [mask[i] and not risco[i] for i in range(4)] + [mask[4]]
        if any(livres):
            if any(risco):
                _stats["emboscada"] += 1
            mask = [int(v) for v in livres]
    validas = [i for i in range(5) if mask[i]]
    if not validas:
        _stats["sem_acao"] += 1
        return "parado", "sem ação segura"

    modo = _modo(mapa, players, tempo_restante)
    _modos[modo] = _modos.get(modo, 0) + 1

    # >>> [JOSÉ PEDRO] sem rede treinada (e sem estar treinando), as regras decidem
    if not (_rede_ok or TREINAR):
        _stats["regras"] += 1
        prof = _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
        acao = prof if (prof in ACOES and mask[ACOES.index(prof)]) else ACOES[validas[0]]
        return acao, "regras (sem rede treinada)"

    _stats["rede"] += 1
    vivos = sum(1 for p in players if p.ativo)
    amax = max([_ameaca(p) for p in inimigos_p] + [0.0])
    extra = ([float(modo == "grupo"), float(modo == "caca"), float(modo == "duelo")]
             + [float(r) for r in risco]
             + [amax / 11.0, vivos / 4.0, sum(l.count(1) for l in mapa) / 143.0])
    s = _features(mapa, eu, nivel, perigo, bloqueados, inimigos, pode_bomba, util, esc_ok, mask,
                  bombas, t_bomba, t_fogo, extra)
    q = np.where(np.array(mask) > 0, _rede.prever(s), -1e9)
    acao_idx = int(np.argmax(q))
    acao_rede = acao_idx          # guarda o que a rede escolheu de verdade (para comparar com o professor)
    origem = "rede"

    # >>> [RYAN] anti-vaivém: evita ficar indo e voltando entre as mesmas casas
    novo, desviou = _evita_vaivem(acao_idx, q, validas, eu, est["hist"])
    if desviou:
        acao_idx = novo
        origem = "rede (anti-vaivém)"
        _stats["vaivem"] += 1

    if TREINAR:
        # professor: o que as regras fariam neste estado?
        prof = _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
        if prof in ACOES:
            # >>> [NOVO] conta quantas vezes a rede pensou igual ao professor
            _stats["comparadas"] += 1
            if ACOES[acao_rede] == prof:
                _stats["concorda"] += 1
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
            origem = "exploração (treino)"
            _stats["exploracao"] += 1
    return ACOES[acao_idx], origem


def _registrar(player, acao, origem, tempo_restante, self_state):
    """
    >>> [NOVO + RYAN]
    O que faz: anota a decisão para o relatório final, guarda a posição no histórico
    (anti-vaivém) e a hora da última bomba (cooldown). Nunca atrapalha o jogo se der erro.
    """
    try:
        est = _estado(player)
        eu = (self_state["grid_x"], self_state["grid_y"])
        est["hist"].append(eu)
        if acao == "bomba":
            est["bomba_t"] = tempo_restante
        _contagem[acao] = _contagem.get(acao, 0) + 1
        _ultimas.append((tempo_restante, eu, acao, origem))
    except Exception:
        pass


def decidir_acao(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state):
    try:
        if USAR_REDE:
            acao, origem = _decidir_rede(player, mapa, players, bombas, tempo_restante, pontos,
                                         hud_info, self_state)
        else:
            acao = _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
            origem = "regras"
            _stats["regras"] += 1
    except Exception:
        traceback.print_exc()   # mostra o erro em vez de engolir em silêncio
        try:
            acao = _decidir(player, mapa, players, bombas, tempo_restante, pontos, hud_info, self_state)
            origem = "regras (reserva após erro)"
            _stats["regras"] += 1
        except Exception:
            acao, origem = "parado", "erro"
    _registrar(player, acao, origem, tempo_restante, self_state)
    return acao


_carregar()
