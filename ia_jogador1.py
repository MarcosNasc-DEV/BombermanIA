import torch
import torch.nn as nn
import os
import random
import pickle
import numpy as np
from collections import deque

# ==============================================================================
# 1. CONFIGURAÇÃO DE PERSISTÊNCIA (Pesos da Rede .pth e Dataset .pkl)
# ==============================================================================
ARQUIVO_MODELO = "rede_jogador1.pth"   # Salva o cérebro da Rede Neural (PyTorch)
DATA_DATASET = "dataset_jogador1.pkl"    # Salva o histórico de experiências

memoria_temporaria = []
historico_fogos = {}

# ==============================================================================
# 2. ARQUITETURA DA REDE NEURAL (DQN - PyTorch)
# ==============================================================================
class BombermanDQN(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, output_size=6):
        super(BombermanDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_size, output_size)  # 6 ações possíveis

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

# Instancia a Rede Neural, Otimizador e Perda
INPUT_SIZE = 42
modelo = BombermanDQN(input_size=INPUT_SIZE)
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
criterio = nn.MSELoss()

# Carrega os pesos salvos anteriormente se existirem (.pth)
if os.path.exists(ARQUIVO_MODELO):
    try:
        modelo.load_state_dict(torch.load(ARQUIVO_MODELO))
        modelo.eval()
        print(f"[IA] Pesos carregados com sucesso de {ARQUIVO_MODELO}")
    except:
        print(f"[IA] Erro ao carregar {ARQUIVO_MODELO}. Iniciando nova rede.")

mapa_acoes = {
    0: "parado",
    1: "cima",
    2: "baixo",
    3: "esquerda",
    4: "direita",
    5: "bomba"
}

# ==============================================================================
# 3. MAPEAMENTO DE PERIGO E FUGA (Escudo Antisuicídio)
# ==============================================================================
def get_danger_cells(bombas, cols, rows, mapa=None):
    global historico_fogos
    danger = set()
    celulas_bombas_ativas = set()

    for b in bombas:
        bx = getattr(b, 'grid_x', getattr(b, 'x', None))
        by = getattr(b, 'grid_y', getattr(b, 'y', None))
        if bx is None and isinstance(b, dict):
            bx = b.get('grid_x', b.get('x'))
        if by is None and isinstance(b, dict):
            by = b.get('grid_y', b.get('y'))
        if bx is not None and by is not None:
            if bx > cols: bx //= 32
            if by > rows: by //= 32
            
            raio = getattr(b, 'range', getattr(b, 'raio', 3))
            
            celulas_bomba_atual = [(bx, by)]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for r in range(1, raio + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if mapa is not None and 0 <= ny < len(mapa) and 0 <= nx < len(mapa[0]):
                            if mapa[ny][nx] == 2:  # Parede indestrutível bloqueia o fogo
                                break
                        celulas_bomba_atual.append((nx, ny))
                    else:
                        break
            
            for c in celulas_bomba_atual:
                celulas_bombas_ativas.add(c)
                danger.add(c)
                historico_fogos[c] = 60

    chaves_remover = []
    for pos, timer in historico_fogos.items():
        if pos not in celulas_bombas_ativas:
            if timer > 0:
                danger.add(pos)
                historico_fogos[pos] -= 1
            else:
                chaves_remover.append(pos)
        else:
            historico_fogos[pos] = 60

    for pos in chaves_remover:
        del historico_fogos[pos]

    return danger

# ==============================================================================
# 4. EXTRATOR DE ESTADOS (Converte o jogo em Tensor para a Rede Neural)
# ==============================================================================
def extrair_estado(self, mapa, players, bombas, hud_info):
    gx, gy = self.grid_x, self.grid_y
    cols = hud_info['cols']
    rows = hud_info['rows']
    features = []
    
    # 1. Visão local 5x5 ao redor do bot (25 valores)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                features.append(float(mapa[ny][nx]))
            else:
                features.append(2.0)  # Borda
                
    # 2. Atributos do bot (7 valores)
    features.append(float(self.max_bombas))
    features.append(float(len(self.bombas) if hasattr(self, 'bombas') else 0))
    features.append(float(self.bomba_nivel if hasattr(self, 'bomba_nivel') else 1))
    features.append(1.0 if self.ativo else 0.0)
    features.append(float(gx))
    features.append(float(gy))
    features.append(float(self.time if hasattr(self, 'time') else 0))
    
    # 3. Posição relativa do adversário mais próximo (2 valores - Foco em Caça)
    adversarios = [p for p in players if getattr(p, 'ativo', True) and p != self]
    if adversarios:
        alvo = min(adversarios, key=lambda p: abs(getattr(p, 'grid_x', getattr(p, 'x', 0))//32 if getattr(p, 'grid_x', getattr(p, 'x', 0))>cols else getattr(p, 'grid_x', getattr(p, 'x', 0)) - gx) + abs(getattr(p, 'grid_y', getattr(p, 'y', 0))//32 if getattr(p, 'grid_y', getattr(p, 'y', 0))>rows else getattr(p, 'grid_y', getattr(p, 'y', 0)) - gy))
        ax = getattr(alvo, 'grid_x', getattr(alvo, 'x', 0))
        ay = getattr(alvo, 'grid_y', getattr(alvo, 'y', 0))
        if ax > cols: ax //= 32
        if ay > rows: ay //= 32
        features.append(float(ax - gx))
        features.append(float(ay - gy))
    else:
        features.append(0.0)
        features.append(0.0)

    # 4. Posição relativa do Power-up / Upgrade mais próximo (2 valores - Foco em Evolução)
    pw_x, pw_y = 0.0, 0.0
    menor_d = 999
    for y in range(rows):
        for x in range(cols):
            if mapa[y][x] > 2:  # Ícones / Upgrades
                d = abs(x - gx) + abs(y - gy)
                if d < menor_d:
                    menor_d = d
                    pw_x, pw_y = float(x - gx), float(y - gy)
    features.append(pw_x)
    features.append(pw_y)

    # Preenche o restante se faltar tamanho
    while len(features) < INPUT_SIZE:
        features.append(0.0)
        
    return torch.tensor(features[:INPUT_SIZE], dtype=torch.float32)

# ==============================================================================
# 5. TOMADA DE DECISÃO PRINCIPAL
# ==============================================================================
def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    global memoria_temporaria
    try:
        gx = self_state['grid_x']
        gy = self_state['grid_y']
        cols = hud_info['cols']
        rows = hud_info['rows']
        
        danger = get_danger_cells(bombas, cols, rows, mapa)
        estao_em_perigo = (gx, gy) in danger

        # ======================================================================
        # PASSO 1: ESCUDO DE SOBREVIVÊNCIA ABSOLUTO (Fuga de emergência)
        # ======================================================================
        if estao_em_perigo:
            direcoes = [("cima", 0, -1), ("baixo", 0, 1), ("esquerda", -1, 0), ("direita", 1, 0)]
            random.shuffle(direcoes)
            for nome, dx, dy in direcoes:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    if mapa[ny][nx] in [0, 3, 4] and (nx, ny) not in danger:
                        return nome

        # ======================================================================
        # PASSO 2: TOMADA DE DECISÃO DA REDE NEURAL (DQN + Exploração)
        # ======================================================================
        estado_tensor = extrair_estado(player, mapa, jogadores, bombas, hud_info)
        
        # Epsilon-Greedy: Garante exploração para nunca ficar parado e aprender caminhos novos
        epsilon = 0.15 
        if random.random() < epsilon:
            acao_idx = random.randint(1, 5)  # Evita ficar parado (0)
        else:
            with torch.no_grad():
                q_values = modelo(estado_tensor.unsqueeze(0))
                acao_idx = torch.argmax(q_values).item()

        acao_escolhida = mapa_acoes.get(acao_idx, "baixo")
        if acao_escolhida == "parado":
            acao_escolhida = "direita"

        # ======================================================================
        # PASSO 3: GATILHO TÁTICO DE BOMBAS (Destruir paredes e atacar inimigos)
        # ======================================================================
        # Verifica se tem parede destrutível (1) ao redor ou se tem inimigo perto
        perto_de_bloco = any(0 <= gx+dx < cols and 0 <= gy+dy < rows and mapa[gy+dy][gx+dx] == 1 for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)])
        
        # Verifica se há rota segura para fugir após plantar a bomba
        tem_saida_segura = any(
            0 <= gx+dx < cols and 0 <= gy+dy < rows and 
            (mapa[gy+dy][gx+dx] == 0 or mapa[gy+dy][gx+dx] > 2) and 
            (gx+dx, gy+dy) not in danger 
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
        )

        if perto_de_bloco and tem_saida_segura and random.random() < 0.60:
            acao_escolhida = "bomba"

        # Salva transição na memória temporária para treino
        mapa_acoes_invertido = {"cima": 1, "baixo": 2, "esquerda": 3, "direita": 4, "bomba": 5, "parado": 0}
        if acao_escolhida in mapa_acoes_invertido:
            memoria_temporaria.append((estado_tensor.numpy(), mapa_acoes_invertido[acao_escolhida]))

        return acao_escolhida

    except Exception as e:
        print(f"⚠️ Erro na IA: {e}")
        return "baixo"

# ==============================================================================
# 6. SALVAMENTO AUTOMÁTICO DE PESOS (.pth) E DATASET (.pkl)
# ==============================================================================
import atexit

Moemoria_salva = False
def salvar_tudo():
    global memoria_temporaria
    if not memoria_temporaria:
        return
    
    # Atualiza Dataset .pkl
    dados_existentes = []
    if os.path.exists(DATA_DATASET):
        try:
            with open(DATA_DATASET, "rb") as f:
                dados_existentes = pickle.load(f)
        except:
            dados_existentes = []
            
    dados_existentes.extend(memoria_temporaria)
    with open(DATA_DATASET, "wb") as f:
        pickle.dump(dados_existentes, f)
        
    # Salva os Pesos da Rede Neural em PyTorch (.pth)
    torch.save(modelo.state_dict(), ARQUIVO_MODELO)
    print(f"💾 [Rede Neural] Pesos salvos em {ARQUIVO_MODELO} | Exemplos no dataset: {len(dados_existentes)}")
    memoria_temporaria.clear()

atexit.register(salvar_tudo)