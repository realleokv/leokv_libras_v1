import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "libras_platform.db")

STATIC_DATA_PATH = os.path.join(DATA_DIR, "static_samples.npz")
DYNAMIC_DATA_PATH = os.path.join(DATA_DIR, "dynamic_samples.npz")

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_rf_model.pkl")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_lstm_model.pt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Parâmetros de Extração e Sequência Temporal
HAND_FEATURES_DIM = 73        # 1 (presença) + 63 (21*3 coords norm) + 5 (extensão dedos) + 4 (distâncias polegar)
TOTAL_FEATURES_DIM = 146      # Duas mãos: Esquerda (73) + Direita (73)
SEQ_LENGTH = 20               # Número de frames para sinais dinâmicos (LSTM)
DEFAULT_CONFIDENCE_THRESHOLD = 0.72
MOTION_VELOCITY_THRESHOLD = 0.045

# Catálogo de Referência Real da Libras (Alfabeto A-Z + Vocabulário Básico)
DEFAULT_SIGNS_CATALOG = [
    # Alfabeto Manual (Estáticos)
    {"label": "A", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Mão fechada, polegar estendido lateralmente encostado na lateral do indicador."},
    {"label": "B", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Quatro dedos estendidos e unidos para cima; polegar dobrado sobre a palma."},
    {"label": "C", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos curvados formando a letra C, polegar em oposição."},
    {"label": "D", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Indicador estendido para cima; demais dedos curvados tocando a ponta do polegar."},
    {"label": "E", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Todos os dedos curvados em garra fechada, sem tocar a base da palma, polegar recolhido abaixo das pontas."},
    {"label": "F", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Indicador levemente flexionado e polegar cruzado pelo lado de fora do indicador; demais dedos estendidos."},
    {"label": "G", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Indicador e polegar estendidos paralelos apontando para cima, demais dedos fechados."},
    {"label": "I", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Apenas o dedo mínimo estendido para cima; demais dedos fechados."},
    {"label": "L", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Indicador estendido para cima e polegar estendido lateralmente em 90 graus."},
    {"label": "M", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos indicador, médio e anelar estendidos apontando para baixo (ou sobre o polegar)."},
    {"label": "N", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos indicador e médio estendidos apontando para baixo (ou sobre o polegar)."},
    {"label": "O", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Pontas de todos os dedos unidas ao polegar formando um círculo."},
    {"label": "P", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Configuração similar ao K/H, mas orientada horizontalmente apontando para frente/baixo."},
    {"label": "Q", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Configuração de G invertida, com indicador e polegar apontando para baixo."},
    {"label": "R", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos indicador e médio estendidos e cruzados."},
    {"label": "S", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Mão fechada em punho com o polegar cruzado por cima dos dedos médios."},
    {"label": "T", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": " Semelhante ao F, porém o polegar fica posicionado por dentro (entre o indicador flexionado e o dedo médio)."},
    {"label": "U", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos indicador e médio estendidos e unidos para cima."},
    {"label": "V", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos indicador e médio estendidos e separados formando um V."},
    {"label": "W", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Dedos indicador, médio e anelar estendidos e separados."},
    {"label": "Y", "category": "alfabeto", "is_dynamic": 0, "two_hands": 0, "description": "Polegar e dedo mínimo estendidos; dedos centrais fechados."},

    # Alfabeto Manual (Dinâmicos - Exigem Movimento)
    {"label": "H", "category": "alfabeto", "is_dynamic": 1, "two_hands": 0, "description": "Indicador e médio estendidos com polegar entre eles, realizando giro de punho."},
    {"label": "J", "category": "alfabeto", "is_dynamic": 1, "two_hands": 0, "description": "Mão em configuração de I desenhando a curva da letra J no ar."},
    {"label": "K", "category": "alfabeto", "is_dynamic": 1, "two_hands": 0, "description": "Indicador para cima, médio inclinado à frente e polegar entre eles, com movimento curto para cima."},
    {"label": "X", "category": "alfabeto", "is_dynamic": 1, "two_hands": 0, "description": "Indicador curvado em gancho puxando o movimento para trás."},
    {"label": "Z", "category": "alfabeto", "is_dynamic": 1, "two_hands": 0, "description": "Indicador estendido desenhando o traçado da letra Z no ar."},

    # Palavras e Expressões Básicas Reais de Libras
    {"label": "Oi", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Configuração de mão passando de O para I com leve movimento lateral."},
    {"label": "Olá", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Mão aberta em configuração B realizando aceno lateral."},
    {"label": "Bom dia", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Sinal composto: BOM (mão fechando/abrindo diante da boca) + DIA (configuração D saindo da têmpora)."},
    {"label": "Boa tarde", "category": "palavra", "is_dynamic": 1, "two_hands": 1, "description": "Sinal composto: BOM + TARDE (mão plana descendo diagonalmente à frente do antebraço)."},
    {"label": "Boa noite", "category": "palavra", "is_dynamic": 1, "two_hands": 1, "description": "Sinal composto: BOM + NOITE (mão deslizando sobre o dorso da outra mão curvada)."},
    {"label": "Tudo bem?", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Sinal de BOM diante da boca seguido de configuração de polegar estendido (positivo)."},
    {"label": "Meu nome", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Mão aberta tocando o peito (MEU) seguida de mão em U movendo horizontalmente (NOME)."},
    {"label": "Obrigado", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Mão plana tocando a testa e movendo-se para frente."},
    {"label": "Por favor", "category": "palavra", "is_dynamic": 1, "two_hands": 1, "description": "Mãos unidas pelas palmas ou mão passando pelo peito em gesto de pedido."},
    {"label": "Desculpa", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Mão em configuração Y tocando a região do queixo."},
    {"label": "Sim", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Mão em configuração S realizando movimento vertical afirmativo."},
    {"label": "Não", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Indicador estendido balançando lateralmente em negação."},
    {"label": "Ajuda", "category": "palavra", "is_dynamic": 1, "two_hands": 1, "description": "Uma mão apoiada sobre a palma da outra mão aberta, movendo ambas para frente."},
    {"label": "Água", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Mão em configuração L com polegar no queixo e indicador balançando."},
    {"label": "Comida", "category": "palavra", "is_dynamic": 1, "two_hands": 0, "description": "Pontas dos dedos unidas aproximando-se repetidamente da boca."},
    {"label": "Escola", "category": "palavra", "is_dynamic": 1, "two_hands": 1, "description": "Sinal composto: CASA (telhado com duas mãos) + ESTUDAR (duas mãos em B batendo palmas)."},
    {"label": "Casa", "category": "palavra", "is_dynamic": 0, "two_hands": 1, "description": "Duas mãos abertas inclinadas unindo as pontas dos dedos formando um telhado."},
    {"label": "Eu", "category": "palavra", "is_dynamic": 0, "two_hands": 0, "description": "Dedo indicador apontando para o próprio peito."},
    {"label": "Você", "category": "palavra", "is_dynamic": 0, "two_hands": 0, "description": "Dedo indicador apontando para frente na direção do interlocutor."}
]