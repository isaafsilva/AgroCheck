
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------
# CONFIGURAÇÕES
# -----------------------------------------

st.set_page_config(
    page_title="AgroCheck | Inteligência Agrícola",
    page_icon="🌱",
    layout="centered"
)

# -----------------------------------------
# ESTILO VISUAL
# -----------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8f3;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header, #MainMenu, footer {
    visibility: hidden;
}

/* Cabeçalho */

.hero {
    background: linear-gradient(135deg, #123d2a, #23734a);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(18, 61, 42, 0.13);
}

.hero-label {
    color: #b8e4c6;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin: 10px 0;
    font-weight: 800;
}

.hero p {
    color: #e0eee4;
    font-size: 15px;
    margin-bottom: 0;
}

/* Seções */

.section-title {
    color: #173f2c;
    font-size: 21px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-description {
    color: #718078;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Upload */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #a8c9b1;
    border-radius: 16px;
    padding: 15px;
}

/* Botão */

.stButton > button {
    background: #23734a;
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-weight: 700;
    font-size: 16px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #174e32;
    color: white;
    border: none;
}

/* Resultado */

.result-card {
    background: white;
    border: 1px solid #e2ebe3;
    border-radius: 18px;
    padding: 25px;
    margin-top: 12px;
    box-shadow: 0 5px 18px rgba(30, 70, 45, 0.05);
    min-height: 115px;
}

.result-label {
    color: #6c8875;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-name {
    color: #174e32;
    font-size: 24px;
    font-weight: 800;
    margin-top: 8px;
}

.result-percent {
    color: #23734a;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

/* Probabilidades */

.probability-item {
    background: white;
    border: 1px solid #e2ebe3;
    border-radius: 14px;
    padding: 17px 20px;
    margin: 12px 0;
}

.probability-item.top-item {
    background: #f0f7f0;
    border: 1px solid #b8d7be;
}

.probability-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.probability-name {
    color: #294b36;
    font-size: 14px;
    font-weight: 600;
}

.top-item .probability-name {
    color: #174e32;
    font-weight: 800;
}

.probability-value {
    color: #174e32;
    font-size: 15px;
    font-weight: 800;
    white-space: nowrap;
}

.probability-tag {
    display: inline-block;
    background: #dcefe0;
    color: #23734a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
    padding: 4px 9px;
    border-radius: 20px;
    margin-left: 7px;
}

.probability-bar {
    width: 100%;
    height: 10px;
    background: #e8efe8;
    border-radius: 20px;
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #80bd8e, #23734a);
}

.top-item .probability-fill {
    background: linear-gradient(90deg, #48a76b, #145b35);
}

/* Aviso */

.notice {
    background: #edf4ed;
    border-left: 4px solid #43865a;
    padding: 16px;
    border-radius: 8px;
    color: #496451;
    font-size: 13px;
    margin-top: 25px;
}

/* Rodapé */

.footer {
    text-align: center;
    color: #91a197;
    font-size: 12px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# CARREGAR MODELO
# -----------------------------------------

@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model("agrocheck_melhor.keras")


with st.spinner("Preparando o AgroCheck..."):
    modelo = carregar_modelo()


# -----------------------------------------
# CLASSES
# -----------------------------------------

classes = [
    "Milho saudável",
    "Ferrugem comum",
    "Mancha cinzenta",
    "Helmintosporiose"
]


# -----------------------------------------
# CABEÇALHO
# -----------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-label">Inteligência artificial no campo</div>
    <h1>🌱 AgroCheck</h1>
    <p>
        Tecnologia e inteligência artificial para auxiliar
        na identificação de doenças em folhas de milho.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------
# ÁREA DE ANÁLISE
# -----------------------------------------

st.markdown(
    '<div class="section-title">Análise de imagem</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Envie uma fotografia da folha de milho para iniciar a análise.'
    '</div>',
    unsafe_allow_html=True
)

imagem = st.file_uploader(
    "Selecione uma imagem",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceitos: JPG, JPEG e PNG."
)

if imagem is not None:

    foto = Image.open(imagem).convert("RGB")

    arquivo_atual = (imagem.name, imagem.size)

    if st.session_state.get("arquivo_analisado") != arquivo_atual:
        st.session_state.pop("resultado_analise", None)

    st.image(
        foto,
        caption="Imagem selecionada para análise",
        use_container_width=True
    )

    if st.button("🔍  Analisar folha", use_container_width=True):

        with st.spinner("A inteligência artificial está analisando a imagem..."):

            imagem_processada = foto.resize((224, 224))

            imagem_array = np.array(
                imagem_processada,
                dtype=np.float32
            )

            imagem_array = np.expand_dims(
                imagem_array,
                axis=0
            )

            previsoes = modelo.predict(
                imagem_array,
                verbose=0
            )[0]

            ranking = np.argsort(previsoes)[::-1]

            indice = int(ranking[0])
            confianca = float(previsoes[indice])

            st.session_state.resultado_analise = {
                "previsoes": previsoes,
                "indice": indice,
                "confianca": confianca
            }

            st.session_state.arquivo_analisado = arquivo_atual


# -----------------------------------------
# EXIBIR RESULTADO
# -----------------------------------------

if "resultado_analise" in st.session_state:

    resultado = st.session_state.resultado_analise

    previsoes = resultado["previsoes"]
    indice = resultado["indice"]
    confianca = resultado["confianca"]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Resultado da análise</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="section-description">
        Abaixo estão os resultados gerados pelo modelo de inteligência artificial.
    </div>
    """, unsafe_allow_html=True)

    st.success("Análise concluída!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Possível identificação</div>
            <div class="result-name">🍃 {classes[indice]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Pontuação do modelo</div>
            <div class="result-percent">{confianca:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------
    # PROBABILIDADES
    # -------------------------------------


    st.markdown(
        '<div class="section-title">📊 Probabilidade por categoria</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Pontuações geradas pelo modelo para cada categoria analisada.'
        '</div>',
        unsafe_allow_html=True
    )

    ranking = np.argsort(previsoes)[::-1]

    dados = {
        "Categoria": [classes[i] for i in ranking],
        "Pontuação": [
            f"{float(previsoes[i]) * 100:.2f}%"
            for i in ranking
        ]
    }

    st.dataframe(
        dados,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Categoria": st.column_config.TextColumn(
                "Categoria"
            ),
            "Pontuação": st.column_config.TextColumn(
                "Pontuação",
                help="Pontuação atribuída pelo modelo"
            )
        }
    )


    # -------------------------------------
    # AVISO
    # -------------------------------------

    st.markdown("""
    <div class="notice">
        <strong>⚠️ Importante</strong><br><br>
        Esta análise é uma estimativa gerada por inteligência artificial,
        baseada em imagens utilizadas no treinamento do modelo.
        O resultado não substitui a avaliação de um profissional agrônomo.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------
# RODAPÉ
# -----------------------------------------

st.markdown("""
<div class="footer">
    AGROCHECK • Inteligência artificial aplicada à agricultura
</div>
""", unsafe_allow_html=True)
