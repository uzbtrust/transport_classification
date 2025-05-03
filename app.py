import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Transportni klassifikatsiya qiluvchi AI')
# Notify about abilities
st.write('Bu AI transport turlarini (Samalyot, Kema, Avtomobil) klassifikatsiya qiladi')

# rasm joylash
file = st.file_uploader('Rasm yuklang', type=['png', 'jpg', 'jpeg'])

# load model
model = load_learner('transport_model.pkl')

# PIL converter
if file is not None:
    img = PILImage.create(file)
    pred, pred_idx, probs = model.predict(img)
    if pred == 'Car':
        pred = 'Avtomobil'
    elif pred == 'Boat':
        pred = 'Qayiq'
    elif pred == 'Airplane':
        pred = 'Avtobus'
    st.image(img.to_thumb(256, 256), caption=f'Predicted: {pred}')
    st.success(f"Natija: {pred}")
    st.info(f"Ehtimollik: {probs[pred_idx]*100:.1f}%")
    
    # plotly express
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
else:
    st.warning("Iltimos, rasm yuklang.")