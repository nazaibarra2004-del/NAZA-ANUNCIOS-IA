import streamlit as st
import google.generativeai as genai
import torch
from diffusers import StableDiffusionPipeline
import edge_tts
import asyncio
import os
from moviepy.video.VideoClip import ImageClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# --- 1. CONFIGURACIÓN DE IA (GEMINI) ---
API_KEY = st.secrets["AIzaSyAp_EK4dQwrM_iRAjDcgQtfZpxdLZLQLQw"]
genai.configure(api_key=API_KEY)

# Nombre oficial sin prefijos raros
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. CONFIGURACIÓN DE IMAGEN (PROCESADOR) ---
@st.cache_resource
def cargar_pipeline_imagen():
    # Usamos el modelo estándar que es el más compatible
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    # IMPORTANTE: Quitamos el torch_dtype=torch.float16 porque en CPU da error
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    pipe = pipe.to("cpu") 
    return pipe

# --- 3. FUNCIÓN PARA GENERAR VOZ ---
async def generar_audio(texto, nombre_archivo):
    communicate = edge_tts.Communicate(texto, "es-AR-TomasNeural")
    await communicate.save(nombre_archivo)

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="NAZA.IA", page_icon="📢")
st.title("📢 NAZA.ANUNCIOS.IA")

producto = st.text_input("¿Qué vendemos hoy?", placeholder="Ej: Zapatillas Nike")
descripcion = st.text_area("Describí el producto")

if st.button("🚀 Crear Anuncio Completo"):
    if producto and descripcion:
        try:
            pipeline_img = cargar_pipeline_imagen()
            with st.spinner('Generando anuncio completo...'):
                # --- A. TEXTO ---
                res_texto = model.generate_content(f"Anuncio corto para {producto}: {descripcion}").text
                
                # --- B. IMAGEN MEJORADA ---
                prompt_visual = (
                    f"Professional product photography of {producto}, {descripcion}, "
                    "placed on a city street, cinematic night lighting, neon signs, "
                    "blurred city background, bokeh, 8k resolution, highly detailed."
                )
                
                # Generación real
                resultado_img = pipeline_img(
                    prompt=prompt_visual, 
                    guidance_scale=9.0, 
                    num_inference_steps=30
                )
                imagen_gen = resultado_img.images[0]
                imagen_gen.save("anuncio_img.png")

                # --- C. AUDIO Y VIDEO ---
                asyncio.run(generar_audio(res_texto, "anuncio_voz.mp3"))
                audio_clip = AudioFileClip("anuncio_voz.mp3")
                video_clip = ImageClip("anuncio_img.png").set_duration(audio_clip.duration)
                video_clip = video_clip.set_audio(audio_clip)
                video_clip.write_videofile("anuncio_video.mp4", fps=24, codec="libx264")

                # --- MOSTRAR RESULTADOS ---
                st.success("¡Hecho!")
                st.write(res_texto)
                st.image("anuncio_img.png")
                st.video("anuncio_video.mp4")

        except Exception as e:
            st.error(f"Error: {e}")
        

        
