import io
import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

config_path = "configs/config.json"

model = Svc("logs/44k/G_114400.pth", "configs/config.json", cluster_model_path="logs/44k/kmeans_10000.pt")



def vc_fn(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    duration = audio.shape[0] / sampling_rate
    if duration > 90:
        return "Please upload audio shorter than 90 seconds. For longer audio, run inference locally.", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = "temp.wav"
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    print(cluster_ratio, auto_f0, noise_scale)
    _audio = model.slice_inference(out_wav_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale)
    return "Success", (44100, _audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Voice Conversion"):
            gr.Markdown(value="""
                ## so-vits-svc 4.0 v2 — Online Demo

                Pre-trained model demo. Available speakers: yunhao, jishuang, huiyu, nen, paimon
                """)
            spks = list(model.spk2id.keys())
            sid = gr.Dropdown(label="Target Speaker", choices=spks, value=spks[0])
            vc_input3 = gr.Audio(label="Upload Audio (max 90 seconds)")
            vc_transform = gr.Number(
                label="Pitch Shift (semitones, positive = higher, negative = lower; +12 = one octave up)",
                value=0
            )
            cluster_ratio = gr.Slider(
                label="Clustering Mix Ratio (0 = off, 1 = full clustering; improves timbre similarity but may reduce clarity)",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.0
            )
            auto_f0 = gr.Checkbox(
                label="Auto F0 Prediction (improves speech conversion; do NOT enable for singing — causes severe pitch issues)",
                value=False
            )
            slice_db = gr.Slider(
                label="Silence Threshold (dB) (default -40; use -30 for noisy audio, -50 to preserve breaths)",
                minimum=-60,
                maximum=-20,
                step=1,
                value=-40
            )
            noise_scale = gr.Slider(
                label="Noise Scale (affects pronunciation clarity and audio quality; recommended: 0.4)",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.4
            )
            vc_submit = gr.Button("Convert", variant="primary")
            vc_output1 = gr.Textbox(label="Status")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale], [vc_output1, vc_output2])

    app.launch()

