import io
import logging
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # Required arguments
    parser.add_argument('-m', '--model_path', type=str, default="logs/G_23000.pth", help='Path to the model')
    parser.add_argument('-c', '--config_path', type=str, default="configs/config.json", help='Path to the config file')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='List of wav file names located in the raw folder')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='Pitch adjustment in semitones (positive or negative)')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['yunhao'], help='Target speaker name(s) for synthesis')

    # Optional arguments
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False,
                        help='Automatic pitch prediction for voice conversion. Do NOT enable when converting singing as it causes severe pitch issues.')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='Path to the clustering model; fill in any value if clustering is not trained')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='Proportion of the clustering solution, range 0-1; fill in 0 if no clustering model is trained')

    # Rarely changed arguments
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='Silence threshold in dB; default -40, use -30 for noisy audio, -50 to preserve breaths')
    parser.add_argument('-d', '--device', type=str, default=None, help='Inference device; None = auto-select CPU or GPU')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='Noise scale; affects pronunciation clarity and audio quality')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='Seconds of silence padding added to each audio segment to avoid artifacts at start/end')
    parser.add_argument('-wf', '--wav_format', type=str, default='flac', help='Output audio format')

    args = parser.parse_args()

    svc_model = Svc(args.model_path, args.config_path, args.device, args.cluster_model_path)
    infer_tool.mkdir(["raw", "results"])
    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds

    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

        for spk in spk_list:
            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                else:
                    # padd
                    pad_len = int(audio_sr * pad_seconds)
                    data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)
                    out_audio, out_sr = svc_model.infer(spk, tran, raw_path,
                                                        cluster_infer_ratio=cluster_infer_ratio,
                                                        auto_predict_f0=auto_predict_f0,
                                                        noice_scale=noice_scale
                                                        )
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]

                audio.extend(list(infer_tool.pad_array(_audio, length)))
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            res_path = f'./results/{clean_name}_{key}_{spk}{cluster_name}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)

if __name__ == '__main__':
    main()
