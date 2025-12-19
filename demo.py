import gradio as gr
import soundfile as sf
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from nets.model_wrapper import SeparationModel
from utils.audio_utils import resample
from utils.average_model_params import average_model_params
from utils.config import yaml_to_parser

RESAMPLE_RATE = 48000

# parameters used to plot the spectrogram
n_fft = 512
hop_length = 128
config_path = "pretrained_models/tuss.medium.2-4src/hparams.yaml"
ckpt_paths = [Path("pretrained_models/tuss.medium.2-4src/checkpoints/model.pth")]
# instantiate the model
hparams = yaml_to_parser(config_path)
hparams = hparams.parse_args([])
model = SeparationModel(
    hparams.encoder_name,
    hparams.encoder_conf,
    hparams.decoder_name,
    hparams.decoder_conf,
    hparams.model_name,
    hparams.model_conf,
    hparams.css_conf,
    hparams.variance_normalization,
)
state_dict = average_model_params(ckpt_paths)
new_state_dict = {}
for key, value in state_dict.items():
    k = key.replace("model.", "")
    new_state_dict[k] = value
model.load_state_dict(new_state_dict)
model.cuda()


def apply_model(audio_path,prompts):
    mix, fs = torchaudio.load(audio_path)
    mix = mix[[0],:]
    mix = mix.cuda()
    if RESAMPLE_RATE != fs:
        mix = resample(mix, fs, RESAMPLE_RATE)
    with torch.no_grad():
        y, *_ = model(mix, [prompts])
    if RESAMPLE_RATE != fs:
        y = resample(y, RESAMPLE_RATE, fs)
    return y.cpu()



def plot_fig(data, save_path, title, fs):
    if isinstance(data, torch.Tensor):
        data = data.clone().cpu().numpy()

    # plot spectrogram
    fig, ax = plt.subplots(figsize=(6, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(
        D,
        y_axis="linear",
        x_axis="time",
        sr=fs,
        ax=ax,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    ax.set(title=title)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [hz]")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.subplots_adjust(left=0.125, right=0.95, top=0.88, bottom=0.11)
    plt.savefig(save_path)
    plt.close()


MAX_OUTPUTS = 12


def separate_from_prompts(
    audio_input,
    speech,
    sfx,
    sfxbg,
    drums,
    bass,
    vocals,
    other,
    musicbg
):
    # convert checkboxes (True/False) to ints (1/0)
    sfxbg = int(sfxbg)
    drums = int(drums)
    bass = int(bass)
    vocals = int(vocals)
    other = int(other)
    musicbg = int(musicbg)

    # build prompts list
    prompts = (
        ["speech"] * speech +
        ["sfx"] * sfx +
        ["sfxbg"] * sfxbg +
        ["drums"] * drums +
        ["bass"] * bass +
        ["vocals"] * vocals +
        ["other"] * other +
        ["musicbg"] * musicbg
    )

    # apply model
    sources = apply_model(audio_input, prompts)

    mix, fs = torchaudio.load(audio_input)

    scale = torch.max(torch.abs(mix)) / 0.95
    mix /= scale
    sources /= scale

    # plot mix
    mix_image_path = "tmp/mix.png"
    plot_fig(mix[0], mix_image_path, "Input mixture", fs)
    mix_image = gr.update(value=mix_image_path, visible=True)

    # save sources
    for i, (src, p) in enumerate(zip(sources, prompts)):
        torchaudio.save(f"tmp/{p}{i}.wav", src[None], fs)
        plot_fig(src, f"tmp/{p}{i}.png", p, fs)
        # plot_fig(src, mix_image_path, p)

    # prepare outputs
    returns = []
    num_outputs = len(prompts)

    for i in range(MAX_OUTPUTS):
        if i < num_outputs:
            returns.append(gr.update(
                value=f"tmp/{prompts[i]}{i}.wav",
                visible=True,
                label=prompts[i]
            ))
            returns.append(gr.update(
                value=f"tmp/{prompts[i]}{i}.png",
                visible=True,
                label=prompts[i]
            ))
        else:
            returns.append(gr.update(visible=False))
            returns.append(gr.update(visible=False))

    return returns + [mix_image] + [gr.update(visible=True)]


with gr.Blocks() as demo:

    gr.Markdown("## Upload a sound file")

    audio_input = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Upload Audio"
    )

    mix_image = gr.Image(label="Mix Spectrogram", visible=False)

    gr.Markdown("## Audio Mix Controls")

    with gr.Row():
        speech = gr.Slider(0, 3, step=1, value=0, label="Speech")
        sfx = gr.Slider(0, 3, step=1, value=0, label="SFX")
        sfxbg = gr.Checkbox(label="SFX Background")
        musicbg = gr.Checkbox(label="Music Background")

    with gr.Row():
        drums = gr.Checkbox(label="Drums")
        bass = gr.Checkbox(label="Bass")
        vocals = gr.Checkbox(label="Vocals")
        other = gr.Checkbox(label="Other")

    apply_btn = gr.Button("Apply", variant="primary")

    outputs_md = gr.Markdown("## Separated Outputs", visible=False)

    output_components = []

    for i in range(MAX_OUTPUTS):
        with gr.Row():
            audio = gr.Audio(label=f"Output {i+1}", visible=False)
            image = gr.Image(label=f"Image {i+1}", visible=False)
        output_components.extend([audio, image])

    apply_btn.click(
        separate_from_prompts,
        inputs=[
            audio_input,
            speech,
            sfx,
            sfxbg,
            drums,
            bass,
            vocals,
            other,
            musicbg
        ],
        outputs=output_components + [mix_image] + [outputs_md]
    ).then(
        lambda: gr.update(interactive=True),
        None,
        apply_btn
    )

demo.launch()

