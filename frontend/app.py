import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


def analyze(video_path):
    if not video_path:
        return "Please select a video.", None

    try:
        with open(video_path, "rb") as f:
            resp = requests.post(
                f"{API_URL}/predict",
                files={"file": (os.path.basename(video_path), f, "video/mp4")},
                timeout=180,
            )
    except requests.RequestException as e:
        return f"Unable to connect to the API ({API_URL}): {e}", None

    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}", None

    data = resp.json()
    p = data["fake_probability"]

    summary = (
        f"## Prediction: **{data['label']}**\n\n"
        f"- Fake probability: **{p * 100:.2f}%**\n"
        f"- Decision threshold: {data['threshold']}\n"
        f"- Frames analyzed: {data['num_frames_used']}\n"
        f"- Inference time: {data['inference_time_ms']:.0f} ms\n"
    )
    return summary, {"REAL": 1.0 - p, "FAKE": p}


with gr.Blocks(title="Deepfake Video Detection") as demo:
    gr.Markdown(
        "# Deepfake Video Detection\n"
        "Upload a video to determine whether it has been manipulated using deepfake techniques."
    )

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input Video", sources=["upload"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            result_md = gr.Markdown()
            label_out = gr.Label(label="Confidence", num_top_classes=2)

    btn.click(analyze, inputs=video_in, outputs=[result_md, label_out])

    gr.Markdown(
        "> The results are for reference only. The model was trained on the DFD dataset "
        "and may perform less accurately on videos outside its training distribution."
    )


if __name__ == "__main__":
    demo.launch(server_port=7860)
