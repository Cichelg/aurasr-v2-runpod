import spaces
import gradio as gr
from gradio_imageslider import ImageSlider
from PIL import Image
import numpy as np
from aura_sr import AuraSR
import torch

# Force CPU usage
torch.set_default_tensor_type(torch.FloatTensor)

# Override torch.load to always use CPU
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, map_location=torch.device('cpu'))

# Initialize the AuraSR model
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

# Restore original torch.load
torch.load = original_load

@spaces.GPU
def process_image(input_image):
    if input_image is None:
        raise gr.Error("Please provide an image to upscale.")

    # Convert to PIL Image for resizing
    pil_image = Image.fromarray(input_image)

    upscaled_image = aura_sr.upscale_4x(pil_image)

    # Convert result to numpy array if it's not already
    result_array = np.array(upscaled_image)

    print(input_image, result_array)

    return (input_image, result_array)
    
title = """<h1 align="center">AuraSR</h1>
<p><center>Upscales your images to x4</center></p>
<p><center>
<a href="https://huggingface.co/fal/AuraSR-v2" target="_blank">[AuraSR-v2]</a>
<a href="https://blog.fal.ai/introducing-aurasr-an-open-reproduction-of-the-gigagan-upscaler-2/" target="_blank">[Blog Post]</a>
<a href="https://huggingface.co/fal-ai/AuraSR" target="_blank">[v1 Model Page]</a>
</center></p>
<br/>
<p>This is an open reproduction of the GigaGAN Upscaler from fal.ai</p>
"""

with gr.Blocks() as demo:
    
    gr.HTML(title)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy")
            process_btn = gr.Button(value="Upscale Image", variant = "primary")
        with gr.Column(scale=1):
            output_slider = ImageSlider(label="Before / After", type="numpy")

    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=output_slider
    )

    # Add examples
    gr.Examples(
        examples=[
            "image1.png",
            "image3.png"
        ],
        inputs=input_image,
        outputs=output_slider,
        fn=process_image,
        cache_examples=True,
    )

demo.launch(debug=True)