<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Aurora

## Overview

The AuroraCap model was first proposed in [AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark](https://arxiv.org/abs/) by Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D Manning.

AuroraCap is a video captioning model that is designed to be efficient and performant. It is based on the llava architecture with token merging techique to sigificantly reduce the nunber of visual tokens before fed into the llm decoder. It is trained on a large corpus of video data and achives state-of-the-art performance on multiple image and video captioning benchmarks.

The abstract from the paper is the following:

*Video detailed captioning is a key task which aims to generate comprehensive and coherent textual descriptions of video content, benefiting both video understanding and generation. In this paper, we propose AuroraCap, a video captioner based on a multimodal large language model. We follow the simplest architecture design without additional parameters for temporal modeling. To address the overhead caused by lengthy video sequences, we implement the token merging strategy, reducing the number of input visual tokens. Surprisingly, we found that this strategy results in little performance drop. AuroraCap shows superior performance on various video and image captioning benchmarks, for example getting a CIDEr of 88.9 on Flickr30k, beating GPT-4V (55.3) and Gemini-1.5 Pro (82.2). However, existing video caption benchmarks only include simple descriptions, consisting of a few dozen words, which limits research in this field. Therefore, we develop VDC, a video detailed captioning benchmark with over one thousand carefully annotated structured captions. In addition, we propose a new LLM-assisted metric VDCscore for bettering evaluation, which adopts a divide-and-conquer strategy to transform the evaluation of long captions into multiple short question-answering pairs. With the help of human Elo ranking, our experiments show that this benchmark better correlates with human judgments of video detailed captioning quality.*

This model was contributed by [jongjyh](https://huggingface.co/wchai).
The original code can be found [here](https://github.com/Reself/aurora).

<Tip>

AuroraCap uses token merging technique to reduce the number of visual tokens before fed into the llm decoder. We using `token_kept_ratio` range from 0 to 1 to control the number of visual tokens kept. For example, if `token_kept_ratio` is 0.5, then 50% of the visual tokens will be kept. We recommend to use `token_kept_ratio` in the range of 0.2 to 0.4 for better performance-cost trade-off for captioning tasks, above 0.5 for visual question answering tasks, and above 0.8 for OCR-related tasks.

</Tip>

## Quick Start

### Single Image Inference

```python
from transformers import AuroraForConditionalGeneration, AuroraProcessor
import torch
from PIL import Image
import requests

model_id = "wchai/AuroraCap-7B-IMG"
processor = AuroraProcessor.from_pretrained(model_id)
model = AuroraForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

url = "https://llava-vl.github.io/static/images/view.jpg"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image in detail."},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1024, token_kept_ratio=0.2)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Video Inference

```python
from transformers import AuroraForConditionalGeneration, AuroraProcessor
import torch
from PIL import Image
import requests

import av
import numpy as np
from huggingface_hub import hf_hub_download

model_id = "wchai/AuroraCap-7B-VID"
processor = AuroraProcessor.from_pretrained(model_id)
model = AuroraForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos, up to 32 frames)
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, # we still use image type for video input
            {"type": "text", "text": "Describe the video in detail."},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda:0", torch.float16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1024, token_kept_ratio=0.2)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Model optimization

### Quantization using bitsandbytes

The model can be loaded in 8 or 4 bits, greatly reducing the memory requirements while maintaining the performance of the original model. First make sure to install bitsandbytes, `pip install bitsandbytes` and make sure to have access to a GPU/accelerator that is supported by the library.

<Tip>

bitsandbytes is being refactored to support multiple backends beyond CUDA. Currently, ROCm (AMD GPU) and Intel CPU implementations are mature, with Intel XPU in progress and Apple Silicon support expected by Q4/Q1. For installation instructions and the latest backend updates, visit [this link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend).

We value your feedback to help identify bugs before the full release! Check out [these docs](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends) for more details and feedback links.

</Tip>

Simply change the snippet above with:

```python
from transformers import AuroraForConditionalGeneration, BitsAndBytesConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AuroraForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
```

### Use Flash-Attention 2 to further speed-up generation

First make sure to install flash-attn. Refer to the [original repository of Flash Attention](https://github.com/Dao-AILab/flash-attention) regarding that package installation. Simply change the snippet above with:

```python
from transformers import AuroraForConditionalGeneration

model = AuroraForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    use_flash_attention_2=True
).to(0)
```


## AuroraConfig

[[autodoc]] AuroraConfig

## AuroraProcessor

[[autodoc]] AuroraProcessor

## AuroraImageProcessor

[[autodoc]] AuroraImageProcessor

## AuroraVideoProcessor

[[autodoc]] AuroraVideoProcessor

## AuroraForConditionalGeneration

[[autodoc]] AuroraForConditionalGeneration
    - forward