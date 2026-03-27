# Z-Fusion

An enhanced Gradio interface for fast image generation powered by a ComfyUI backend.

This project is designed for 1-click installing via [Pinokio](https://pinokio.co/)

---

## Screenshots

<details>
<summary>📸 Click to view screenshots</summary>

<br>

| [⚡ Image](#-image) | [✏️ Edit](#️-edit) | [🔍 Upscale](#-upscale) | [🧪 Experimental](#-experimental) |
|:---:|:---:|:---:|:---:|

---

#### ⚡ Image
<img src="./app/assets/image-tab.png" width="900"/>

#### ✏️ Edit
<img src="./app/assets/edit-tab.png" width="900"/>

#### 🔍 Upscale
<img src="./app/assets/upscale-tab.png" width="900"/>

#### 🧪 Experimental
<img src="./app/assets/experimental-tab.png" width="900"/>

</details>

## Features

- **Z-Image** - Image generation with the Z-Image Turbo and Base models
- **FLUX2 Klein** - Edit and Image gen workflows
- **Txt2Img & Img2Img** - with comfyui node parameters exposed to the UI
- **SeedVR2 4K Upscaler** - High-quality image and video upscaling
- **LLM Prompt Assistants** - Via Joycaption or in-built LLM support
- **LoRA Support** - Apply style/character Z-Image and Klein LoRAs
- **GGUF friendly** - Supports quantized models for low VRAM

## License

MIT License - See [LICENSE](LICENSE) file

### Third-Party
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - GPL-3.0
- [ComfyKit](https://github.com/puke3615/ComfyKit) - MIT
- [Gradio](https://gradio.app/) - Apache-2.0

## Credits

- Built with [Gradio](https://gradio.app/) and [ComfyKit](https://puke3615.github.io/ComfyKit/)
- Powered by [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- GGUF support via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0)
- Upscaling via [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache-2.0)
- Video export via [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (GPL-3.0)
- Seed variance via [SeedVarianceEnhancer](https://github.com/ChangeTheConstants/SeedVarianceEnhancer) (MIT)
- Wildcards based on [comfyui_wildcards](https://github.com/lordgasmic/comfyui_wildcards)
- Camera prompts reference by [CameraPromptsGenerator](https://github.com/demon4932/CameraPromptsGenerator) (MIT)
- Captioning via [JoyCaption](https://github.com/1038lab/ComfyUI-JoyCaption) (GPL-3.0)
