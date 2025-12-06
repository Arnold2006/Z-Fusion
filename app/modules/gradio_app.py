import argparse
import gradio as gr
from prompt_assistant import PromptAssistant

# ----------------------------
# Parse CLI Args
# ----------------------------
parser = argparse.ArgumentParser(description="Standalone LLM Prompt Enhancer")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

# Initialize the Assistant logic
assistant = PromptAssistant()

# ----------------------------
# Build UI
# ----------------------------

css = """
textarea {
    overflow-y: auto !important; 
    resize: vertical;
}
"""

with gr.Blocks(css=css, title="Prompt Assistant Standalone") as demo:
    
    gr.Markdown("## 🤖 Universal Prompt Assistant")
    
    with gr.Tabs():
        # --- TAB 1: Main Application Simulation ---
        with gr.TabItem("Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    main_image_input = gr.Image(
                        type="filepath", 
                        label="Input Image",
                        height=300
                    )
                    
                with gr.Column(scale=1):
                    # FIX: Added max_lines to ensure scrollbar appears for long prompts
                    main_prompt_input = gr.Textbox(
                        label="Prompt", 
                        placeholder="Type a prompt here...", 
                        lines=6,
                        max_lines=20
                    )
                    
                    # -----------------------------------------------
                    # INTEGRATION
                    # -----------------------------------------------
                    assistant.render_main_ui(
                        target_textbox=main_prompt_input, 
                        image_input=main_image_input
                    )
                    # -----------------------------------------------

        # --- TAB 2: Settings ---
        with gr.TabItem("LLM Settings"):
            assistant.render_settings_ui()


if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share
    )