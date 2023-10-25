import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset():
    # reset chatbot, image, text_input, upload_button, chat_state, img_list, img_emb_list, gallery
    return None, \
        gr.update(value=None, interactive=True), \
        gr.update(placeholder='请先上传图片', interactive=False), \
        gr.update(value="上传图片，开始对话", interactive=True), \
        CONV_VISION.copy(), \
        [], \
        [], \
        []


def upload_img(gr_img, chat_state, img_list, img_emb_list):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, img_list, img_emb_list
    img_list.append(gr_img)
    # upload an image to the chat
    chat.upload_img(gr_img, chat_state, img_emb_list)
    # update image, text_input, upload_button, chat_state, gallery, img_emb_list
    return gr.update(value=None, interactive=False), \
        gr.update(interactive=True, placeholder='输入，按回车发送'), \
        gr.update(value="发消息后上传更多图片", interactive=False), \
        chat_state, \
        img_list, \
        img_emb_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='输入不能为空'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    # update chatbot, chat_state, image, upload_button
    return chatbot, \
        chat_state, \
        gr.update(interactive=True), \
        gr.update(value="上传更多图片", interactive=True)


title = """<h1 align="center">MiniGPT 4</h1>"""

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="上传图片开始对话", interactive=True, variant="primary")
            clear = gr.Button("重新开始")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="光束搜索数",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="随机性",
            )

        with gr.Column():
            chat_state = gr.State(CONV_VISION.copy())
            img_list = gr.State([])
            img_emb_list = gr.State([])
            gallery = gr.Gallery(label="已上传图片", show_label=True) \
                .style(rows=[1], object_fit="scale-down", height="500px", preview=True)
            chatbot = gr.Chatbot(label='MiniGPT 4')
            text_input = gr.Textbox(label='用户输入', placeholder='请先上传图片', interactive=False)

    upload_button.click(upload_img, [image, chat_state, img_list, img_emb_list],
                        [image, text_input, upload_button, chat_state, gallery, img_emb_list])

    text_input \
        .submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]) \
        .then(gradio_answer,
              [chatbot, chat_state, img_emb_list, num_beams, temperature],
              [chatbot, chat_state, image, upload_button])

    clear.click(gradio_reset,
                None,
                [chatbot, image, text_input, upload_button, chat_state, img_list, img_emb_list, gallery])

demo.launch(share=False, enable_queue=True, server_name="0.0.0.0", server_port=7861)
