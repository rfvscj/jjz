# 这是一个示例界面
import os
import json
import torch
import gradio as gr
from pathlib import Path
from src.law_interface import MyArgs, quicklook, analyze
from src.model import ModelShell


args = MyArgs()
args.device = torch.device("cpu")
shell = ModelShell(args)



sample_list = []
if hasattr(args, "sample_option_file") and os.path.exists(args.sample_option_file):
    with open(args.sample_option_file, 'r', encoding='utf8') as f_sample:
        sample_list = list(json.load(f_sample))

def ql(judgement):
    return quicklook(judgement=judgement)

def pred(judgement):
    return analyze(shell=shell, judgement=judgement)['conclusion']

# created block, added inout uploader button 
def upload_file(file):
    print(file)
    file_path = Path(file.name)
    with open(file_path, 'r', encoding='utf8') as f:
        text = f.readline()
    return text

def on_select(evt: gr.SelectData):
    if evt.index < len(sample_list):
        return sample_list[evt.index]
    else:
        return ""
    
def type_select(evt: gr.SelectData):
    if evt.index == 0:
        return 0
    else:
        return 1


# css_code = 'div{background-image: url("http://jxjs.court.gov.cn/resourcesE/gaofa/201410/281456500jx3.png");}\
#     #sss{background-color:#b0c4de;}'
    
with gr.Blocks() as demo:
    gr.Markdown('# 减假暂智能监督系统')
    # gr.HTML('<img src="http://jxjs.court.gov.cn/resourcesE/gaofa/201410/281456500jx3.png" alt="some_text" style="margin:0 auto;">')
    # gr.HTML('<img src="sources/pic.png" alt="some_text" style="margin:0 auto;">')
    with gr.Tab("减假暂智能监督", elem_id='sss'):
        gr.Markdown("您可以手动输入/粘贴文书，或者选择批量上传（支持txt, json格式），然后点击核对")
        with gr.Row():
            with gr.Column():
                file_output = gr.Textbox(placeholder="输入文本或上传文件...", label="文书输入", lines=3)
                with gr.Row():
                    # sample_option = gr.Dropdown(choices=[str(i + 1) for i, _ in enumerate(sample_list)], label="选择示例")
                    # sample_option.select(on_select, None, file_output)
                    upload_button = gr.UploadButton(label="上传文书", file_types=[".txt",".json", "csv"])
                upload_button.upload(upload_file, upload_button, file_output)
            with gr.Column():
                text_output1 = gr.Textbox(label="案件描述")
                text_output2 = gr.Textbox(label="监狱建议/法院判决")
                file_output.change(ql, file_output, [text_output1, text_output2])
        with gr.Column():
            text_button = gr.Button("核对")
            text_output3 = gr.Textbox(label="核对结果")
            text_button.click(pred, file_output,  text_output3)
    with gr.Tab("others"):
        gr.Markdown("# coming soon")
        
demo.queue().launch(debug=True, server_name='0.0.0.0', server_port=7777)