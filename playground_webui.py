from config import global_config
import os
import uuid

save_preference_dir = "./configs/user_preference.json"
global_config.now = "infer_service"
infer_config = global_config.infer_service_config

os.environ["RWKV_HEAD_SIZE_A"] = str(infer_config.model.head_size)
os.environ["RWKV_CTXLEN"] = str(infer_config.model.ctx_len)

import gradio as gr
from utils.webui.webui_infer import InferAgent, update_plot
from utils.webui.webui_labeling import create_labeling_tab
from utils.webui.webui_train import TrainAgent
from utils.webui.webui_api import (
    api_base_input_list,
    api_key_input_list,
    api_model_input_list,
    api_chat,
    api_add,
    api_back,
    api_infer,
    api_listen,
    api_regenerate,
    api_reset,
    save_api_hist,
    load_api_hist,
)
from utils.webui.webui_utils import save_user_preference,load_user_preference
from utils.webui.webui_benchmark import (
    run_benchmark,
    stop_benchmark,
    run_mmlu,
    run_humaneval,
)
import json
from functools import partial
import matplotlib.pyplot as plt


agent = InferAgent(
    infer_server="http://{host}:{port}".format(
        host=global_config.server_config.infer.host,
        port=global_config.server_config.infer.port,
    )
)
train_agent = TrainAgent(
    train_server="http://{host}:{port}".format(
        host=global_config.server_config.train.host,
        port=global_config.server_config.train.port_begin,
    )
)

MODEL_DIR = global_config.ckpt_dir  # 替换为实际模型文件夹路径
model_files = []
history_files = []
ckpt_dirs = []
mmlu_folders = []
bmk_question_dirs = []
api_hist_dirs = []
for root, dirs, files in os.walk(MODEL_DIR):
    for file in files:
        if file.endswith(".pth"):
            model_files.append(os.path.join(root, file))
        elif file.endswith(".state"):
            history_files.append(os.path.join(root, file))
    if "now.state" in files and "args.json" in files:
        for sub_dir in dirs:
            sub_dir_path = os.path.join(root, sub_dir)
            if "conversations_hist.json" in os.listdir(sub_dir_path):
                ckpt_dirs.append(root)
                break

benchmark_DIR = global_config.webui.benchmark.mmlu_dir
try:
    for f in os.listdir(benchmark_DIR):
        item_path = os.path.join(benchmark_DIR, f)
        if os.path.isdir(item_path):
            mmlu_folders.append(item_path)
except:
    print("waring: mmlu数据集所在目录 (config.webui.benchmark.mmlu_dir)不存在！请在configs/webui.json中设置。")
bmk_question_DIR = global_config.webui.benchmark.questions_dir
try:
    for root, dirs, files in os.walk(bmk_question_DIR):
        for file in files:
            if file.endswith(".json"):
                bmk_question_dirs.append(os.path.join(root, file))
except:
    print("waring: 问题集所在目录 (config.webui.benchmark.questions_dir)不存在！请在configs/webui.json中设置。")
api_hist_DIR = global_config.webui.api_hist_dir
try:
    for root, dirs, files in os.walk(api_hist_DIR):
        for file in files:
            if file.endswith(".json"):
                api_hist_dirs.append(os.path.join(root, file))
except:
    print("waring: api前端保存历史记录所在目录 (config.webui.api_hist_dir)不存在！请在configs/webui.json中设置。")

# Gradio界面
with gr.Blocks() as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("聊天"):
            infer_setting_selector = gr.Dropdown(
                label="配置",
                choices=[
                    "加载模型",
                    "加载存档点",
                    "加载state",
                    "更新参数",
                    "读取初始提示词",
                ],
                value="更新参数",
            )
            with gr.Row(visible=False) as load_model_section:
                model_dropdown = gr.Dropdown(
                    label="模型路径", choices=model_files, scale=4
                )
                with gr.Column():
                    load_model_btn = gr.Button("读取模型并重置", scale=1)
                    refresh_model_btn = gr.Button("刷新列表", scale=1)

            with gr.Row(visible=False) as load_state_section:
                history_dropdown = gr.Dropdown(
                    label="模型历史", choices=history_files, scale=4
                )
                with gr.Column():
                    load_history_btn = gr.Button("读取状态文件并重置")
                    refresh_history_btn = gr.Button("刷新列表")
            with gr.Row(visible=False) as load_svpt_section:
                ckpt_dropdown = gr.Dropdown(label="存档点", choices=ckpt_dirs, scale=4)
                with gr.Column():
                    load_ckpt_btn = gr.Button("读取存档点并重置")
                    refresh_ckpt_btn = gr.Button("刷新列表")
            with gr.Row(visible=True) as update_param_section:
                with gr.Column():
                    sender_name_input = gr.Textbox(label="用户名字", value=agent.sender)
                    replier_name_input = gr.Textbox(label="AI名字", value=agent.replier)
                with gr.Column():
                    usr_sp_token_dropdown = gr.Dropdown(
                        label="用户 special tokens",
                        choices=[
                            "conversation",
                            "text",
                            "system",
                            "think",
                            "response",
                            "rwkv_legacy_eos",
                        ],
                        value=agent.chatbot.usr_sp_token_role,
                    )
                    bot_sp_token_dropdown = gr.Dropdown(
                        label="AI special tokens",
                        choices=[
                            "conversation",
                            "text",
                            "system",
                            "think",
                            "response",
                            "rwkv_legacy_eos",
                        ],
                        value=agent.chatbot.bot_sp_token_role,
                    )
                with gr.Column():
                    temp_input = gr.Number(label="温度", value=1.0)
                    top_p_input = gr.Number(label="Top P", value=0.7)
                with gr.Column():
                    presence_penalty_input = gr.Number(label="历史惩罚", value=0.2)
                    frequency_penalty_input = gr.Number(label="频率惩罚", value=0.2)
                with gr.Column():
                    decay_penalty_input = gr.Number(label="惩罚衰减", value=0.9961)
                update_params_btn = gr.Button("更新参数")

            with gr.Row(visible=False) as load_init_prompt_section:
                initial_setting_input = gr.Textbox(
                    label="初始设定", placeholder="输入初始设定...", lines=2, scale=4
                )
                with gr.Column():
                    special_token_dropdown = gr.Dropdown(
                        label="设定special token",
                        choices=[
                            "conversation",
                            "text",
                            "system",
                            "response",
                            "think",
                            "rwkv_legacy_eos",
                        ],
                        value="system",
                    )
                    reset_and_load_btn = gr.Button("重置并加载设定")

            infer_setting_selector.change(
                lambda x: (
                    gr.update(visible=x == "加载模型"),
                    gr.update(visible=x == "加载存档点"),
                    gr.update(visible=x == "加载state"),
                    gr.update(visible=x == "更新参数"),
                    gr.update(visible=x == "读取初始提示词"),
                ),
                inputs=infer_setting_selector,
                outputs=[
                    load_model_section,
                    load_svpt_section,
                    load_state_section,
                    update_param_section,
                    load_init_prompt_section,
                ],
            )

            chat_operation_output = gr.Textbox(
                label="操作提示", lines=1, interactive=False
            )

            with gr.Row():
                with gr.Column(scale=6):
                    # chat_output = gr.Textbox(
                    #     label="聊天", lines=17, interactive=False, scale=6
                    # )
                    output_chatbot = gr.Chatbot(label="聊天", scale=6, type="messages")
                    with gr.Row():
                        time_toggle = gr.Checkbox(label="时间", value=False)
                        auto_think_toggle = gr.Checkbox(label="自动思考", value=False)
                        save_ckpt_btn = gr.Button("保存存档点")
                        reset_btn = gr.Button("重置")
                    with gr.Row():
                        regenerate_btn = gr.Button("重新生成")
                        back_btn = gr.Button("删除该轮对话")
                        listen_btn = gr.Button("等待回复")
                        think_btn = gr.Button("等待思考")
                with gr.Column():
                    in_ppl_plot = gr.Plot(label="困惑(输入)")
                    out_ppl_plot = gr.Plot(label="困惑(输出)")

            with gr.Row():
                format_toggle = gr.Checkbox(label="按格式回复", value=False)
                format_input = gr.Textbox(
                    label="回复格式字符串",
                    value='```json\n    "action":$100->45,279$示例文本```$100->65535->11$',
                    placeholder="固定字符串<选项1/选项2/……>$a->b,c,d->x,y,z$ a为最大生成长度，bcd为检测停止token号码，xyz为最后补充的token。如$512->65535->11$",
                    scale=6,
                    visible=False,
                )

                format_toggle.change(
                    lambda x: gr.update(visible=x),
                    format_toggle,
                    format_input,
                )

                ban_input = gr.Textbox(
                    label="禁止的tokens", placeholder="例如：1, 2, 3, 4", scale=2
                )
            with gr.Row():
                message_input = gr.Textbox(
                    label="输入消息", placeholder="输入消息...", scale=9
                )
                with gr.Column(scale=1):
                    send_btn = gr.Button("发送")
                    regenerate_btn2 = gr.Button("重新生成")
                max_resp_len_input = gr.Number(label="最大回复长度", value=512, scale=2)

            with gr.Row():
                special_token = gr.Dropdown(
                    label="special token",
                    choices=[
                        "conversation",
                        "text",
                        "system",
                        "response",
                        "think",
                        "rwkv_legacy_eos",
                    ],
                    value="conversation",
                )
                add_message_input = gr.Textbox(
                    label="添加消息",
                    placeholder="输入要添加的消息...例如“AAA: BBB”",
                    scale=6,
                )
                add_btn = gr.Button("添加")

            with gr.Row():
                save_hist_btn = gr.Button("保存数据集")
                save_rl_btn = gr.Button("保存生成历史(强化学习)")

            # 定义交互逻辑
            send_btn.click(
                agent.send_message,
                (
                    message_input,
                    max_resp_len_input,
                    format_toggle,
                    format_input,
                    ban_input,
                ),
                (output_chatbot, message_input, in_ppl_plot, out_ppl_plot),
            )
            message_input.submit(
                agent.send_message,
                (
                    message_input,
                    max_resp_len_input,
                    format_toggle,
                    format_input,
                    ban_input,
                ),
                (output_chatbot, message_input, in_ppl_plot, out_ppl_plot),
            )
            time_toggle.change(agent.toggle_time, time_toggle, chat_operation_output)
            regenerate_btn.click(
                agent.regenerate,
                (max_resp_len_input, format_toggle, format_input, ban_input),
                (output_chatbot, in_ppl_plot, out_ppl_plot),
            )
            regenerate_btn2.click(
                agent.regenerate,
                (max_resp_len_input, format_toggle, format_input, ban_input),
                (output_chatbot, in_ppl_plot, out_ppl_plot),
            )
            reset_btn.click(agent.reset, None, (chat_operation_output, output_chatbot))
            listen_btn.click(
                agent.listen,
                (max_resp_len_input, format_toggle, format_input, ban_input),
                (output_chatbot, in_ppl_plot, out_ppl_plot),
            )
            think_btn.click(
                agent.think,
                (max_resp_len_input),
                (output_chatbot, in_ppl_plot, out_ppl_plot),
            )
            back_btn.click(
                agent.back_to_last, None, (chat_operation_output, output_chatbot)
            )
            save_hist_btn.click(agent.save_as_dataset, None, chat_operation_output)
            save_rl_btn.click(agent.save_as_rl_pairs, None, chat_operation_output)

            load_model_btn.click(
                agent.load_model, model_dropdown, chat_operation_output
            )
            load_history_btn.click(
                agent.load_history_and_reset, history_dropdown, chat_operation_output
            )
            load_ckpt_btn.click(
                agent.load_checkpoint, ckpt_dropdown, chat_operation_output
            )

            def refresh_model_files():
                global model_files
                model_files = []
                for root, dirs, files in os.walk(MODEL_DIR):
                    for file in files:
                        if file.endswith(".pth"):
                            model_files.append(os.path.join(root, file))
                return gr.update(choices=model_files)

            def refresh_history_files():
                global history_files
                history_files = []
                for root, dirs, files in os.walk(MODEL_DIR):
                    for file in files:
                        if file.endswith(".state"):
                            history_files.append(os.path.join(root, file))
                return gr.update(choices=history_files)

            def refresh_ckpt_dirs():
                global ckpt_dirs
                ckpt_dirs = []
                for root, dirs, files in os.walk(MODEL_DIR):
                    if "now.state" in files and "args.json" in files:
                        for sub_dir in dirs:
                            sub_dir_path = os.path.join(root, sub_dir)
                            if "conversations_hist.json" in os.listdir(sub_dir_path):
                                ckpt_dirs.append(root)
                                break
                return gr.update(choices=ckpt_dirs)

            refresh_model_btn.click(refresh_model_files, None, model_dropdown)
            refresh_history_btn.click(refresh_history_files, None, history_dropdown)
            refresh_ckpt_btn.click(refresh_ckpt_dirs, None, ckpt_dropdown)

            update_params_btn.click(
                lambda sender, replier, temp, top_p, presence_penalty, frequency_penalty, decay_penalty, usr_sp_token, bot_sp_token: agent.update_chatbot_params(
                    sender,
                    replier,
                    temp,
                    top_p,
                    presence_penalty,
                    frequency_penalty,
                    decay_penalty,
                    usr_sp_token,
                    bot_sp_token,
                ),
                [
                    sender_name_input,
                    replier_name_input,
                    temp_input,
                    top_p_input,
                    presence_penalty_input,
                    frequency_penalty_input,
                    decay_penalty_input,
                    usr_sp_token_dropdown,
                    bot_sp_token_dropdown,
                ],
                None,
            )

            add_btn.click(
                agent.add_custom_message,
                [special_token, add_message_input],
                output_chatbot,
            )
            save_ckpt_btn.click(
                lambda: agent.save_checkpoint(MODEL_DIR), None, chat_operation_output
            )

            def reset_and_load_setting(special_token, initial_setting):
                agent.reset()
                agent.add_custom_message(special_token, initial_setting)
                return "已重置并加载设定。"

            reset_and_load_btn.click(
                reset_and_load_setting,
                [special_token_dropdown, initial_setting_input],
                chat_operation_output,
            )

        with gr.Tab("标注"):
            create_labeling_tab()
        with gr.Tab("评估"):
            with gr.Row():
                bmk_temp_input = gr.Number(label="温度", value=0.2)
                bmk_top_p_input = gr.Number(label="Top P", value=0.2)
                bmk_presence_penalty_input = gr.Number(label="重现惩罚", value=0.2)
                bmk_frequency_penalty_input = gr.Number(label="频率惩罚", value=0.2)
                bmk_penalty_decay_input = gr.Number(label="惩罚衰减", value=0.9961)
            bmk_using_init_state_checkbox = gr.Checkbox(
                label="加载初始state", value=False
            )
            bmk_begin_with_state_dir_input = gr.Textbox(
                label="初始state路径",
                value="",
                interactive=True,
                visible=False,
            )
            eval_mode_selector = gr.Dropdown(
                choices=["自定义问题", "选择题", "代码"],
                label="选择评估类型",
                value="自定义问题",
            )
            bmk_using_init_state_checkbox.change(
                lambda x: gr.update(visible=x),
                bmk_using_init_state_checkbox,
                bmk_begin_with_state_dir_input,
            )
            with gr.Column(visible=True) as free_chat_section:
                question_list = []
                fold_questions_checkbox = gr.Checkbox(label="折叠问题", value=True)
                with gr.Row():
                    question_dropdown = gr.Dropdown(
                        label="问题集路径", choices=bmk_question_dirs, scale=4
                    )
                    with gr.Column():
                        load_question_btn = gr.Button("从文件读取问题集", scale=1)
                        refresh_question_btn = gr.Button("刷新列表", scale=1)

                with gr.Column(visible=False) as free_questions_section:
                    question_num_input = gr.Number(
                        label="问题数目", value=len(question_list), scale=6
                    )
                    add_question_button = gr.Button("+")

                    def add_question(question_list):
                        unique_id = str(uuid.uuid4())
                        question_list.append(
                            {
                                "id": unique_id,
                                "quesetion_role": "conversation",
                                "quesetion_text": "",
                                "response_role": "conversation",
                                "response_text_prefix": "",
                            }
                        )
                        return len(question_list)

                    def update_question_list(question_list: list, q_num: int):
                        # 确保 question_list 的长度与 q_num 一致
                        if q_num > len(question_list):
                            # 添加新的问题
                            for _ in range(q_num - len(question_list)):
                                unique_id = str(uuid.uuid4())
                                question_list.append(
                                    {
                                        "id": unique_id,
                                        "quesetion_role": "conversation",
                                        "quesetion_text": "",
                                        "response_role": "conversation",
                                        "response_text_prefix": "",
                                    }
                                )
                        elif q_num < len(question_list):
                            # 截断多余的元素
                            question_list[:] = question_list[:q_num]
                        return gr.update()

                    def find_question(unique_id: str):
                        ql = [q for q in question_list if q["id"] == unique_id]
                        return ql[0] if ql else None

                    def delete_question(unique_id: str):
                        question_list.remove(find_question(unique_id))
                        return len(question_list)

                    def edit_question(
                        unique_id: str,
                        role: str,
                        content: str,
                        response_role: str,
                        response_content: str,
                    ):
                        question = find_question(unique_id)
                        question["quesetion_role"] = role
                        question["quesetion_text"] = content
                        question["response_role"] = response_role
                        question["response_text_prefix"] = response_content

                    def refresh_questions_files_list():
                        global bmk_question_dirs
                        bmk_question_dirs = []
                        for root, dirs, files in os.walk(bmk_question_DIR):
                            for file in files:
                                if file.endswith(".json"):
                                    bmk_question_dirs.append(os.path.join(root, file))
                        return gr.update(choices=bmk_question_dirs)

                    def load_questions_from_file(questions_dir: str):
                        global question_list
                        with open(questions_dir, "r", encoding="utf-8") as file:
                            question_list[:] = json.load(file)
                        return len(question_list)

                    load_question_btn.click(
                        load_questions_from_file,
                        inputs=[question_dropdown],
                        outputs=[question_num_input],
                    )
                    refresh_question_btn.click(
                        refresh_questions_files_list, outputs=[question_dropdown]
                    )

                    add_question_button.click(
                        partial(add_question, question_list), outputs=question_num_input
                    )

                    @gr.render(inputs=question_num_input)
                    def render_questions(q_num):
                        update_question_list(
                            question_list, q_num
                        )  # 根据问题数更新问题列表
                        if q_num == 0:
                            return gr.Markdown(
                                "<span style='color: gray; font-style: italic;'>点击\"+\"以添加一个问题</span>"
                            )
                        else:
                            for i in range(q_num):
                                with gr.Row() as row:
                                    with gr.Column(scale=6):
                                        with gr.Row():
                                            r_dd = gr.Dropdown(
                                                label="提问者 special tokens",
                                                choices=[
                                                    "conversation",
                                                    "text",
                                                    "system",
                                                    "think",
                                                    "response",
                                                    "rwkv_legacy_eos",
                                                ],
                                                value=question_list[i][
                                                    "quesetion_role"
                                                ],
                                                interactive=True,
                                            )
                                            c_tb = gr.Textbox(
                                                value=question_list[i][
                                                    "quesetion_text"
                                                ],
                                                label="提问者 Text",
                                                scale=4,
                                            )
                                        with gr.Row() as row:
                                            rr_dd = gr.Dropdown(
                                                label="回复者 special tokens",
                                                choices=[
                                                    "conversation",
                                                    "text",
                                                    "system",
                                                    "think",
                                                    "response",
                                                    "rwkv_legacy_eos",
                                                ],
                                                value=question_list[i]["response_role"],
                                                interactive=True,
                                            )
                                            rc_tb = gr.Textbox(
                                                value=question_list[i][
                                                    "response_text_prefix"
                                                ],
                                                label="回复者起始字符",
                                                scale=4,
                                            )

                                    r_dd.change(
                                        partial(edit_question, question_list[i]["id"]),
                                        inputs=[r_dd, c_tb, rr_dd, rc_tb],
                                    )
                                    c_tb.change(
                                        partial(edit_question, question_list[i]["id"]),
                                        inputs=[r_dd, c_tb, rr_dd, rc_tb],
                                    )
                                    rr_dd.change(
                                        partial(edit_question, question_list[i]["id"]),
                                        inputs=[r_dd, c_tb, rr_dd, rc_tb],
                                    )
                                    rc_tb.change(
                                        partial(edit_question, question_list[i]["id"]),
                                        inputs=[r_dd, c_tb, rr_dd, rc_tb],
                                    )
                                    delete_btn = gr.Button(value="-")
                                    delete_btn.click(
                                        partial(
                                            delete_question, question_list[i]["id"]
                                        ),
                                        inputs=[],
                                        outputs=[question_num_input],
                                    )

                fold_questions_checkbox.change(
                    lambda x: gr.update(visible=not x),
                    fold_questions_checkbox,
                    free_questions_section,
                )
                with gr.Row():
                    start_benchmark_btn = gr.Button("开始基准测试")
                    stop_benchmark_btn = gr.Button("停止")
                    

                benchmark_output = gr.Textbox(
                    label="基准测试结果", lines=15, interactive=True
                )

                start_benchmark_btn.click(
                    partial(run_benchmark, agent, question_list),
                    (
                        bmk_temp_input,
                        bmk_top_p_input,
                        bmk_presence_penalty_input,
                        bmk_frequency_penalty_input,
                        bmk_penalty_decay_input,
                        bmk_using_init_state_checkbox,
                        bmk_begin_with_state_dir_input,
                    ),
                    [benchmark_output],
                )
                stop_benchmark_btn.click(stop_benchmark, None, None)
            with gr.Column(visible=False) as choice_question_section:
                mmlu_select_folder_dropdown = gr.Dropdown(
                    label="数据集", choices=mmlu_folders
                )
                with gr.Row():
                    mmlu_type_dropdown = gr.Dropdown(
                        label="数据类型", choices=["mmlu", "cmmlu", "ceval"]
                    )
                    mmlu_constraint_output_checkbox = gr.Checkbox(
                        label="约束采样", value=True
                    )
                    mmlu_start_with = gr.Textbox(
                        label="回复格式开头", interactive=True, value="answer:"
                    )
                start_mmlu_benchmark_btn = gr.Button("开始测试")
                with gr.Row():
                    mmlu_output = gr.Textbox(
                        label="测试结果", lines=15, interactive=True, scale=7
                    )
                    with gr.Column():
                        mmlu_score_output = gr.Textbox(label="目前分数")
                        mmlu_progress = gr.Slider(
                            label="进度",
                            minimum=0,
                            maximum=1,
                            value=0,
                            interactive=False,
                        )
                save_fp = f"{global_config.webui.benchmark.out_res_dir}/{mmlu_type_dropdown.value}.json"
                start_mmlu_benchmark_btn.click(
                    partial(run_mmlu, agent, save_fp),
                    [
                        mmlu_select_folder_dropdown,
                        bmk_temp_input,
                        bmk_top_p_input,
                        bmk_presence_penalty_input,
                        bmk_frequency_penalty_input,
                        bmk_penalty_decay_input,
                        mmlu_start_with,
                        mmlu_type_dropdown,
                        mmlu_constraint_output_checkbox,
                    ],
                    [mmlu_output, mmlu_score_output, mmlu_progress],
                )
            with gr.Column(visible=False) as code_question_section:
                with gr.Row():
                    humaneval_constraint_output_checkbox = gr.Checkbox(
                        label="约束采样", value=True
                    )
                    humaneval_req_start_with = gr.Textbox(
                        label="提问者格式开头", interactive=True, value="user: "
                    )
                    humaneval_resp_start_with = gr.Textbox(
                        label="回复格式开头", interactive=True, value="assistant:"
                    )
                start_humaneval_benchmark_btn = gr.Button("开始测试")
                humaneval_output = gr.Textbox(
                    label="测试结果", lines=15, interactive=True
                )
                start_humaneval_benchmark_btn.click(
                    partial(
                        run_humaneval,
                        agent,
                        global_config.webui.benchmark.out_res_dir,
                        global_config.webui.benchmark.human_eval_dir,
                    ),
                    [
                        bmk_temp_input,
                        bmk_top_p_input,
                        bmk_presence_penalty_input,
                        bmk_frequency_penalty_input,
                        bmk_penalty_decay_input,
                        humaneval_req_start_with,
                        humaneval_resp_start_with,
                        humaneval_constraint_output_checkbox,
                    ],
                    [humaneval_output],
                )

            # 处理模式切换的函数
            def switch_mode_bmk(choice):
                if choice == "自定义问题":
                    return (
                        gr.Column(visible=True),
                        gr.Column(visible=False),
                        gr.Column(visible=False),
                    )
                elif choice == "选择题":
                    return (
                        gr.Column(visible=False),
                        gr.Column(visible=True),
                        gr.Column(visible=False),
                    )
                elif choice == "代码":
                    return (
                        gr.Column(visible=False),
                        gr.Column(visible=False),
                        gr.Column(visible=True),
                    )

            # 绑定模式切换事件
            eval_mode_selector.change(
                fn=switch_mode_bmk,
                inputs=eval_mode_selector,
                outputs=[
                    free_chat_section,
                    choice_question_section,
                    code_question_section,
                ],
            )
        with gr.Tab("API聊天工具"):
            api_settings_selector = gr.Dropdown(
                label="预设",
                choices=["预设1", "预设2", "预设3", "预设4", "预设5"],
                value="预设1",
            )
            api_base_input = gr.Textbox(
                label="API Base",
                placeholder="https://...",
                interactive=True,
                value=api_base_input_list[0],
            )
            api_key_input = gr.Textbox(
                label="API Key",
                placeholder="sk-xxx...",
                interactive=True,
                value=api_key_input_list[0],
            )
            api_model_input = gr.Textbox(
                label="模型",
                placeholder="deepseek-chat",
                interactive=True,
                value=api_model_input_list[0],
            )

            def update_api_list(new_value, selector_value, target_list):
                index = api_settings_selector.choices.index(
                    (selector_value, selector_value)
                )
                target_list[index] = new_value

            def update_inputs_from_lists(selector_value):
                index = api_settings_selector.choices.index(
                    (selector_value, selector_value)
                )
                return (
                    api_base_input_list[index],
                    api_key_input_list[index],
                    api_model_input_list[index],
                )

            api_base_input.change(
                fn=lambda x, y: update_api_list(x, y, api_base_input_list),
                inputs=[api_base_input, api_settings_selector],
                outputs=None,
            )
            api_key_input.change(
                fn=lambda x, y: update_api_list(x, y, api_key_input_list),
                inputs=[api_key_input, api_settings_selector],
                outputs=None,
            )
            api_model_input.change(
                fn=lambda x, y: update_api_list(x, y, api_model_input_list),
                inputs=[api_model_input, api_settings_selector],
                outputs=None,
            )

            api_settings_selector.change(
                fn=update_inputs_from_lists,
                inputs=[api_settings_selector],
                outputs=[api_base_input, api_key_input, api_model_input],
            )
            with gr.Row():
                api_selector = gr.Dropdown(
                    label="配置",
                    choices=[
                        "推理参数",
                        "读取历史记录",
                    ],
                    value="更新参数",
                )
            with gr.Row(visible=False) as api_hist_section:
                api_hist_dropdown = gr.Dropdown(
                    label="历史记录路径", choices=api_hist_dirs, scale=4
                )
                with gr.Column():
                    api_load_hist_btn = gr.Button("读取历史记录")
                    refresh_api_hist_btn = gr.Button("刷新列表", scale=1)

            with gr.Row(visible=True) as api_param_section:
                with gr.Column():
                    api_sender_name_input = gr.Textbox(label="用户名字", value="user")
                    api_bot_name_input = gr.Textbox(label="AI名字", value="assistant")
                    api_temp_input = gr.Number(label="温度", value=1.0)
                with gr.Column():
                    api_top_p_input = gr.Number(label="Top P", value=0.7)
                    api_presence_penalty_input = gr.Number(label="历史惩罚", value=0.2)
                    api_frequency_penalty_input = gr.Number(label="频率惩罚", value=0.2)

            api_log_output = gr.Textbox(label="操作提示", lines=1, interactive=False)

            def refresh_api_hist_files():
                global api_hist_dirs
                api_hist_dirs = []
                for root, dirs, files in os.walk(api_hist_DIR):
                    for file in files:
                        if file.endswith(".json"):
                            api_hist_dirs.append(os.path.join(root, file))
                return gr.update(choices=api_hist_dirs)

            refresh_api_hist_btn.click(refresh_api_hist_files, None, api_hist_dropdown)
            api_load_hist_btn.click(
                load_api_hist, inputs=[api_hist_dropdown], outputs=[api_log_output]
            )

            api_selector.change(
                lambda x: (
                    gr.update(visible=x == "推理参数"),
                    gr.update(visible=x == "读取历史记录"),
                ),
                inputs=api_selector,
                outputs=[
                    load_svpt_section,
                    api_hist_section,
                ],
            )

            api_chatbot = gr.Chatbot(label="聊天", scale=6, type="messages")
            with gr.Row():
                api_reset_btn = gr.Button("重置")
                api_save_hist_btn = gr.Button("保存历史记录")
            with gr.Row():
                api_regenerate_btn = gr.Button("重新生成")
                api_back_btn = gr.Button("删除该轮对话")
                api_listen_btn = gr.Button("等待回复")

            with gr.Row():
                api_message_input = gr.Textbox(
                    label="输入消息", placeholder="输入消息...", scale=9
                )
                with gr.Column(scale=1):
                    api_send_btn = gr.Button("发送")
                    api_regenerate_btn2 = gr.Button("重新生成")
            with gr.Row():
                api_add_msg_role = gr.Dropdown(
                    label="role",
                    choices=[
                        "user",
                        "assistant",
                        "system",
                    ],
                    value="conversation",
                )
                api_add_message_content = gr.Textbox(
                    label="添加消息",
                    placeholder="输入要添加的消息...例如“AAA: BBB”",
                    scale=6,
                )
                api_add_btn = gr.Button("添加")
            api_save_hist_btn.click(
                partial(save_api_hist, global_config.webui.api_hist_dir),
                outputs=[api_log_output],
            )
            api_reset_btn.click(api_reset, outputs=[api_chatbot])
            api_regenerate_btn.click(
                api_regenerate,
                inputs=[
                    api_bot_name_input,
                    api_base_input,
                    api_key_input,
                    api_model_input,
                    temp_input,
                    top_p_input,
                    api_presence_penalty_input,
                    api_frequency_penalty_input,
                ],
            )
            api_regenerate_btn2.click(
                api_regenerate,
                inputs=[
                    api_bot_name_input,
                    api_base_input,
                    api_key_input,
                    api_model_input,
                    temp_input,
                    top_p_input,
                    api_presence_penalty_input,
                    api_frequency_penalty_input,
                ],
                outputs=[api_chatbot],
            )
            api_back_btn.click(api_back, outputs=[api_chatbot])
            api_listen_btn.click(
                api_listen,
                inputs=[
                    api_bot_name_input,
                    api_base_input,
                    api_key_input,
                    api_model_input,
                    api_temp_input,
                    api_top_p_input,
                    api_presence_penalty_input,
                    api_frequency_penalty_input,
                ],
                outputs=[api_chatbot],
            )
            api_send_btn.click(
                api_chat,
                inputs=[
                    api_sender_name_input,
                    api_bot_name_input,
                    api_message_input,
                    api_base_input,
                    api_key_input,
                    api_model_input,
                    api_temp_input,
                    api_top_p_input,
                    api_presence_penalty_input,
                    api_frequency_penalty_input,
                ],
                outputs=[api_chatbot],
            )
            api_message_input.submit(
                api_chat,
                inputs=[
                    api_sender_name_input,
                    api_bot_name_input,
                    api_message_input,
                    api_base_input,
                    api_key_input,
                    api_model_input,
                    api_temp_input,
                    api_top_p_input,
                    api_presence_penalty_input,
                    api_frequency_penalty_input,
                ],
                outputs=[api_chatbot],
            )
            api_add_btn.click(
                api_add,
                inputs=[api_add_msg_role, api_add_message_content],
                outputs=[api_chatbot],
            )

        with gr.Tab("训练"):
            with gr.Row():
                train_model_dropdown = gr.Dropdown(
                    label="重新加载训练模型", choices=[""] + model_files, scale=4
                )
                with gr.Column():
                    train_load_model_btn = gr.Button("读取模型", scale=1)
                    train_refresh_model_btn = gr.Button("刷新列表", scale=1)
            train_load_model_btn.click(
                train_agent.load_train_model, train_model_dropdown, None
            )
            train_refresh_model_btn.click(
                refresh_model_files, None, train_model_dropdown
            )
            # 添加模式选择下拉框
            train_mode_selector = gr.Dropdown(
                choices=["单文件夹数据训练", "多文件夹数据采样", "强化学习"],
                label="请选择训练模式",
                value="单文件夹数据训练",
            )

            # 在线学习模式部分
            with gr.Column(visible=True) as online_learning_section:
                with gr.Row():
                    with gr.Column(scale=8):
                        ollr_data_folder_input = gr.Textbox(
                            label="数据文件夹地址", placeholder="数据文件夹地址"
                        )
                        with gr.Row():
                            ollr_use_init_state_checkbox = gr.Checkbox(
                                label="加载初始state", value=False
                            )
                            ollr_init_state_path_input = gr.Textbox(
                                label="初始state路径",
                                placeholder="初始state路径",
                                lines=1,
                                scale=6,
                            )
                        ollr_start_train_btn = gr.Button("开始训练")
                    with gr.Column(scale=1):
                        with gr.Row():
                            ollr_train_epoch_input = gr.Number(label="Epoch", value=1)
                            ollr_batch_size_input = gr.Number(
                                label="Batch Size", value=1
                            )
                            ollr_n_save_ckpt_input = gr.Number(
                                label="保存检查点频率", value=1
                            )
                    with gr.Column(scale=1):
                        ollr_ctx_len_input = gr.Number(
                            label="最大上下文长度", value=3072
                        )
                        ollr_multi_scale_alpha_input = gr.Number(
                            label="多尺度缩放因子", value=0.9
                        )
                        ollr_keep_states_mode_dropdown = gr.Dropdown(
                            label="历史state延续模式",
                            choices=["never", "epoch", "step"],
                            interactive=True,
                        )

            # 多文件夹数据采样模式部分
            with gr.Column(visible=False) as folder_learning_section:
                with gr.Row():
                    with gr.Column():
                        fllr_dataset_list_input = gr.Textbox(
                            label="数据集列表",
                            placeholder="""[
            [
                "数据文件夹地址",
                -1 (所有样本)
            ],
            [
                "数据文件夹地址",
                10 (抽样10个)
            ],
        ]""",
                            lines=5,
                            scale=5,
                        )
                        fllr_start_train_btn = gr.Button("开始训练")
                    with gr.Column():
                        fllr_epoch_input = gr.Number(label="Epoch", value=1)
                        fllr_batch_size_input = gr.Number(label="Batch Size", value=1)
                    with gr.Column():
                        fllr_n_save_ckpt_epoch_input = gr.Number(
                            label="保存检查点频率 (epoch)", value=1
                        )
                        fllr_use_n_save_ckpt_step_checkbox = gr.Checkbox(
                            label="开启按step保存", value=False
                        )
                        fllr_n_save_ckpt_step_input = gr.Number(
                            label="保存检查点频率 (step)", value=1, visible=False
                        )

            # 强化学习部分
            with gr.Column(visible=False) as rl_section:
                with gr.Row():
                    rl_dataset_list = gr.Textbox(
                        label="强化学习数据集列表",
                        placeholder="""[
        [
            "数据文件夹地址",
            -1
        ],
        [
            "数据文件夹地址",
            10
        ],
    ]""",
                        lines=8,
                        scale=7,
                    )
                    with gr.Column():
                        with gr.Row():
                            rl_epoch_input = gr.Number(label="Epoch", value=1)
                            rl_n_use_max_choices_input = gr.Number(
                                label="最大对比标签数", value=5
                            )
                        start_rl_train_btn = gr.Button("开始训练")

            # 处理模式切换的函数
            def switch_mode_train(choice):
                if choice == "单文件夹数据训练":
                    return (
                        gr.Column(visible=True),
                        gr.Column(visible=False),
                        gr.Column(visible=False),
                    )
                elif choice == "多文件夹数据采样":
                    return (
                        gr.Column(visible=False),
                        gr.Column(visible=True),
                        gr.Column(visible=False),
                    )
                elif choice == "强化学习":
                    return (
                        gr.Column(visible=False),
                        gr.Column(visible=False),
                        gr.Column(visible=True),
                    )

            # 绑定模式切换事件
            train_mode_selector.change(
                fn=switch_mode_train,
                inputs=train_mode_selector,
                outputs=[online_learning_section, folder_learning_section, rl_section],
            )

            with gr.Row():
                train_output_info = gr.Textbox(
                    label="输出信息", lines=1, interactive=False, value=""
                )
                progress_slider = gr.Slider(
                    label="训练进度", minimum=0, maximum=100, value=0, interactive=False
                )
            with gr.Row():
                text_loss_curve_plot = gr.Plot(label="文本Loss曲线")

            ollr_start_train_btn.click(
                train_agent.train_single_folder,
                [
                    ollr_data_folder_input,
                    ollr_train_epoch_input,
                    ollr_batch_size_input,
                    ollr_n_save_ckpt_input,
                    ollr_ctx_len_input,
                    ollr_multi_scale_alpha_input,
                    ollr_keep_states_mode_dropdown,
                ],
                [progress_slider, train_output_info, text_loss_curve_plot],
            )

            fllr_start_train_btn.click(
                train_agent.train_multiple_folders,
                [
                    fllr_dataset_list_input,
                    fllr_epoch_input,
                    fllr_batch_size_input,
                    fllr_n_save_ckpt_epoch_input,
                    fllr_use_n_save_ckpt_step_checkbox,
                    fllr_n_save_ckpt_step_input,
                ],
                [progress_slider, train_output_info, text_loss_curve_plot],
            )

            fllr_use_n_save_ckpt_step_checkbox.change(
                lambda x: gr.update(visible=x),
                fllr_use_n_save_ckpt_step_checkbox,
                fllr_n_save_ckpt_step_input,
            )

            def start_rl_training(rl_dataset_list_str, epoch, n_use_max_choices):
                variables = {}
                variables["folder_weight_dir_list"] = json.loads(rl_dataset_list_str)
                variables["n_use_max_choices"] = n_use_max_choices
                for e in range(epoch):
                    for msgs in train_agent.train_dpo(variables):
                        if len(msgs) > 1:
                            (
                                step,
                                rl_loss,
                                chosen_rewards,
                                rejected_rewards,
                            ) = msgs
                            output_text = f"Step: {step}, RL Loss: {rl_loss}, Chosen Rewards: {chosen_rewards}, Rejected Rewards: {rejected_rewards}"
                            print(output_text)
                            yield step, output_text
                        else:
                            yield 100, f"{output_text}\n训练完毕，已保存至{msgs}"

            start_rl_train_btn.click(
                start_rl_training,
                [rl_dataset_list, rl_epoch_input, rl_n_use_max_choices_input],
                [progress_slider, train_output_info],
            )

    demo.load(
        partial(load_user_preference, save_preference_dir, agent),
        None,
        outputs=[
            sender_name_input,
            replier_name_input,
            temp_input,
            top_p_input,
            presence_penalty_input,
            frequency_penalty_input,
            decay_penalty_input,
            usr_sp_token_dropdown,
            bot_sp_token_dropdown,
            bmk_temp_input,
            bmk_top_p_input,
            bmk_presence_penalty_input,
            bmk_frequency_penalty_input,
            bmk_penalty_decay_input,
            bmk_using_init_state_checkbox,
            bmk_begin_with_state_dir_input,
            api_base_input,
            api_key_input,
            api_model_input,
            api_sender_name_input,
            api_bot_name_input,
            api_temp_input,
            api_top_p_input,
            api_presence_penalty_input,
            api_frequency_penalty_input,
            ollr_data_folder_input,
            ollr_use_init_state_checkbox,
            ollr_init_state_path_input,
            ollr_train_epoch_input,
            ollr_batch_size_input,
            ollr_n_save_ckpt_input,
            ollr_ctx_len_input,
            ollr_multi_scale_alpha_input,
            ollr_keep_states_mode_dropdown,
            fllr_dataset_list_input,
            fllr_epoch_input,
            fllr_batch_size_input,
            fllr_n_save_ckpt_epoch_input,
            fllr_use_n_save_ckpt_step_checkbox,
            fllr_n_save_ckpt_step_input,
            rl_dataset_list,
            rl_epoch_input,
            rl_n_use_max_choices_input,
            chat_operation_output,
        ],
    )
    save_preference_modules = [
        sender_name_input, replier_name_input,
        temp_input, top_p_input,
        presence_penalty_input, frequency_penalty_input,
        decay_penalty_input,
        usr_sp_token_dropdown, bot_sp_token_dropdown,
        bmk_temp_input, bmk_top_p_input,
        bmk_presence_penalty_input, bmk_frequency_penalty_input,
        bmk_penalty_decay_input,
        bmk_using_init_state_checkbox, bmk_begin_with_state_dir_input,
        api_base_input, api_key_input, api_model_input,
        api_sender_name_input, api_bot_name_input,
        api_temp_input, api_top_p_input,
        api_presence_penalty_input, api_frequency_penalty_input,
        ollr_data_folder_input,
        ollr_use_init_state_checkbox, ollr_init_state_path_input,
        ollr_train_epoch_input, ollr_batch_size_input,
        ollr_n_save_ckpt_input, ollr_ctx_len_input,
        ollr_multi_scale_alpha_input, ollr_keep_states_mode_dropdown,
        fllr_dataset_list_input, fllr_epoch_input,
        fllr_batch_size_input, fllr_n_save_ckpt_epoch_input,
        fllr_use_n_save_ckpt_step_checkbox, fllr_n_save_ckpt_step_input,
        rl_dataset_list, rl_epoch_input, rl_n_use_max_choices_input
    ]
    if not os.path.exists(save_preference_dir):
        demo.load(
            fn=partial(save_user_preference, save_preference_dir),
            inputs=save_preference_modules
        )

    for gr_module in save_preference_modules:
        gr_module.change(
            fn=partial(save_user_preference, save_preference_dir),
            inputs=save_preference_modules,  # 所有模块作为输入
        )


demo.launch()
