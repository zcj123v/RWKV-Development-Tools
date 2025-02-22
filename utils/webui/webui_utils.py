import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gradio as gr


def loss_curve_plot(data, caption:str="Loss Curve"):
    plt.close()
    # 如果数据为空，返回一个空图
    if not data:
        return plt.figure()

    fig, ax = plt.subplots(facecolor="#2F2F2F")
    ax.set_facecolor("#2F2F2F")

    # 绘制损失曲线
    ax.plot(data, label="Loss Curve", color="#00FF7F", linewidth=2)  # 更亮的绿色

    # 添加标题和标签
    ax.set_title(caption, color="white")
    ax.set_xlabel("Steps", color="white")
    ax.set_ylabel("Loss", color="white")

    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # 添加图例
    legend = ax.legend()
    legend.get_frame().set_facecolor("#2F2F2F")  # 图例背景设为深色
    legend.get_frame().set_edgecolor("white")  # 图例边框设为白色
    for text in legend.get_texts():
        text.set_color("white")  # 图例文字设为白色
    # 调整图形边距
    plt.tight_layout()

    # 返回图形对象
    return fig


def save_user_preference(save_path, *args):
    """保存用户偏好到指定路径
    
    Args:
        save_path: 保存路径
        *args: 按顺序包含所有需要保存的偏好值
    """
    # 尝试加载已有的偏好文件
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            preferences = json.load(f)
    except FileNotFoundError:
        preferences = {}
        
    # 根据参数顺序设置对应的键值
    keys = [
        "sender_name", "replier_name", "temp", "top_p",
        "presence_penalty", "frequency_penalty", "decay_penalty",
        "usr_sp_token", "bot_sp_token",
        "bmk_temp", "bmk_top_p", "bmk_presence_penalty",
        "bmk_frequency_penalty", "bmk_penalty_decay",
        "bmk_use_init_state", "bmk_use_init_state_dir",
        "api_base", "api_key", "api_model",
        "api_sender_name", "api_bot_name",
        "api_temp", "api_top_p",
        "api_presence_penalty", "api_frequency_penalty",
        "ollr_train_data_folder_dir", "ollr_train_init_state",
        "ollr_train_load_init_state_dir", "ollr_train_epoch",
        "ollr_train_batch_size", "ollr_train_n_save_ckpt",
        "ollr_train_ctx_len", "ollr_train_multi_scale_alpha",
        "ollr_train_keep_states_mode",
        "fllr_dataset_list", "fllr_train_epoch",
        "fllr_train_batch_size", "fllr_train_n_save_ckpt",
        "fllr_train_save_ckpt_step", "fllr_train_n_step_save_ckpt",
        "parquet_file_path", "grpo_req_sp_token", "grpo_req_prefix",
        "grpo_resp_sp_token", "grpo_resp_prefix", "grpo_n_save_ckpt",
        "grpo_max_resp_ctx_len", "grpo_tiny_batch_size",
        "grpo_num_rollouts", "grpo_train_batch_size", "grpo_n_save_episode",
        "grpo_lr_init", "grpo_lr_final", "grpo_lr_warmup",
        "grpo_accumulate_grad", "grpo_temperature",
        "grpo_top_p", "grpo_presence_penalty", "grpo_frequency_penalty",
        "grpo_penalty_decay",
    ]
    
    # 更新偏好字典
    for key, value in zip(keys, args):
        if value is not None and not isinstance(value, dict):  # 跳过gr.update()返回的值
            preferences[key] = value
            
    # 保存到文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preferences, f, ensure_ascii=False, indent=4)
        


def load_user_preference(load_path, agent):
    """从指定路径加载用户偏好"""
    try:
        with open(load_path, "r") as f:
            preferences = json.load(f)

        (
            sender_name,
            replier_name,
            temp,
            top_p,
            presence_penalty,
            frequency_penalty,
            decay_penalty,
            usr_sp_token,
            bot_sp_token,
        ) = (
            preferences.get("sender_name"),
            preferences.get("replier_name"),
            preferences.get("temp"),
            preferences.get("top_p"),
            preferences.get("presence_penalty"),
            preferences.get("frequency_penalty"),
            preferences.get("decay_penalty"),
            preferences.get("usr_sp_token"),
            preferences.get("bot_sp_token"),
        )

        agent.update_chatbot_params(
            sender_name,
            replier_name,
            float(temp),
            float(top_p),
            float(presence_penalty),
            float(frequency_penalty),
            float(decay_penalty),
            usr_sp_token,
            bot_sp_token,
        )
        
        

        
        return [
            sender_name,
            replier_name,
            temp,
            top_p,
            presence_penalty,
            frequency_penalty,
            decay_penalty,
            usr_sp_token,
            bot_sp_token,
            preferences.get("bmk_temp", gr.update()),
            preferences.get("bmk_top_p", gr.update()),
            preferences.get("bmk_presence_penalty", gr.update()),
            preferences.get("bmk_frequency_penalty", gr.update()),
            preferences.get("bmk_penalty_decay", gr.update()),
            preferences.get("bmk_use_init_state", gr.update()),
            preferences.get("bmk_use_init_state_dir", gr.update()),
            preferences.get("api_base", gr.update()),
            preferences.get("api_key", gr.update()),
            preferences.get("api_model", gr.update()),
            preferences.get("api_sender_name", gr.update()),
            preferences.get("api_bot_name", gr.update()),
            preferences.get("api_temp", gr.update()),
            preferences.get("api_top_p", gr.update()),
            preferences.get("api_presence_penalty", gr.update()),
            preferences.get("api_frequency_penalty", gr.update()),
            preferences.get("ollr_train_data_folder_dir", gr.update()),
            preferences.get("ollr_train_init_state", gr.update()),
            preferences.get("ollr_train_load_init_state_dir", gr.update()),
            preferences.get("ollr_train_epoch", gr.update()),
            preferences.get("ollr_train_batch_size", gr.update()),
            preferences.get("ollr_train_n_save_ckpt", gr.update()),
            preferences.get("ollr_train_ctx_len", gr.update()),
            preferences.get("ollr_train_multi_scale_alpha", gr.update()),
            preferences.get("ollr_train_keep_states_mode", gr.update()),
            preferences.get("fllr_dataset_list", gr.update()),
            preferences.get("fllr_train_epoch", gr.update()),
            preferences.get("fllr_train_batch_size", gr.update()),
            preferences.get("fllr_train_n_save_ckpt", gr.update()),
            preferences.get("fllr_train_save_ckpt_step", gr.update()),
            preferences.get("fllr_train_n_step_save_ckpt", gr.update()),
            preferences.get("parquet_file_path", gr.update()),
            preferences.get("grpo_req_sp_token", gr.update()),
            preferences.get("grpo_req_prefix", gr.update()),
            preferences.get("grpo_resp_sp_token", gr.update()),
            preferences.get("grpo_resp_prefix", gr.update()),
            preferences.get("grpo_n_save_ckpt", gr.update()),
            preferences.get("grpo_max_resp_ctx_len", gr.update()),
            preferences.get("grpo_tiny_batch_size", gr.update()),
            preferences.get("grpo_num_rollouts", gr.update()),
            preferences.get("grpo_train_batch_size", gr.update()),
            preferences.get("grpo_n_save_episode", gr.update()),
            preferences.get("grpo_lr_init", gr.update()),
            preferences.get("grpo_lr_final", gr.update()),
            preferences.get("grpo_lr_warmup", gr.update()),
            preferences.get("grpo_accumulate_grad", gr.update()),
            preferences.get("grpo_temperature", gr.update()),
            preferences.get("grpo_top_p", gr.update()),
            preferences.get("grpo_presence_penalty", gr.update()),
            preferences.get("grpo_frequency_penalty", gr.update()),
            preferences.get("grpo_penalty_decay", gr.update()),
            "已加载历史填写。",
        ]
    except FileNotFoundError:
        return [gr.update()] * 60 + ["无历史操作记录"]
    except Exception as e:
        return [gr.update()] * 60 + [f"加载失败: {str(e)}"]
