import requests
import json
import os
from utils.message_manager import cList
from utils.webui.webui_utils import loss_curve_plot
import asyncio

class TrainAgent:
    def __init__(self, train_server="http://0.0.0.0:3000"):
        self.train_server = train_server

    def load_train_model(self, model_path):
        """加载模型的逻辑"""
        response = requests.post(
            self.train_server + "/load_model", json={"load_dir": model_path}
        ).json()
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
        return f"模型已加载：{model_path}"

    def train_single_folder(
        self,
        forder_dir,
        epoch,
        batch_size_per_gpu,
        n_save_ckpt,
        multi_scale_ctx,
        multi_scale_alpha,
        keep_states_mode,
        use_qa_mask=False,
        begin_with_state_dir=None,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        """训练单个文件夹的逻辑"""
        variables = {
            "forder_dir": forder_dir,
            "epoch": epoch,
            "batch_size_per_gpu": batch_size_per_gpu,
            "n_save_ckpt": n_save_ckpt,
            "multi_scale_ctx": multi_scale_ctx,
            "multi_scale_alpha": multi_scale_alpha,
            "keep_states_mode": keep_states_mode,
            "begin_with_state_dir": begin_with_state_dir,
            "use_qa_mask": use_qa_mask,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
        }
        text_loss_list = []
        with requests.post(
            self.train_server + "/train_single_datafolder", json=variables, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield 0, f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield 100, output_text, loss_curve_plot(text_loss_list)
                else:
                    epoch = result["epoch"]
                    step = result["step"]
                    mean_text_loss = result["mean_text_loss"]
                    text_loss = result["text_loss"]
                    text_loss_list.append(text_loss)
                    n_tokens = result["n_tokens"]
                    left_tokens = result["left_tokens"]
                    progress_percent = (
                        (n_tokens - left_tokens) / (n_tokens + 1e-4) * 100
                    )
                    output_text = (
                        f"Epoch: {epoch}, Step: {step}, Loss: {mean_text_loss}"
                    )

                    yield progress_percent, output_text, loss_curve_plot(text_loss_list)

    def train_multiple_folders(
        self,
        folder_weight_dir_list,
        epoch,
        batch_size_per_gpu,
        n_save_ckpt,
        save_step_on=False,
        n_save_step=None,
        use_qa_mask=False,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        """训练多个文件夹的逻辑"""
        n_save_step = n_save_step if save_step_on else None
        variables = {
            "epoch": epoch,
            "batch_size_per_gpu": batch_size_per_gpu,
            "n_save_ckpt": n_save_ckpt,
            "n_save_step": n_save_step,
            "use_qa_mask": use_qa_mask,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
        }
        variables["folder_weight_dir_list"] = json.loads(folder_weight_dir_list)
        text_loss_list = []
        with requests.post(
            self.train_server + "/train_from_folders", json=variables, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield 0, f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield 100, output_text, loss_curve_plot(text_loss_list)
                else:
                    epoch = result["epoch"]
                    step = result["step"]
                    mean_loss = result["mean_text_loss"]
                    text_loss = result["text_loss"]
                    text_loss_list.append(text_loss)
                    n_data = result["n_data"]
                    left_data = result["left_data"]
                    progress_percent = (n_data - left_data) / n_data * 100
                    output_text = f"Epoch: {epoch}, Step: {step}, Loss: {mean_loss}"
                    yield progress_percent, output_text, loss_curve_plot(text_loss_list)

    def train_dpo(self, variables):
        # variables["stream"] = False
        # variables["allow_multilabel"]=False
        # requests.post(
        #     self.train_server + "/train_dpo_from_folders", json=variables
        # )

        with requests.post(
            self.train_server + "/train_dpo_from_folders", json=variables, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    yield [result["to_dir"]]
                else:
                    step = result["step"]
                    dpo_loss = result["dpo_loss"]
                    chosen_rewards = result["chosen_rewards"]
                    rejected_rewards = result["rejected_rewards"]
                    print(step, dpo_loss, chosen_rewards, rejected_rewards)
                    yield step, dpo_loss, chosen_rewards, rejected_rewards
