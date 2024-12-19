# 导入所需的库和模块
import time
import torch
import numpy as np
from train_eval import train, init_network, predict  # 导入训练、初始化网络和预测函数
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif  # 导入数据构建和迭代器构建函数
from models.bert_CNN import Config, Model  # 导入配置类和模型类
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig  # 导入transformers库中的模型、分词器和配置
from preprocess import tags, tag2id, id2tag  # 导入预处理标签和ID映射

# 主程序入口
if __name__ == '__main__':
    train_flag = False  # 标志位，用于控制是否进行训练
    print(tag2id)  # 打印标签到ID的映射
    print(id2tag)  # 打印ID到标签的映射

    # 获取环境变量，判断是否为本地运行环境
    run_env = os.getenv('ENV', 'NULL')
    if run_env is not None and run_env == "local":
        local = True
    else:
        local = False
    print(f"run_env = {run_env}")  # 打印运行环境

    # 初始化配置
    config = Config("kk", local=local)  # "kk"是配置目录的路径
    # 加载模型
    model = Model(config)
    if config.save_path_best is not None:
        # 加载预训练模型
        load_result = model.load_state_dict(torch.load(config.save_path_best, map_location=torch.device('cpu')), strict=False)
        print(f"Load ckpt to for init:{config.save_path_best}")
        print("Load ckpt to continue init result : {}".format(str(load_result)))
    model.to(config.device)  # 将模型发送到配置的设备（CPU或GPU）

    # 设置随机种子，保证结果可复现
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    start_time = time.time()  # 记录开始时间

    # 如果train_flag为True，则进行训练
    if train_flag == True:
        print("Loading data...")  # 打印数据加载信息
        train_data, dev_data, test_data = build_dataset(config)  # 构建训练、验证和测试数据集
        train_iter = build_iterator(train_data, config)  # 构建训练数据迭代器
        dev_iter = build_iterator(dev_data, config)  # 构建验证数据迭代器
        test_iter = build_iterator(test_data, config)  # 构建测试数据迭代器
        time_dif = get_time_dif(start_time)  # 获取已使用时间
        print("Time usage:", time_dif)  # 打印已使用时间
        # 进行训练
        train(config, model, train_iter, dev_iter, test_iter)

    # 生成测试数据
    print("开始生成测试数据")  # 打印开始生成测试数据信息
    predict(model, config)  # 调用预测函数，生成测试数据