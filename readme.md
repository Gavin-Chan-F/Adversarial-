对抗训练实验
===========================

# 1.	实验简介
	
	本项目使用 FGSM、PGD、FreeLB 对情绪分类数据集进行对抗性训练的 tensorflow2 实现，baseline模型为TextCNN。

# 2.	实验环境
	
	Python3
	Tensorflow==2.6.2

# 3.	使用

	pip3 install -r requirements.txt
	python3 main.py
	
# 4.	实验结果

	实验结果保存在"./save_data/report.txt"