## 参考链接
- torch版本: [https://github.com/xue-pai/FuxiCTR]
- tf版本: [https://github.com/DSXiangLi/CTR]

## torch版本
- 数据集[taobao_tiny]
	- 读取csv得到df
	- 预处理: 填充空值
	- 特征处理
		- numeric: normalizing
		- categorical	
		- sequence
- 模型[DeepFM]

## tf版本
- 数据集[amazon(eletronics)][movielens]
	- tfrecords
- 模型[DIN][MOE][Bias][UserPerExpert]