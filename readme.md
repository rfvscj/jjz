## 减假暂逻辑


### 各个模块介绍
- model：模型类
- model shell：对model的封装，将模型无关的输入处理为模型对应的输入
- interface：调用shell进行处理，并汇总输出
- api：将操作封装为api
- app: 利用gradio搭建的简易界面
- train: 模型训练


### 建议操作
后台运行api.py，然后调用api即可，输入输出均为字典，字典中需要有judgement字段，即整篇的判决书
peek接口可以较快地返回将文书切分的结果，可以不用
analyze接口返回了一系列信息，其中result中的conclusion字段是汇总后的输出，可以直接展示这个
