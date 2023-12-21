import json
import time
from flask import Flask, request, jsonify
from law_interface import MyArgs, quicklook, analyze
from model import ModelShell

args = MyArgs()
shell = ModelShell(args)

app = Flask(__name__)

@app.route('/peek', methods=['POST'])
def func_peek():
    t0 = time.time()
    res = {
        "err_code": 0,
        "message": "",
        "time_cost": 0,
        "result": None
    }
    # 获取 JSON 输入
    input_data = request.get_json()
    judgement = input_data.get('judgement')
    # 判断是否存在judgement字段
    if judgement is None:
        res['err_code'] = 1
        res['message'] = "未找到judgment字段"
        return jsonify(res)
    
    fact_jdg, jdg = quicklook(judgement)
    
    res['message'] = "已处理"
    
    res['result'] = {
        "fact_jdg": fact_jdg,
        "jdg": jdg,
    }
    t1 = time.time()
    res['time_cost'] = round(t1 - t0, 4)
    # 返回 JSON 输出
    return jsonify(res)

@app.route('/analyze', methods=['POST'])
def func_analyze():
    t0 = time.time()
    res = {
        "err_code": 0,
        "time_cost": 0,
        "message": "",
        "result": None
    }
    # 获取 JSON 输入
    input_data = request.get_json()
    judgement = input_data.get('judgement')
    # 判断是否存在judgement字段
    if judgement is None:
        res['err_code'] = 1
        res['message'] = "未找到judgment字段"
        return jsonify(res)
    
    result = analyze(shell, judgement)
    
    res['message'] = "已处理"
    res['result'] = result
    t1 = time.time()
    res['time_cost'] = round(t1 - t0, 4)


    # 返回 JSON 输出
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=False, port=4567)

