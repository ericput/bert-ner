import argparse
from flask import Flask, jsonify, request
from njuner import NJUNER

def start(host='0.0.0.0', port=5000, debug=True):
    app = Flask(__name__)
    @app.route('/ner', methods=['POST'])
    def ner():
        text = request.json['text']
        if isinstance(text, str):
            text = [text]
        result = model.label(text)
        return jsonify(result)
    app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NER REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="55555")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    model = NJUNER(model_dir=args.model_dir, batch_size=args.batch_size)
    start(host=args.ip, port=args.port, debug=args.debug)

'''
上面的代码使用Flask库架设NER REST服务器（需要先安装pip包njuner），服务器地址为http://ip:port/ner，服务方法为POST。
数据请求格式为{'text': str|list<str>}；数据回复格式为list<lineResult>，其中lineResult格式为list<[token_tag, token]>。

使用例子：
服务器端：
    在ip为111.112.113.114的服务器上运行'python run_ner_server.py --model_dir model_msra'
客户端：
    import requests
    r = requests.post(url='http://ip:port/ner', json={'text': '李雷和韩梅梅去上海迪斯尼乐园玩。'})
    print(r.json())
    '[[['B-PER', '李'], ['I-PER', '雷'], ['O', '和'], ['B-PER', '韩'], ['I-PER', '梅'], ['I-PER', '梅'], ['O', '去'], 
    ['B-ORG', '上'], ['I-ORG', '海'], ['I-ORG', '迪'], ['I-ORG', '斯'], ['I-ORG', '尼'], ['I-ORG', '乐'], ['I-ORG', '园'], 
    ['O', '玩'], ['O', '。']]]'
'''
