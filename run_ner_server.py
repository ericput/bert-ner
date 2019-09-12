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
