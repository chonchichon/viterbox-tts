"""
Viterbox - API Server
"""
from flask import Flask, request, jsonify
from viterbox import Viterbox
import torch
import gc

app = Flask(__name__)

# Load model khi khởi động
print("Đang tải Viterbox model...")
tts = Viterbox.from_pretrained("cuda")
print("✅ Viterbox model đã sẵn sàng")

# Counter để track requests
request_counter = 0
CLEANUP_INTERVAL = 100  # Cleanup mỗi 100 requests


@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint để tạo audio từ text và lưu vào output_path

    Request JSON:
    {
        "text": "Văn bản cần chuyển đổi",
        "output_path": "D:/path/to/output.wav",  # required
        "language": "vi",  # optional, mặc định "vi"
        "audio_prompt": null,  # optional, đường dẫn file audio tham chiếu
        "exaggeration": 0.5,  # optional, 0.0-2.0
        "cfg_weight": 0.5,  # optional, 0.0-1.0
        "temperature": 0.8,  # optional, 0.1-1.0
        "max_new_tokens": null,  # optional, auto-detect
        "sentence_pause": 0.5  # optional, giây
    }
    """
    global request_counter

    try:
        data = request.json

        # Lấy parameters
        text = data.get('text')
        if not text:
            return jsonify({"error": "Thiếu tham số 'text'"}), 400

        output_path = data.get('output_path')
        if not output_path:
            return jsonify({"error": "Thiếu tham số 'output_path'"}), 400

        language = "vi"
        audio_prompt = data.get('audio_prompt', None)
        exaggeration = data.get('exaggeration', 0.5)
        cfg_weight = data.get('cfg_weight', 0.5)
        temperature = data.get('temperature', 0.8)
        sentence_pause = data.get('sentence_pause', 0.5)

        # Auto-detect max_new_tokens
        max_new_tokens = data.get('max_new_tokens')
        if max_new_tokens is None:
            text_len = len(text)
            if text_len < 50:
                max_new_tokens = 400
            elif text_len < 100:
                max_new_tokens = 800
            elif text_len < 150:
                max_new_tokens = 1200
            else:
                max_new_tokens = 1500

        # Generate audio
        audio = tts.generate(
            text=text,
            language=language,
            audio_prompt=audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            sentence_pause_ms=int(sentence_pause * 1000),
        )

        # Lưu trực tiếp vào output_path
        tts.save_audio(audio, output_path)

        # Cleanup memory sau mỗi request
        del audio

        # Định kỳ cleanup CUDA cache
        request_counter += 1
        if request_counter % CLEANUP_INTERVAL == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 Cleaned up memory after {request_counter} reque")

        return jsonify({
            "success": True,
            "output_path": output_path,
            "message": "Audio đã được lưu thành công"
        })

    except Exception as e:
        # Cleanup khi có lỗi
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Kiểm tra trạng thái server"""
    return jsonify({"status": "ok", "model": "viterbox"})



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port, debug=False)

