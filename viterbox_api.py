from viterbox import Viterbox

# Load model (tự động download từ HuggingFace)
tts = Viterbox.from_pretrained("cuda")

# Tạo audio với giọng mặc định
audio = tts.generate("Xin chào, tôi là Viterbox!")

# Tạo audio với voice cloning
audio = tts.generate(
    text="Xin chào, tôi là Viterbox!",
    language="vi",
    audio_prompt=r"D:\Python\viterbox-tts\wavs\ffe28b36-6b4c-4a1d-a806-90266a1111b1.wav",
    exaggeration=0.5,
    cfg_weight=0.5,
    max_new_tokens=100,
    temperature=0.8,
    sentence_pause_ms=500,
)
tts.save_audio(audio, "output1.wav")
audio = tts.generate(
    text="Xin chào viterbox, tôi là wayto",
    language="vi",
    audio_prompt=r"D:\Python\viterbox-tts\wavs\ffe28b36-6b4c-4a1d-a806-90266a1111b1.wav",
    exaggeration=0.5,
    cfg_weight=0.5,
    max_new_tokens=200,
    temperature=0.8,
    sentence_pause_ms=500,
)
# Lưu file
tts.save_audio(audio, "output.wav")