import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# 모델, 토크나이저, 설정 파일 경로 설정

import os

models_path = os.path.abspath(os.path.join(os.getcwd(), "models"))
tokenizer_path = "facebook/bart-base"
config_path = os.path.join(models_path, "config.json")
generation_config_path = os.path.join(models_path, "generation_config.json")
model_weights_path = os.path.join(models_path, "pytorch_model.bin")

sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample.txt")
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "summary.txt")

#os.chdir(os.path.abspath(os.path.join(os.path.abspath(''), os.pardir)))

# 토크나이저, 모델 초기화 및 설정 파일 로드
tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
model = BartForConditionalGeneration.from_pretrained(
    models_path,
    config=config_path,
    output_attentions=False,
    output_hidden_states=False,
)
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()
generation_config = BartForConditionalGeneration.from_pretrained(
    models_path,
    config=generation_config_path,
    output_attentions=False,
    output_hidden_states=False,
)
generation_config.eval()

# 입력 문장
with open(sample_path, "r", encoding="utf-8") as f:
    input_text = f.read().strip()

# 문장 토크나이징 및 encoding
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 요약문 생성
summary_ids = model.generate(input_ids,
                              num_beams=4,
                              max_length=50,
                              no_repeat_ngram_size=2,
                              length_penalty=2.0,
                              early_stopping=True)

# 요약문 decoding 및 출력
summary_text = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

# 요약문 출력 및 저장
print(summary_text)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(summary_text)