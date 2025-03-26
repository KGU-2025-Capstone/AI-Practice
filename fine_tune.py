from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경변수 읽기
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API를 사용할 클라이언트 객체를 생성하고, 발급받은 API 키를 넣어서 인증
client = OpenAI(
    api_key = openai_api_key
)

# 학습시킬jsonl.jsonl 파일을 OpenAI 서버에 업로드함, 파일 목적은 파인튜닝 학습용으로 지정
file = client.files.create(
  file=open("학습시킬jsonl.jsonl", "rb"),
  purpose="fine-tune"
)

# 방금 업로드한 파일을 이용해 파인튜닝 학습을 시작, model="gpt-3.5-turbo"는 기반 모델로 GPT-3.5-turbo를 사용하겠다는 뜻
client.fine_tuning.jobs.create(
  training_file=file.id,
  model="gpt-3.5-turbo"
)