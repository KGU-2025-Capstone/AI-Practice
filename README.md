# AI-Practice
AI 테스트용 리포지토리입니다.

# 파인튜닝이란?
파인튜닝은 기존 GPT 모델에 너만의 데이터를 추가 학습시키는 것입니다. 랭체인이 기존 모델을 그대로 사용한다면, 파인튜닝은 학습 데이터를 통해 기존 모델을 커스터마이징해서 사용한다고 보시면 됩니다.

랭체인은 실시간 DB나 문서를 검색해서 최신 정보를 반영할 수 있고, 수천~수만 건의 문서도 검색해가며 처리 가능하지만 저희는 이미 의약품 API를 통해 데이터를 쓸 예정이고, 약품 설명에는 정확성이 중요하기 때문에 파인튜닝이 조금 더 프로젝트에맞다고 생각합니다. 

### 파인튜닝 장점
- 정확성 향상	도메인(예: 의약품)에 대해 더 정확하게 답변
- 맞춤형 말투	말투, 어투, 표현 방식도 통일 가능
- 빠르고 저렴	반복되는 패턴에 강해서 prompt를 길게 쓸 필요가 없음

# OpenAPI 사용하는 법
https://platform.openai.com/docs/overview 

![image](https://github.com/user-attachments/assets/be668c3a-81f0-4d73-b953-6bf7b1ec0e50)

openapi 사이트에 들어가서 가입을 해주고, API 키를 생성해줍니다. API 키는 다시 확인할 수 없으므로, 따로 저장해두셔야 합니다.

![image](https://github.com/user-attachments/assets/4d6a31eb-6d7a-4cf4-ab63-c0874464ed90)

(API 키를 생성하고 사용하기 위해서는, 결제를 하셔야합니다!)

# 파이썬으로 파인튜닝하기
```
pip install openai
```
시작하기 전에 OpenAI 패키지를 설치해주셔야합니다.

```
from openai import OpenAI

# OpenAI API를 사용할 클라이언트 객체를 생성하고, 발급받은 API 키를 넣어서 인증
client = OpenAI(
    api_key = "서비스키"
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
```
이후 파이썬 코드를 이런 식으로 작성해줍니다. "서비스키" 부분에 아까 받았던 API키를 넣으면 되고, GPT-3.5-turbo 같은 Chat 모델용 파인튜닝에서는 jsonl 파일로 된 학습 데이터가 필요합니다. 

```
{
  "messages": [
    {
      "role": "system",
      "content": "약사처럼 친절하고 이해하기 쉽게 대답해줘."
    },
    {
      "role": "user",
      "content": "이 약은 식욕감퇴(식욕부진), 위부팽만감, 소화불량, 과식, 체함, 구역, 구토에 사용합니다."
    },
    {
      "role": "assistant",
      "content": "이 약은 식욕감퇴, 소화불량, 과식, 체함, 구역, 구토 증상에 사용돼요."
    }
  ]
}
```
- "messages": 역할(role)과 내용(content)로 구성된 대화 리스트
- "system": 모델의 역할 지시 (예: 약사처럼 설명해줘)
- "user": 사용자의 질문
- "assistant": 그 질문에 대한 모델의 정답 (= 학습시키고 싶은 답변)

저희는 이미 데이터베이스에 있는 의약품 데이터를 사용하여 요약한 결과를 보여줄 예정이니, 이를 통한 학습 데이터를 만들어보겠습니다.

실제로 돌아가는 코드에서는 DB에서 조회해서 가져온 제품의 사용법 혹은 증상 혹은 주의사항을 "user"에 넣고, 그에 따른 요약(assistant)을 보여줄 예정입니다. 이 요약의 정확도를 높이기 위해서는, 
위처럼 몇 개의 샘플 데이터를 저희가 직접 혹은 GPT를 통해 요약한 다음 assistant에 넣어줍니다. 이 과정을 여러 번 반복하여 jsonl 파일을 만들면 정교한 학습데이터를 만들 수 있습니다.

```
{"messages": [{"role": "system", "content": "약사처럼 친절하고 이해하기 쉽게 대답해줘."}, {"role": "user", "content": "이 약은 식욕감퇴(식욕부진), 위부팽만감, 소화불량, 과식, 체함, 구역, 구토에 사용합니다."}, {"role": "assistant", "content": "이 약은 식욕감퇴, 소화불량, 과식, 체함, 구역, 구토 증상에 사용돼요."}]}
{"messages": [{"role": "system", "content": "약사처럼 친절하고 이해하기 쉽게 대답해줘."}, {"role": "user", "content": "대환은 1회 1∼2환을 1일 3회 식후에 소량의 물과 함께 복용하십시오.소환은 1회 8∼16환을 1일 3회 식후에 소량의 물과 함께 복용하십시오."}, {"role": "assistant", "content": "대환은 1회 1∼2환, 소환은 1회 8∼16환을 1일 3회 식후에 소량의 물과 함께 복용하면 돼요."}]}
{"messages": [{"role": "system", "content": "약사처럼 친절하고 이해하기 쉽게 대답해줘."}, {"role": "user", "content": "만 7세 이하의 소아는 이 약을 복용하지 마십시오.이 약을 복용하기 전에 혈압이 높은 환자, 고령자, 심장 또는 신장(콩팥)에 장애가 있는 환자, 부종(부기)이 있는 사람은 의사 또는 약사와 상의하십시오.정해진 용법과 용량을 잘 지키십시오.어린이에게 투여할 경우 보호자의 지도 감독하에 투여하십시오."}, {"role": "assistant", "content": "그러나 만 7세 이하 소아나 고혈압, 고령자, 심장·신장 장애, 부종이 있는 사람은 주의하시고, 복용 전에 의사나 약사와 상의를 하시는게 좋습니다!"}]}
{"messages": [{"role": "system", "content": "약사처럼 친절하고 이해하기 쉽게 대답해줘."}, {"role": "user", "content": "이 약은 감기의 여러 증상(콧물, 코막힘, 재채기, 인후(목구멍)통, 기침, 가래, 오한(춥고 떨리는 증상), 발열, 두통, 관절통, 근육통)의 완화에 사용합니다."}, {"role": "assistant", "content": "이 약은 감기의 증상(콧물, 코막힘, 재채기, 인후통, 기침, 가래, 오한, 발열, 두통, 근육통, 관절통) 완화에 사용돼요."}]}
```
예시

![image](https://github.com/user-attachments/assets/60a566bb-f68f-4ceb-a4a3-21b10a9703ea)

그 다음에는 위에 파이썬 코드를 실행시키면 OpenAI의 finetuning 탭에 모델이 학습하는 과정을 볼 수 있을 겁니다. 몇 분 기다려주면 학습이 끝나는데, 그러면 저런 식으로
Ouput Model이 뜨게 됩니다. 우리는 이걸 사용할 것입니다.

# 파이썬 파인튜닝한 GPT 모델 사용 예시
```
from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI

client = OpenAI(
    api_key = "서비스키"
)

app = Flask(__name__)

# MongoDB URI
uri = "몽고DB URI (민감한 정보라 표시하지 않겠습니다)"

# MongoDB 연결
client2 = MongoClient(uri)

# 데이터베이스 선택
db = client2['SpringDatabaseApi']

# 컬렉션 선택
collection = db['Api']  # 컬렉션 이름을 적어주세요.

@app.route('/medicine/summary', methods=['POST'])
def summarize_medicine():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "query 파라미터가 필요합니다."}), 400

    # 1. 제품명 또는 효능에서 검색
    result = collection.find_one({
        "$or": [
            {"itemName": {"$regex": query, "$options": "i"}},
            {"efcyQesitm": {"$regex": query, "$options": "i"}}
        ]
    })

    if not result:
        return jsonify({"error": "관련 약을 찾을 수 없습니다."}), 404

    # 2. 필요한 정보 추출
    item_name = result.get("itemName", "")
    effect = result.get("efcyQesitm", "")
    usage = result.get("useMethodQesitm", "")
    caution = result.get("atpnQesitm", "")

    context_text = f"""
    효능: {effect}
    복용법: {usage}
    주의사항: {caution}
    """

    # 3. OpenAI에 요약 요청
    response =  client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BFDFfQvI",
        messages=[
            {"role": "system", "content": "Act like a pharmacist and explain the medicine clearly in English."},
            {"role": "user", "content": f"{context_text}"}
        ]
    )

    summary = response.choices[0].message.content
    return jsonify({
        "itemName": item_name,
        "summary": summary
    })


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
```
![image](https://github.com/user-attachments/assets/52f7f633-67f1-4d6a-b5cc-1916be02a6a0)

이후 이런 식으로 코드를 작성해서, 실행시킨 후 query에 예시로 요청하면, ai가 학습한 결과를 토대로 요약 데이터를 생성하여 줍니다.
