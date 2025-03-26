from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경변수에서 값 읽기
openai_api_key = os.environ.get("OPENAI_API_KEY")
mongo_uri = os.environ.get("MONGODB_URI")

client = OpenAI(
    api_key = openai_api_key
)

app = Flask(__name__)

# MongoDB 연결
mongo_client = MongoClient(mongo_uri)
db = mongo_client['SpringDatabaseApi']
collection = db['Api']

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
            {"role": "system", "content": "약사처럼 친절하고 이해하기 쉽게 대답해줘."},
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
