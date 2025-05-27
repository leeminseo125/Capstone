# Capstone
류연준 이민서 우용석 

## 가상환경
python=3.11.11
conda install --file requirements.txt

## 데이터 이동
- need_code 디렉토리에서
- 각각 다른 터미널에서
- 스트리밍 서버 열기
```
uvicorn video_loop_server:app --host 0.0.0.0 --port 8080 --reload
```
- 추론 서버 열기
```
uvicorn yolo_server_api_re:app --reload
```
- 코어 서버 열기
```
streamlit run app_streamlit.py
```

## DB
- 아직 데이터 이동 같은 세부적인 구현 못함
- 이번 주에 구현 예정

## 기타
- src/detect에 있는 파일들은 연습용 파일
- 실질적 사용 파일은 ./need_code