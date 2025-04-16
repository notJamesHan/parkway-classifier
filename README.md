# parkway-classifier
박명수가 말했을 만한 어록 예측기: Would comedian Myung-Soo Park say this saying?

Practicing making prediction model, make service API for the model,
setup infrastructre

Park Myung-Soo data from: hand collected from 나무위키 and googled.
Chat Bot data from: https://github.com/songys/Chatbot_data

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt

uvicorn api.main:app --reload
```

