import streamlit as st
import pandas as pd
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import asyncio

file_path = "dev/Two_Face_Chatbot.csv"
df = pd.read_csv(file_path)

df["ジャンル"] = df["ジャンル"].str.strip()
df["口調レベル"] = df["口調レベル"].str.strip()



    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        return pred

    tokenizer = BertTokenizer.from_pretrained("./TFC_model")
    model = BertForSequenceClassification.from_pretrained("./TFC_model")

    model.eval()

    user_input = st.text_input("あなたの悩みをどうぞ")

    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_id = torch.argmax(logits, dim=1).item()

    id2label = {
        0: "やる気が出ない",
        1: "将来が不安",
        2: "自己否定",
        3: "人間関係がつらい",
        4: "眠れない・疲れがとれない"
    }

    genre = id2label[predicted_class_id]

    st.title("メンタル相談チャットボット")
    st.write("スライダーでBotの優しさレベルを調整して、相談してみてください。")

    tone_level = st.slider("Botのやさしさレベル", 0, 10, 5)


    def get_response_df(genre, tone_level):
        if tone_level <= 3:
            tone = "厳しめ"
        elif tone_level <= 7:
            tone = "普通"
        else:
            tone = "優しめ"

        filtered = df[(df["ジャンル"] == genre) & (df["口調レベル"] == tone)]

        if not filtered.empty:
            return filtered.iloc[0]["応答例"]
        else:
            return "そのジャンルと口調の組み合わせにはまだ応答がありません。"


    if user_input:
        response = get_response_df(genre, tone_level)
        st.write(f"Bot:{response}")


if __name__ == "__main__":
    asyncio.run(main())
