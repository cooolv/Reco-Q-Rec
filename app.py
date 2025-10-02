import os
import librosa
import numpy as np
import requests
import json
import streamlit as st
from scipy.spatial.distance import cosine

# ====== إعداد الملفات ======
DATA_FILE = "reciters_db.json"
FEATURES_DIR = "features_cache"
TEMP_AUDIO = "temp_upload.wav"

# ====== إنشاء مجلد الميزات إذا لم يكن موجودًا ======
os.makedirs(FEATURES_DIR, exist_ok=True)

# ====== استخراج الميزات ======
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.tolist()

# ====== تحميل الصوت ======
def download_audio(url, save_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)

# ====== قراءة قاعدة البيانات ======
def load_database():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ====== حفظ قاعدة البيانات ======
def save_database(db_data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db_data, f, ensure_ascii=False, indent=4)

# ====== حفظ الميزات ======
def save_features(name, features):
    path = os.path.join(FEATURES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(features, f)

# ====== تحميل الميزات ======
def load_features(name):
    path = os.path.join(FEATURES_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ====== بناء قاعدة الميزات ======
def build_features_db():
    db_data = load_database()
    features_db = []

    for entry in db_data:
        features = load_features(entry["name"])
        if not features:
            st.info(f"تحميل ومعالجة: {entry['name']}")
            try:
                audio_path = os.path.join(FEATURES_DIR, f"{entry['name']}_temp.wav")
                download_audio(entry["audio_url"], audio_path)
                features = extract_features(audio_path)
                save_features(entry["name"], features)
            except Exception as e:
                st.error(f"فشل تحميل/معالجة {entry['name']}: {e}")
                continue
        entry["features"] = features
        features_db.append(entry)

    return features_db

# ====== التعرف على القارئ ======
def recognize_reciter(test_features, features_db):
    best_match = None
    best_score = float("inf")

    for entry in features_db:
        score = cosine(test_features, entry["features"])
        if score < best_score:
            best_score = score
            best_match = entry

    return best_match, best_score

# ====== واجهة Streamlit ======
st.title("📖 تحديد قارئ القرآن الكريم")
st.write("ارفع مقطع صوتي وسأحاول تحديد القارئ باستخدام قاعدة بيانات محلية.")

uploaded_file = st.file_uploader("اختر ملف صوتي", type=["wav", "mp3"])

if uploaded_file:
    with open(TEMP_AUDIO, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("تحديد القارئ"):
        with st.spinner("جارٍ التحليل..."):
            features_db = build_features_db()
            test_features = extract_features(TEMP_AUDIO)
            match, score = recognize_reciter(test_features, features_db)

        if match:
            st.success(f"📌 القارئ المحتمل هو: **{match['name']}**")
            st.write(f"🔗 مصدر: {match['source']}")
            st.write(f"🧮 درجة التشابه: {round(1-score, 2)}")
        else:
            st.error("لم يتم العثور على معلومات القارئ.")

# ====== إضافة قارئ جديد ======
st.header("➕ إضافة قارئ جديد")
with st.form(key="add_reciter_form"):
    name = st.text_input("اسم القارئ")
    audio_url = st.text_input("رابط التلاوة المباشرة")
    source = st.text_input("رابط المصدر")
    submit_button = st.form_submit_button(label="إضافة القارئ")

    if submit_button:
        if name and audio_url and source:
            try:
                st.info("جارٍ تحميل ومعالجة التلاوة...")
                audio_path = os.path.join(FEATURES_DIR, f"{name}_new.wav")
                download_audio(audio_url, audio_path)
                features = extract_features(audio_path)

                db_data = load_database()
                db_data.append({
                    "name": name,
                    "audio_url": audio_url,
                    "source": source,
                    "features": features
                })
                save_database(db_data)
                save_features(name, features)

                st.success("✅ تم إضافة القارئ بنجاح مع تخزين الميزات!")
            except Exception as e:
                st.error(f"❌ فشل إضافة القارئ: {e}")
        else:
            st.error("❌ الرجاء ملء جميع الحقول.")
