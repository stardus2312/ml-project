# Drug Quality Classification with Machine Learning

เว็บแอปพลิเคชันสำหรับการจำแนกคุณภาพยาด้วย Machine Learning โดยใช้ Streamlit

## 🚀 การติดตั้งและรัน

### 1. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### 2. เตรียมไฟล์โมเดล
ให้แน่ใจว่าคุณมีไฟล์โมเดลทั้ง 3 ไฟล์ในโฟลเดอร์เดียวกัน:
- `Decision_bayes_model.pkl`
- `logistic_regression_model.pkl`
- `naive_bayes_model.pkl`

### 3. รันแอปพลิเคชัน
```bash
streamlit run app.py
```

## 📋 ฟีเจอร์

- **เลือกโมเดล**: สามารถเลือกใช้โมเดล ML ได้ 3 แบบ
  - Decision Tree
  - Logistic Regression  
  - Naive Bayes

- **ป้อนข้อมูลยา**: 
  - ป้อนข้อมูลด้วยตนเอง
  - เลือกจากข้อมูลตัวอย่าง

- **การทำนาย**: แสดงผลการทำนายคุณภาพยาพร้อมความเชื่อมั่น

## 🎯 ข้อมูลที่ใช้

แอปพลิเคชันใช้ข้อมูลคุณสมบัติทางเคมีของยา:
- Molecular Weight (น้ำหนักโมเลกุล)
- LogP (ความชอบไขมัน)
- HBD (จำนวน Hydrogen Bond Donors)
- HBA (จำนวน Hydrogen Bond Acceptors)
- RotB (จำนวน Rotatable Bonds)
- TPSA (Topological Polar Surface Area)
- Aromatic Rings (จำนวนวงแหวนอะโรมาติก)
- Heavy Atoms (จำนวนอะตอมหนัก)

## 🔧 การใช้งาน

1. เลือกโมเดลที่ต้องการใช้จาก sidebar
2. เลือกวิธีการป้อนข้อมูล (Manual Input หรือ Sample Data)
3. ป้อนข้อมูลยาหรือเลือกจากตัวอย่าง
4. กดปุ่ม "Predict Quality" เพื่อดูผลการทำนาย

## 📊 ผลลัพธ์

แอปพลิเคชันจะแสดง:
- ชื่อยา
- โมเดลที่ใช้
- ผลการทำนายคุณภาพ
- ระดับความเชื่อมั่น (Confidence)

## 🛠️ การแก้ไขปัญหา

หากพบปัญหา:
1. ตรวจสอบว่าไฟล์ .pkl อยู่ในโฟลเดอร์เดียวกัน
2. ตรวจสอบเวอร์ชัน dependencies ใน requirements.txt
3. ตรวจสอบ log error ใน terminal

