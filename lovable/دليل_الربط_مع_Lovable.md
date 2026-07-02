# ربط CashYar Dataset مع Lovable

## الخطوات (15 دقيقة)

### 1) أنشئ مشروع Supabase مجاني
1. ادخل [supabase.com](https://supabase.com) وسجّل
2. **New Project** → اختر اسم مثل `cashyar`
3. انسخ من **Settings → API**:
   - **Project URL** → مثل: `https://abcdefgh.supabase.co`
   - **anon public key**

> هذا الرابط (`Project URL`) هو اللي تربطه في Lovable.

---

### 2) أنشئ الجداول
1. في Supabase → **SQL Editor** → **New query**
2. انسخ محتوى ملف `schema.sql` والصقه → **Run**

---

### 3) ارفع البيانات (CSV)

**للتطوير السريع** (عينة 500 مستخدم — ~9 MB):
| الجدول | الملف |
|--------|-------|
| users | `lovable/sample/users.csv` |
| transactions | `lovable/sample/transactions.csv` |
| behavioral_summary | `lovable/sample/behavioral_summary.csv` |

**للبيانات الكاملة** (5,000 مستخدم — ~92 MB):
| الجدول | الملف |
|--------|-------|
| users | `data/users.csv` (0.8 MB) |
| transactions | `data/transactions.csv` (82 MB) |
| behavioral_summary | `data/behavioral_summary.csv` (9 MB) |

**طريقة الرفع:**
1. Supabase → **Table Editor** → اختر الجدول
2. **Insert** → **Import data from CSV**
3. ارفع الملف المناسب (users أولاً، ثم transactions، ثم behavioral_summary)

---

### 4) اربط Lovable مع Supabase
1. افتح [lovable.dev](https://lovable.dev) → مشروع جديد
2. **Settings / Integrations** → **Connect Supabase**
3. الصق **Project URL** و **anon key**

---

### 5) الصق الـ Prompt في Lovable
انسخ محتوى ملف `LOVABLE_PROMPT.txt` والصقه في شات Lovable.

---

## الروابط اللي تحتاجها

| الغرض | الرابط |
|-------|--------|
| إنشاء Supabase | https://supabase.com/dashboard |
| بناء التطبيق | https://lovable.dev |
| توثيق الربط | https://docs.lovable.dev/integrations/supabase |

**بعد الربط، رابط مشروعك يكون:**
```
https://YOUR-PROJECT-ID.supabase.co
```
وهذا اللي Lovable يستخدمه لجلب البيانات.

---

## إذا تبي رابط GitHub للملفات

ارفع المشروع على GitHub ثم استخدم روابط raw:

```
https://raw.githubusercontent.com/YOUR-USER/cashyar/main/lovable/sample/users.csv
https://raw.githubusercontent.com/YOUR-USER/cashyar/main/lovable/sample/transactions.csv
https://raw.githubusercontent.com/YOUR-USER/cashyar/main/lovable/sample/behavioral_summary.csv
```

---

## ملخص Dataset

| الجدول | صفوف (كامل) | صفوف (عينة) |
|--------|------------:|------------:|
| users | 5,000 | 500 |
| transactions | 499,031 | 49,742 |
| behavioral_summary | 59,984 | 5,999 |
