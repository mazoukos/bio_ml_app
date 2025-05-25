# Βασικό image με Python
FROM python:3.10-slim

# Ορισμός του working directory μέσα στο container
WORKDIR /app

# Αντιγραφή του requirements (θα το δημιουργήσουμε)
COPY requirements.txt .

# Εγκατάσταση των απαιτούμενων packages
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή όλου του κώδικα μέσα στο container
COPY . .

# Ανοίγουμε την πόρτα που θα χρησιμοποιήσει το Streamlit
EXPOSE 8501

# Εκτέλεση της εφαρμογής Streamlit όταν ξεκινάει το container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
