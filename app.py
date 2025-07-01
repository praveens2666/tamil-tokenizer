# import sys
# sys.path.append("c:/users/ramac/installation-packages")
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
import sentencepiece as spm
import os
import regex as re
import string
import psycopg2

# ========== INIT APP ==========
app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# ========== LOAD SENTENCEPIECE MODEL ==========
sp = spm.SentencePieceProcessor()
sp.load("tamil_tokenizer.model")

# ========== UTILITY FUNCTIONS ==========
def grapheme_tokenize(text):
    pattern = re.compile(r'\X', re.UNICODE)
    return pattern.findall(text)

def clean_word(word):
    return word.strip(string.punctuation + " ").strip()

# ========== PARSE SPECIAL_SPLITS FILE ==========
def tokenize(inplines):
    res = {}
    i = 0
    while i < len(inplines):
        line = inplines[i].strip().replace("\t", " ")
        if line.startswith("#"):
            i += 1
            continue
        elif len(line) > 1 and line[1] == '-':
            key = line[3:].replace(" ", "")
            dif = int(line[2]) - int(line[0]) + i
            en = dif + 1
            split = []
            while i < en:
                i += 1
                split.append(inplines[i][2:].strip())
            res[key] = split
        else:
            res[line[2:].strip()] = [line[2:].strip()]
        i += 1
    return res

def getTokens(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return tokenize(file.readlines())

special_splits = getTokens("filtered_output_file.txt")

# ========== DB CONNECTION ==========
def get_db_connection():
    return psycopg2.connect(
        dbname="tokenizer_db",
        user="postgres",
        password="praveen2666",
        host="localhost",
        port="5432"
    )

def get_corrected_token(original):
    original = clean_word(original)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT corrected FROM corrections WHERE original = %s", (original,))
    row = cursor.fetchone()
    conn.close()
    return row[0].split() if row else None

# ========== MAIN TOKENIZER CLASS ==========
class TamilTokenizer:
    def __init__(self, sp_model, special_splits):
        self.sp = sp_model
        self.special_splits = special_splits

    def tokenize(self, text):
        for punct in ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']']:
            text = text.replace(punct, f" {punct}")
        words = text.split()
        segmented_words = []

        for word in words:
            base = clean_word(word)

            # 1️⃣ First check DB
            corrected = get_corrected_token(base)

            # 2️⃣ Then check special splits
            if corrected:
                subwords = corrected
            elif base in self.special_splits:
                subwords = self.special_splits[base]
                
            else:
                # 3️⃣ Else SentencePiece fallback
                pieces = self.sp.encode_as_pieces(base)
                subwords = [p for p in pieces if len(p) > 1] or [base]
                sub = "".join(subwords)
                subwords=[sub]
                
            segmented_words.extend(subwords)

        return segmented_words

tamil_tokenizer = TamilTokenizer(sp, special_splits)

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/tokenize', methods=['POST'])
def tokenize_route():
    algo = request.form.get("algorithm")
    text = request.form.get("text")
    file = request.files.get("file")

    if file:
        text = file.read().decode("utf-8").strip()

    if not text or not algo:
        return jsonify({"error": "Missing input"}), 400

    if algo == "xlmr":
        tokens = tokenizer.tokenize(text)
    elif algo == "sentencepiece":
        tokens = tamil_tokenizer.tokenize(text)
    elif algo == "grapheme":
        tokens = grapheme_tokenize(text)
    else:
        return jsonify({"error": "Unknown algorithm"}), 400

    return jsonify({"tokens": tokens})

@app.route('/save_correction_batch', methods=['POST'])
def save_correction_batch():
    data = request.json
    corrections = data.get("corrections", [])

    if not corrections:
        return jsonify({"message": "No corrections found."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    for corr in corrections:
        original = clean_word(corr.get("word", ""))
        corrected = " ".join(corr.get("corrected", []))
        expert = corr.get("expert", "unknown")

        if original and corrected:
            cursor.execute("DELETE FROM corrections WHERE original = %s", (original,))
            cursor.execute(
                "INSERT INTO corrections (original, corrected, expert_name) VALUES (%s, %s, %s)",
                (original, corrected, expert)
            )

    conn.commit()
    conn.close()

    return jsonify({"message": f"Saved {len(corrections)} corrections successfully!"})

@app.route('/decode_tokens', methods=['POST'])
def decode_tokens():
    data = request.get_json()
    tokens = data.get("tokens", [])

    if not tokens:
        return jsonify({"decoded": ""})

       # Convert to token IDs first
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Decode using token IDs
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    return jsonify({"decoded": decoded_text})



# ========== RUN APP ==========
if __name__ == "__main__":
    app.run(debug=True)
