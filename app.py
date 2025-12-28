import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# --- Backend Functions (unchanged) ---
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    # --- Gradient Header ---
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #36D1DC, #5B86E5);
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;">
            <h1 style="color:white;margin:0;font-family: 'Segoe UI', sans-serif;">📄 Resume Category Prediction</h1>
        </div>
        """, unsafe_allow_html=True
    )

    # Description
    st.markdown(
        """
        <p style="text-align:center;font-size:16px;color:#555;margin-top:15px;font-family: 'Segoe UI', sans-serif;">
        Upload your resume in <b>PDF</b>, <b>DOCX</b>, or <b>TXT</b> format and get the predicted job category instantly.
        </p>
        """, unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted successfully!")

            col1, col2 = st.columns([2, 1])

            # Left column: extracted text
            with col1:
                st.markdown(
                    """
                    <div style="background-color:#f1f1f1;padding:15px;border-radius:10px;
                                box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
                                transition: all 0.3s ease;">
                        <h3 style="color:#333;">Extracted Resume Text</h3>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.text_area("", resume_text, height=350)

            # Right column: prediction card with dynamic background
            with col2:
                category = pred(resume_text)
                # Dynamic gradient based on category length (simple variation)
                gradient_colors = ["#FF758C", "#FF7EB3", "#36D1DC", "#5B86E5", "#43e97b", "#38f9d7"]
                color_index = len(category) % len(gradient_colors)
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, {gradient_colors[color_index]}, #f9a3b8);
                                padding:35px;
                                border-radius:15px;
                                text-align:center;
                                color:white;
                                box-shadow: 0px 6px 25px rgba(0,0,0,0.35);
                                transition: transform 0.3s ease, box-shadow 0.3s ease;
                                cursor: pointer;"
                                onmouseover="this.style.transform='scale(1.05)';this.style.boxShadow='0px 8px 35px rgba(0,0,0,0.5)';"
                                onmouseout="this.style.transform='scale(1)';this.style.boxShadow='0px 6px 25px rgba(0,0,0,0.35)';">
                        <h3 style="margin-bottom:15px;font-family: 'Segoe UI', sans-serif;">Predicted Category</h3>
                        <h1 style="font-size:40px;margin:0;font-family: 'Segoe UI', sans-serif;">{category}</h1>
                    </div>
                    """, unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
