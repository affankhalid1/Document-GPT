# Document GPT

Document GPT is an AI-powered chatbot designed to assist users by answering questions based on uploaded documents. By leveraging natural language processing, it extracts key information and provides insightful responses, making it easier for users to navigate and understand complex content. Ideal for students, researchers, and professionals, Document GPT enhances information retrieval and knowledge sharing.

## Steps to run Document GPT locally:

### 1. Set up API Key:
- Create a `.streamlit` folder in the root directory.
- Inside the folder, create a file named `secrets.toml` with the following content:
    ```toml
    HUGGING_FACE_API_KEY = "<Enter_your_HuggingFace_Api_key>"
    ```

### 2. Create a Conda Environment and Install Dependencies:
- Run the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Run the App on Localhost:
- Start the application by running:
    ```bash
    streamlit run streamlit_bot_app.py
    ```

## Additional Information:
Document GPT uses Streamlit for an interactive interface and Hugging Face API for natural language processing capabilities. It allows users to upload documents, enabling the chatbot to extract essential information and answer questions based on the content. This tool is valuable for simplifying the process of information retrieval and knowledge sharing across various fields.

---

Follow these steps to get started, and enjoy using Document GPT as your AI-powered assistant for document-based learning and exploration!
