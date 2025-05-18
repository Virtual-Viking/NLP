# pip install streamlit pandas matplotlib nltk 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
import nltk
import string
# import io

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set page title
st.set_page_config(page_title="Wordsmith - Text Analysis Suite", layout="wide")

# Add custom CSS after the page config to constrain the width
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1024px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    div.stButton > button.tf-idf-btn {
        background-color: #4CAF50; 
        color: white;
        padding: 10px 24px;
        margin: 10px 0px;
        font-weight: bold;
        border-radius: 4px;
        border: none;
    }
    div.stButton > button.tf-idf-btn:hover {
        background-color: #45a049;
    }
    div.stButton > button#tf-idf-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-weight: bold;
        border-radius: 4px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Wordsmith - Msg Analysis Suite")

st.write("Please upload the dataset")
uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])

# Function to safely load data with error handling
def safe_read_csv(file):
    try:
        # Try different encodings
        for encoding in ['utf-8', 'ISO-8859-1', 'latin1']:
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Read a sample of the file to check if it's valid
                sample = file.read(1024).decode(encoding)
                file.seek(0)
                
                # Check if there's content in the file
                if not sample.strip():
                    return None, "The uploaded file appears to be empty."
                
                # Try to read with pandas
                df = pd.read_csv(file, encoding=encoding)
                
                # Basic validation - check if we have data
                if df.empty:
                    return None, "The file was loaded but contains no data."
                
                # Check if we have the required columns
                if 'v1' not in df.columns or 'v2' not in df.columns:
                    # Try to infer column names if missing
                    if len(df.columns) >= 2:
                        # Rename columns if they exist but have different names
                        new_cols = list(df.columns)
                        new_cols[0] = 'v1'
                        new_cols[1] = 'v2'
                        df.columns = new_cols
                    else:
                        return None, "The file doesn't contain the required columns (v1 for class, v2 for message)."
                
                return df, None
            except UnicodeDecodeError:
                # Try the next encoding
                continue
            except Exception as e:
                # Other errors
                return None, f"Error reading file with {encoding} encoding: {str(e)}"
        
        # If we get here, none of the encodings worked
        return None, "Unable to read the file with any supported encoding."
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"

if uploaded_file is not None:
    # Try to load the data safely
    df, error_message = safe_read_csv(uploaded_file)
    
    if error_message:
        st.error(f"Error loading file: {error_message}")
        st.info("Try uploading a different file or check the file format. The file should be a CSV with at least two columns for class labels and messages.")
        
        # Optionally, show a sample of the file content for debugging
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read(2048).decode('utf-8', errors='replace')
            with st.expander("View file content sample"):
                st.text(content)
        except:
            st.write("Unable to display file content.")
    else:
        st.success("File uploaded and processed successfully!")
        
        # Display the data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), height=200)
        
        # Add the "View Full Dataset" button aligned to the right
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("View Full Dataset", key="view_dataset_btn"):
                st.session_state['show_full_dataset'] = True

        # Create a separate section for the full dataset view
        if 'show_full_dataset' in st.session_state and st.session_state['show_full_dataset']:
            # Create a full-page section with its own header
            st.markdown("---")
            st.subheader("Full Dataset")
            
            # Create a search box
            search_term = st.text_input("Search in messages:")
            
            # Add pagination controls in a row
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                page_size = st.selectbox("Rows per page:", [10, 25, 50, 100])
            
            # Calculate total pages based on page size
            total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
            
            with col2:
                page_number = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
            
            with col3:
                # Add a "Close View" button
                if st.button("Close Full View", key="close_full_view"):
                    st.session_state['show_full_dataset'] = False
            
            # Calculate start and end indices for current page
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            
            # Filter and display data
            if search_term:
                filtered_df = df[df['v2'].str.contains(search_term, case=False, na=False)]
                st.write(f"Found {len(filtered_df)} messages containing '{search_term}'")
                st.dataframe(filtered_df, height=600, use_container_width=True)
            else:
                # Show paginated data
                st.write(f"Showing rows {start_idx+1} to {end_idx} of {len(df)}")
                st.dataframe(df.iloc[start_idx:end_idx], height=600, use_container_width=True)
            
            # Add download button for the full dataset
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Dataset",
                data=csv,
                file_name="sms_dataset.csv",
                mime="text/csv",
            )

        # Add the text preprocessing section if data is loaded successfully
        st.markdown("## Text Preprocessing")
        
        # Create columns for the preprocessing options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lowercase_option = st.checkbox("Convert to lowercase", value=True)
            remove_punct_option = st.checkbox("Remove punctuation", value=True)
        
        with col2:
            remove_stopwords_option = st.checkbox("Remove stopwords", value=True)
        
        with col3:
            remove_numbers_option = st.checkbox("Remove numbers", value=True)
            stemming_option = st.checkbox("Apply stemming", value=True)
        
        # Style the Start Preprocessing button
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FF5252;
            color: white;
            padding: 8px 16px;
            font-weight: bold;
            border-radius: 4px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #FF7575;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Define preprocessing functions
        def lowercase_text(text):
            if isinstance(text, str) and lowercase_option:
                return text.lower()
            else:
                return text if isinstance(text, str) else ""
        
        def remove_punctuation(text):
            if isinstance(text, str) and remove_punct_option:
                return text.translate(str.maketrans('', '', string.punctuation))
            else:
                return text if isinstance(text, str) else ""
        
        def remove_stopwords(text):
            if isinstance(text, str) and remove_stopwords_option:
                tokens = text.split()
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [word for word in tokens if word not in stop_words]
                return ' '.join(filtered_tokens)
            else:
                return text if isinstance(text, str) else ""
        
        def remove_numbers(text):
            if isinstance(text, str) and remove_numbers_option:
                return ''.join([i for i in text if not i.isdigit()])
            else:
                return text if isinstance(text, str) else ""
        
        def apply_stemming(text):
            if isinstance(text, str) and stemming_option:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                tokens = text.split()
                stemmed_tokens = [stemmer.stem(word) for word in tokens]
                return ' '.join(stemmed_tokens)
            else:
                return text if isinstance(text, str) else ""
        
        def tokenize_text(text):
            if isinstance(text, str):
                return word_tokenize(text)
            else:
                return []
        
        # Initialize a key to track if preprocessing should be shown
        if 'show_preprocessing' not in st.session_state:
            st.session_state['show_preprocessing'] = False
        
        # Button to start preprocessing and show example
        start_preprocessing = st.button("Start Preprocessing")
        
        if start_preprocessing:
            st.session_state['show_preprocessing'] = True
        
        # If preprocessing should be shown, display the example
        if st.session_state['show_preprocessing']:
            # Get the first row as an example
            if len(df) > 0:
                example_row = df.iloc[0].copy()
                
                # Process the example
                example = {}
                example['original'] = example_row['v2']
                example['message_type'] = example_row['v1'] if 'v1' in df.columns else "Unknown"
                
                # Apply preprocessing steps
                current_text = example['original']
                
                if lowercase_option:
                    current_text = lowercase_text(current_text)
                    example['lowercase'] = current_text
                
                if remove_punct_option:
                    current_text = remove_punctuation(current_text)
                    example['no_punct'] = current_text
                
                if remove_stopwords_option:
                    current_text = remove_stopwords(current_text)
                    example['no_stopwords'] = current_text
                
                if remove_numbers_option:
                    current_text = remove_numbers(current_text)
                    example['no_numbers'] = current_text
                
                if stemming_option:
                    current_text = apply_stemming(current_text)
                    example['stemmed'] = current_text
                
                example['tokens'] = tokenize_text(current_text)
                example['processed_text'] = current_text
                
                # Display the preprocessing example
                st.subheader("Preprocessing Example")
                
                # Original message
                st.markdown(f"### Sample Text ({example['message_type']})")
                st.markdown("**Original Message:**")
                st.code(example['original'])
                
                # Show each applied step
                step_num = 1
                
                if lowercase_option:
                    st.markdown(f"**Step {step_num}: Convert to Lowercase**")
                    st.code(example['lowercase'])
                    step_num += 1
                
                if remove_punct_option:
                    st.markdown(f"**Step {step_num}: Remove Punctuation**")
                    st.code(example['no_punct'])
                    step_num += 1
                
                if remove_stopwords_option:
                    st.markdown(f"**Step {step_num}: Remove Stopwords**")
                    st.code(example['no_stopwords'])
                    step_num += 1
                
                if remove_numbers_option:
                    st.markdown(f"**Step {step_num}: Remove Numbers**")
                    st.code(example['no_numbers'])
                    step_num += 1
                
                if stemming_option:
                    st.markdown(f"**Step {step_num}: Apply Stemming**")
                    st.code(example['stemmed'])
                
                # Tokenization
                st.markdown("**Tokenization:**")
                st.code(str(example['tokens']))
                
                # Final Processed Text
                st.markdown("**Final Processed Text:**")
                st.code(example['processed_text'])
                
                # Automatically process the full dataset
                st.subheader("Processed Dataset")
                
                # Show progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process in batches to show progress
                total_steps = sum([lowercase_option, remove_punct_option, remove_stopwords_option, 
                                remove_numbers_option, stemming_option])
                current_step = 0
                
                status_text.text("Processing dataset...")
                progress_bar.progress(0)
                
                # Apply preprocessing steps sequentially
                df_processed = df.copy()
                
                if lowercase_option:
                    current_step += 1
                    status_text.text(f"Step {current_step}/{total_steps}: Converting to lowercase...")
                    df_processed['processed_text'] = df_processed['v2'].apply(lowercase_text)
                    progress_bar.progress(current_step/total_steps)
                else:
                    df_processed['processed_text'] = df_processed['v2']
                
                if remove_punct_option:
                    current_step += 1
                    status_text.text(f"Step {current_step}/{total_steps}: Removing punctuation...")
                    df_processed['processed_text'] = df_processed['processed_text'].apply(remove_punctuation)
                    progress_bar.progress(current_step/total_steps)
                
                if remove_stopwords_option:
                    current_step += 1
                    status_text.text(f"Step {current_step}/{total_steps}: Removing stopwords...")
                    df_processed['processed_text'] = df_processed['processed_text'].apply(remove_stopwords)
                    progress_bar.progress(current_step/total_steps)
                
                if remove_numbers_option:
                    current_step += 1
                    status_text.text(f"Step {current_step}/{total_steps}: Removing numbers...")
                    df_processed['processed_text'] = df_processed['processed_text'].apply(remove_numbers)
                    progress_bar.progress(current_step/total_steps)
                
                if stemming_option:
                    current_step += 1
                    status_text.text(f"Step {current_step}/{total_steps}: Applying stemming...")
                    df_processed['processed_text'] = df_processed['processed_text'].apply(apply_stemming)
                    progress_bar.progress(current_step/total_steps)
                
                # Update progress
                progress_bar.progress(1.0)
                
                # Save preprocessed data to session state
                st.session_state['preprocessed_df'] = df_processed
                
                # Display success message
                st.markdown("""
                <div class="success-message">
                    ✅ Preprocessing completed successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Preview the processed dataset
                st.subheader("Processed Dataset Preview")
                st.dataframe(df_processed[['v2', 'processed_text']].head())
                
                # Option to download preprocessed data
                preprocessed_csv = df_processed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Preprocessed Dataset",
                    data=preprocessed_csv,
                    file_name="preprocessed_sms_dataset.csv",
                    mime="text/csv",
                )
                
                # Show TF-IDF button
                st.markdown("---")
                st.subheader("Text Analysis Options")
                
                # Create TF-IDF button with a unique state variable
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("TF-IDF Analysis", key="tf_idf_btn", use_container_width=True, 
                            help="Analyze term frequencies and importance using TF-IDF"):
                        # FIX: Use the same state variable name that's checked later
                        st.session_state['show_tf_idf'] = True
            else:
                st.error("Dataset is empty, no example to show.")

        # Model Building Section
if 'preprocessed_df' in st.session_state:
    st.markdown("---")
    st.subheader("Model Building")
    
    # Create a button for model building
    if st.button("Build Classification Models", key="build_models_btn", use_container_width=True,
                help="Train machine learning models to classify messages as spam or ham"):
        st.session_state['show_models'] = True

# TF-IDF Analysis Section - FIX: Move this section before Model Analysis
if 'show_tf_idf' in st.session_state and st.session_state['show_tf_idf'] and 'preprocessed_df' in st.session_state:
    st.markdown("---")
    st.title("TF-IDF Analysis")
    
    # Get the preprocessed data
    df_processed = st.session_state['preprocessed_df']
    
    # Import necessary libraries for TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Show a progress indicator for TF-IDF computation
    with st.spinner("Computing TF-IDF..."):
        # Create a TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        
        # Apply TF-IDF to the processed text
        tfidf_matrix = vectorizer.fit_transform(df_processed['processed_text'])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create a DataFrame with TF-IDF values
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    # Display success message for TF-IDF computation
    st.markdown("""
    <div class="success-message">
        ✅ TF-IDF computation completed successfully!
    </div>
    """, unsafe_allow_html=True)
    
    # Display TF-IDF information
    st.subheader("TF-IDF Overview")
    st.write(f"Number of documents (messages): {tfidf_matrix.shape[0]}")
    st.write(f"Number of terms (vocabulary): {len(feature_names)}")
    
    # Display TF-IDF DataFrame preview
    st.subheader("TF-IDF Matrix Preview (First 5 rows, First 10 columns)")
    st.dataframe(tfidf_df.iloc[:5, :10])
    
    # Calculate mean TF-IDF scores across all documents
    mean_tfidf = tfidf_df.mean(axis=0)
    top_terms = mean_tfidf.sort_values(ascending=False)[:20]
    
    # Visualization of top terms
    st.subheader("Top 20 Terms by TF-IDF Score")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(top_terms.index[::-1], top_terms.values[::-1], color='skyblue')
    
    # Add values to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=8)
    
    ax.set_xlabel('Mean TF-IDF Score')
    ax.set_title('Top 20 Most Important Terms (by Mean TF-IDF)')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Class-specific analysis (if class labels are available)
    if 'v1' in df_processed.columns:
        st.subheader("Class-Specific TF-IDF Analysis")
        
        # Get class labels
        class_labels = df_processed['v1'].unique()
        
        # Create TF-IDF analysis by class
        for label in class_labels:
            st.write(f"### Top Terms in '{label}' Messages")
            
            # Filter messages by class
            class_indices = df_processed[df_processed['v1'] == label].index
            
            # Get TF-IDF values for this class
            class_tfidf = tfidf_matrix[class_indices].toarray()
            
            # Calculate mean TF-IDF scores for this class
            class_mean_tfidf = pd.Series(class_tfidf.mean(axis=0), index=feature_names)
            class_top_terms = class_mean_tfidf.sort_values(ascending=False)[:10]
            
            # Display top terms
            st.write(pd.DataFrame({
                'Term': class_top_terms.index,
                'TF-IDF Score': class_top_terms.values.round(4)
            }))
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(class_top_terms.index, class_top_terms.values, color='lightgreen' if label == 'ham' else 'salmon')
            plt.xticks(rotation=45, ha='right')
            ax.set_ylabel('Mean TF-IDF Score')
            ax.set_title(f'Top 10 Terms in {label.upper()} Messages')
            plt.tight_layout()
            st.pyplot(fig)
    
    # TF-IDF Term Explorer
    st.subheader("TF-IDF Term Explorer")
    
    # Let user search for specific terms
    search_term = st.text_input("Search for a term in the TF-IDF vocabulary:")
    
    if search_term:
        # Find matching terms
        matching_terms = [term for term in feature_names if search_term.lower() in term.lower()]
        
        if matching_terms:
            st.write(f"Found {len(matching_terms)} matching terms:")
            
            # Get TF-IDF scores for matching terms
            term_scores = {term: mean_tfidf[term] for term in matching_terms}
            term_df = pd.DataFrame({
                'Term': term_scores.keys(),
                'TF-IDF Score': [round(score, 4) for score in term_scores.values()]
            }).sort_values('TF-IDF Score', ascending=False)
            
            st.dataframe(term_df)
        else:
            st.info(f"No terms containing '{search_term}' found in the vocabulary.")
    
    # Document similarity analysis
    st.subheader("Document Similarity Analysis")
    st.write("This section shows how TF-IDF can be used to find similar messages.")
    
    # Let user select a sample message
    sample_idx = st.selectbox(
        "Select a sample message:",
        options=list(range(min(5, len(df_processed)))),
        format_func=lambda x: f"Message {x+1}: {df_processed['v2'].iloc[x][:50]}..."
    )
    
    if sample_idx is not None:
        # Get the selected message
        sample_message = df_processed['v2'].iloc[sample_idx]
        sample_processed = df_processed['processed_text'].iloc[sample_idx]
        
        st.write("**Selected message:**")
        st.write(sample_message)
        
        st.write("**Processed version:**")
        st.write(sample_processed)
        
        # Get TF-IDF vector for the sample
        sample_vector = tfidf_matrix[sample_idx:sample_idx+1]
        
        # Calculate similarities to all other messages
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(sample_vector, tfidf_matrix).flatten()
        
        # Remove self-similarity
        similarities[sample_idx] = 0
        
        # Get top similar messages
        most_similar_indices = similarities.argsort()[-5:][::-1]
        
        st.write("**Most similar messages:**")
        for i, idx in enumerate(most_similar_indices):
            if similarities[idx] > 0:  # Only show if somewhat similar
                st.write(f"{i+1}. Similarity: {similarities[idx]:.4f}")
                st.write(f"   Message: {df_processed['v2'].iloc[idx]}")
                st.write("   ---")

# Model Analysis Section
if 'show_models' in st.session_state and st.session_state['show_models'] and 'preprocessed_df' in st.session_state:
    st.markdown("---")
    st.title("Model Building and Evaluation")
    
    # Get the preprocessed data
    df_processed = st.session_state['preprocessed_df']
    
    # Check if 'v1' column exists for class labels
    if 'v1' not in df_processed.columns:
        st.error("Cannot build models: No class labels (v1 column) found in the dataset")
    else:
        # Import necessary libraries
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
        from sklearn.pipeline import Pipeline
        import numpy as np
        import seaborn as sns
        from sklearn.model_selection import learning_curve
        
        # Create tabs for different sections of model building
        model_tabs = st.tabs(["Data Split", "Model Training", "Model Evaluation", "Hyperparameter Tuning", "Cross-Validation"])
        
        with model_tabs[0]:
            st.header("Data Splitting")
            st.write("""
            Before training models, we need to split our data into training and testing sets.
            The training set is used to train the model, while the testing set is used to evaluate its performance.
            """)
            
            # Let user choose test size
            test_size = st.slider("Test Set Size (percentage of data):", 10, 40, 20)
            test_size_fraction = test_size / 100.0
            
            # Let user choose random seed
            random_seed = st.number_input("Random Seed for Reproducibility:", 0, 1000, 42)
            
            # Create a button to perform the split
            if st.button("Split Data", key="split_data_btn"):
                with st.spinner("Splitting data into training and testing sets..."):
                    # Prepare features and target
                    X = df_processed['processed_text']
                    y = (df_processed['v1'] == 'spam').astype(int)  # Convert to binary (1 for spam, 0 for ham)
                    
                    # Perform the split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_fraction, random_state=random_seed, stratify=y
                    )
                    
                    # Save the split data in session state
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    
                    # Show split information
                    st.success(f"Data successfully split into training and testing sets!")
                    
                    # Create a summary table
                    split_data = {
                        'Dataset': ['Training Set', 'Testing Set', 'Total'],
                        'Number of Samples': [len(X_train), len(X_test), len(X)],
                        'Percentage': [
                            f"{len(X_train)/len(X):.1%}",
                            f"{len(X_test)/len(X):.1%}",
                            "100.0%"
                        ],
                        'Spam Count': [
                            sum(y_train),
                            sum(y_test),
                            sum(y)
                        ],
                        'Ham Count': [
                            len(y_train) - sum(y_train),
                            len(y_test) - sum(y_test),
                            len(y) - sum(y)
                        ],
                        'Spam Percentage': [
                            f"{sum(y_train)/len(y_train):.1%}",
                            f"{sum(y_test)/len(y_test):.1%}",
                            f"{sum(y)/len(y):.1%}"
                        ]
                    }
                    
                    split_df = pd.DataFrame(split_data)
                    st.table(split_df)
                    
                    # Visualize the split
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Training set distribution
                    train_counts = [len(y_train) - sum(y_train), sum(y_train)]
                    ax1.pie(train_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
                    ax1.set_title('Training Set Distribution')
                    
                    # Testing set distribution
                    test_counts = [len(y_test) - sum(y_test), sum(y_test)]
                    ax2.pie(test_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
                    ax2.set_title('Testing Set Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with model_tabs[1]:
            st.header("Model Training")
            
            # Check if data has been split
            if 'X_train' not in st.session_state:
                st.warning("Please split the data in the 'Data Split' tab first.")
            else:
                st.write("""
                Now we'll train several machine learning models on our preprocessed data.
                Different algorithms have different strengths and weaknesses for text classification tasks.
                """)
                
                # Let user select which models to train
                st.write("### Select Models to Train")
                
                col1, col2 = st.columns(2)
                with col1:
                    train_nb = st.checkbox("Naive Bayes", value=True)
                    train_lr = st.checkbox("Logistic Regression", value=True)
                
                with col2:
                    train_svm = st.checkbox("Support Vector Machine", value=True)
                    train_rf = st.checkbox("Random Forest", value=True)
                
                # Vectorization options
                st.write("### Text Vectorization Options")
                max_features = st.slider("Maximum number of features (vocabulary size):", 1000, 10000, 3000)
                
                # Button to train models
                train_btn = st.button("Train Models", key="train_models_btn")
                
                if train_btn:
                    with st.spinner("Training models, please wait..."):
                        # Get the training data from session state
                        X_train = st.session_state['X_train']
                        y_train = st.session_state['y_train']
                        X_test = st.session_state['X_test']
                        y_test = st.session_state['y_test']
                        
                        # Create a TF-IDF vectorizer
                        vectorizer = TfidfVectorizer(max_features=max_features)
                        
                        # Fit and transform the training data
                        X_train_tfidf = vectorizer.fit_transform(X_train)
                        X_test_tfidf = vectorizer.transform(X_test)
                        
                        # Save vectorizer and transformed data to session state
                        st.session_state['vectorizer'] = vectorizer
                        st.session_state['X_train_tfidf'] = X_train_tfidf
                        st.session_state['X_test_tfidf'] = X_test_tfidf
                        
                        # Initialize dictionary to store models and results
                        models = {}
                        results = {}
                        
                        # Progress bar for model training
                        total_models = sum([train_nb, train_lr, train_svm, train_rf])
                        progress_bar = st.progress(0)
                        current_model = 0
                        
                        # Train Naive Bayes if selected
                        if train_nb:
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                            
                            model_name = "Naive Bayes"
                            st.write(f"Training {model_name}...")
                            
                            nb = MultinomialNB()
                            nb.fit(X_train_tfidf, y_train)
                            
                            # Make predictions
                            y_pred = nb.predict(X_test_tfidf)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                            
                            # Store model and results
                            models['Naive Bayes'] = nb
                            results['Naive Bayes'] = {
                                'y_pred': y_pred,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                        
                        # Train Logistic Regression if selected
                        if train_lr:
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                            
                            model_name = "Logistic Regression"
                            st.write(f"Training {model_name}...")
                            
                            lr = LogisticRegression(max_iter=1000, random_state=random_seed)
                            lr.fit(X_train_tfidf, y_train)
                            
                            # Make predictions
                            y_pred = lr.predict(X_test_tfidf)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                            
                            # Store model and results
                            models['Logistic Regression'] = lr
                            results['Logistic Regression'] = {
                                'y_pred': y_pred,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                        
                        # Train SVM if selected
                        if train_svm:
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                            
                            model_name = "Support Vector Machine"
                            st.write(f"Training {model_name}...")
                            
                            svm = LinearSVC(random_state=random_seed)
                            svm.fit(X_train_tfidf, y_train)
                            
                            # Make predictions
                            y_pred = svm.predict(X_test_tfidf)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                            
                            # Store model and results
                            models['SVM'] = svm
                            results['SVM'] = {
                                'y_pred': y_pred,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                        
                        # Train Random Forest if selected
                        if train_rf:
                            current_model += 1
                            progress_bar.progress(current_model / total_models)
                            
                            model_name = "Random Forest"
                            st.write(f"Training {model_name}...")
                            
                            rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
                            rf.fit(X_train_tfidf, y_train)
                            
                            # Make predictions
                            y_pred = rf.predict(X_test_tfidf)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                            
                            # Store model and results
                            models['Random Forest'] = rf
                            results['Random Forest'] = {
                                'y_pred': y_pred,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        
                        # Save models and results to session state
                        st.session_state['models'] = models
                        st.session_state['results'] = results
                        
                        # Display success message
                        st.success(f"Successfully trained {len(models)} models!")
                
                # If models have been trained, show a summary table
                if 'models' in st.session_state and st.session_state['models']:
                    st.write("### Model Training Results")
                    
                    # Create a DataFrame with model performance metrics
                    metrics_df = pd.DataFrame({
                        'Model': [],
                        'Accuracy': [],
                        'Precision': [],
                        'Recall': [],
                        'F1 Score': []
                    })
                    
                    for model_name, result in st.session_state['results'].items():
                        new_row = pd.DataFrame({
                            'Model': [model_name],
                            'Accuracy': [result['accuracy']],
                            'Precision': [result['precision']],
                            'Recall': [result['recall']],
                            'F1 Score': [result['f1']]
                        })
                        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
                    
                    # Display metrics table with formatting
                    st.dataframe(metrics_df.style.format({
                        'Accuracy': '{:.2%}',
                        'Precision': '{:.2%}',
                        'Recall': '{:.2%}',
                        'F1 Score': '{:.2%}'
                    }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']))
                    
                    # Create bar chart comparing model performance
                    st.write("### Model Performance Comparison")
                    
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Prepare data for plotting
                        models = metrics_df['Model']
                        x = np.arange(len(models))
                        width = 0.2
                        
                        # Plot bars for each metric
                        ax.bar(x - width*1.5, metrics_df['Accuracy'], width, label='Accuracy', color='#4285F4')
                        ax.bar(x - width/2, metrics_df['Precision'], width, label='Precision', color='#EA4335')
                        ax.bar(x + width/2, metrics_df['Recall'], width, label='Recall', color='#FBBC05')
                        ax.bar(x + width*1.5, metrics_df['F1 Score'], width, label='F1 Score', color='#34A853')
                        
                        # Add labels and legend
                        ax.set_xlabel('Model')
                        ax.set_ylabel('Score')
                        ax.set_title('Model Performance Metrics')
                        ax.set_xticks(x)
                        ax.set_xticklabels(models)
                        ax.legend()
                        ax.set_ylim(0, 1.1)
                        
                        # Format y-axis as percentage - use a safer method
                        from matplotlib.ticker import PercentFormatter
                        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                        
                        # Add value labels on bars
                        for i, model in enumerate(models):
                            ax.text(i - width*1.5, metrics_df['Accuracy'].iloc[i] + 0.02, f"{metrics_df['Accuracy'].iloc[i]:.2%}", ha='center', va='bottom', fontsize=8)
                            ax.text(i - width/2, metrics_df['Precision'].iloc[i] + 0.02, f"{metrics_df['Precision'].iloc[i]:.2%}", ha='center', va='bottom', fontsize=8)
                            ax.text(i + width/2, metrics_df['Recall'].iloc[i] + 0.02, f"{metrics_df['Recall'].iloc[i]:.2%}", ha='center', va='bottom', fontsize=8)
                            ax.text(i + width*1.5, metrics_df['F1 Score'].iloc[i] + 0.02, f"{metrics_df['F1 Score'].iloc[i]:.2%}", ha='center', va='bottom', fontsize=8)
                        
                        # Safer approach to tight_layout
                        try:
                            fig.tight_layout()
                        except:
                            pass
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating performance comparison chart: {str(e)}")
                        # Display the raw data as fallback
                        st.write("Performance metrics:")
                        st.dataframe(metrics_df)
                    
                    # Identify the best model based on F1 score
                    best_model_idx = metrics_df['F1 Score'].idxmax()
                    best_model_name = metrics_df.loc[best_model_idx, 'Model']
                    
                    st.write(f"### Best Performing Model: {best_model_name}")
                    st.write(f"The {best_model_name} model achieved the highest F1 score of {metrics_df.loc[best_model_idx, 'F1 Score']:.2%}.")
                    
                    # Save best model info to session state
                    st.session_state['best_model_name'] = best_model_name
        
        with model_tabs[2]:
            st.header("Model Evaluation")
            
            # Check if models have been trained
            if 'models' not in st.session_state or not st.session_state['models']:
                st.warning("Please train models in the 'Model Training' tab first.")
            else:
                st.write("""
                Let's evaluate the models in more detail, looking at confusion matrices, classification reports,
                and for some models, feature importance.
                """)
                
                # Let user select which model to evaluate
                model_names = list(st.session_state['models'].keys())
                selected_model = st.selectbox("Select a model to evaluate:", model_names)
                
                if selected_model:
                    # Get the selected model and its results
                    model = st.session_state['models'][selected_model]
                    results = st.session_state['results'][selected_model]
                    
                    # Get the test data and predictions
                    X_test = st.session_state['X_test']
                    y_test = st.session_state['y_test']
                    y_pred = results['y_pred']
                    
                    # Confusion Matrix
                    st.write("### Confusion Matrix")
                    
                    try:
                        cm = confusion_matrix(y_test, y_pred)
                        fig = plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                  xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.title(f'Confusion Matrix - {selected_model}')
                        # Safer approach to tight_layout
                        try:
                            fig.tight_layout()
                        except:
                            pass
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating confusion matrix: {str(e)}")
                        # Display the raw confusion matrix as fallback
                        st.write("Raw confusion matrix:")
                        st.write(cm)
                    
                    # Calculate performance metrics
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Display metrics with explanation
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.write("### Performance Metrics")
                        st.write(f"**Accuracy**: {results['accuracy']:.2%}")
                        st.write(f"**Precision**: {results['precision']:.2%}")
                        st.write(f"**Recall**: {results['recall']:.2%}")
                        st.write(f"**F1 Score**: {results['f1']:.2%}")
                        
                        # Additional metrics
                        specificity = tn / (tn + fp)
                        st.write(f"**Specificity**: {specificity:.2%}")
                        
                        false_positive_rate = fp / (fp + tn)
                        st.write(f"**False Positive Rate**: {false_positive_rate:.2%}")
                        
                        false_negative_rate = fn / (fn + tp)
                        st.write(f"**False Negative Rate**: {false_negative_rate:.2%}")
                    
                    with metrics_col2:
                        st.write("### Metrics Explanation")
                        st.write("**Accuracy**: Proportion of correct predictions (both ham and spam)")
                        st.write("**Precision**: Proportion of spam predictions that are actually spam")
                        st.write("**Recall**: Proportion of actual spam messages that are correctly identified")
                        st.write("**F1 Score**: Harmonic mean of precision and recall")
                        st.write("**Specificity**: Proportion of actual ham messages correctly identified")
                        st.write("**False Positive Rate**: Proportion of ham messages incorrectly classified as spam")
                        st.write("**False Negative Rate**: Proportion of spam messages incorrectly classified as ham")
                    
                    # Classification Report
                    st.write("### Classification Report")
                    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format({
                        'precision': '{:.2f}',
                        'recall': '{:.2f}',
                        'f1-score': '{:.2f}',
                        'support': '{:.0f}'
                    }))
                    
                    # Feature Importance (for applicable models)
                    if selected_model in ['Logistic Regression', 'SVM', 'Random Forest']:
                        st.write(f"### Feature Importance for {selected_model}")
                        
                        vectorizer = st.session_state['vectorizer']
                        feature_names = vectorizer.get_feature_names_out()
                        
                        if selected_model == 'Logistic Regression':
                            feature_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model.coef_[0]
                            })
                            feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
                            
                        elif selected_model == 'SVM':
                            feature_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model.coef_[0]
                            })
                            feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
                            
                        elif selected_model == 'Random Forest':
                            feature_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model.feature_importances_
                            })
                            feature_importance['Abs_Importance'] = feature_importance['Importance']
                        
                        # Sort by absolute importance
                        feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
                        
                        # Display top features
                        st.write("#### Top 20 Most Important Features")
                        
                        top_features = feature_importance.head(20)
                        st.dataframe(top_features[['Feature', 'Importance']].reset_index(drop=True))
                        
                        # Visualization of feature importance
                        plt.figure(figsize=(10, 8))
                        plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
                        plt.xlabel('Importance')
                        plt.title(f'Top 20 Features by Importance - {selected_model}')
                        plt.tight_layout()
                        st.pyplot(plt)
                    
                    # ROC Curve (for models that support predict_proba)
                    if selected_model in ['Naive Bayes', 'Logistic Regression', 'Random Forest']:
                        st.write("### ROC Curve")
                        
                        # Get probabilities
                        X_test_tfidf = st.session_state['X_test_tfidf']
                        y_proba = model.predict_proba(X_test_tfidf)[:, 1]
                        
                        # Calculate ROC curve
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        # Plot ROC curve
                        fig = plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'Receiver Operating Characteristic - {selected_model}')
                        plt.legend(loc="lower right")
                        # Safer approach to tight_layout
                        try:
                            fig.tight_layout()
                        except:
                            pass
                        st.pyplot(fig)
                        
                        st.write(f"**Area Under the ROC Curve (AUC)**: {roc_auc:.4f}")
                        st.write("""
                        The ROC curve plots the true positive rate against the false positive rate at various 
                        threshold settings. The AUC (Area Under the Curve) is a measure of the model's ability 
                        to distinguish between classes. A higher AUC indicates better performance.
                        """)
        
        with model_tabs[3]:
            st.header("Hyperparameter Tuning")
            
            # Check if data has been split
            if 'X_train' not in st.session_state:
                st.warning("Please split the data in the 'Data Split' tab first.")
            else:
                st.write("""
                Hyperparameter tuning is the process of finding the optimal hyperparameters for a model
                to improve its performance. We'll use Grid Search with cross-validation to find the best parameters.
                """)
                
                # Let user select which model to tune
                model_options = {
                    'Naive Bayes': MultinomialNB(),
                    'Logistic Regression': LogisticRegression(random_state=42),
                    'SVM': LinearSVC(random_state=42),
                    'Random Forest': RandomForestClassifier(random_state=42)
                }
                
                selected_model_type = st.selectbox("Select a model type to tune:", list(model_options.keys()))
                
                if selected_model_type:
                    # Define parameter grids for each model
                    param_grids = {
                        'Naive Bayes': {
                            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
                        },
                        'Logistic Regression': {
                            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                            'penalty': ['l1', 'l2'],
                            'solver': ['liblinear']
                        },
                        'SVM': {
                            'C': [0.01, 0.1, 1.0, 10.0],
                            'loss': ['hinge', 'squared_hinge']
                        },
                        'Random Forest': {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20, 30],
                            'min_samples_split': [2, 5, 10]
                        }
                    }
                    
                    # Show the parameter grid for the selected model
                    st.write(f"### Parameter Grid for {selected_model_type}")
                    
                    param_grid = param_grids[selected_model_type]
                    param_df = pd.DataFrame()
                    
                    for param, values in param_grid.items():
                        param_df[param] = [str(values)]
                    
                    st.table(param_df)
                    
                    # Number of cross-validation folds
                    cv_folds = st.slider("Number of cross-validation folds:", 3, 10, 5)
                    
                    # Button to run hyperparameter tuning
                    tune_btn = st.button("Run Hyperparameter Tuning", key="tune_model_btn")
                    
                    if tune_btn:
                        with st.spinner(f"Tuning hyperparameters for {selected_model_type}. This may take a while..."):
                            # Get the training data
                            X_train = st.session_state['X_train']
                            y_train = st.session_state['y_train']
                            
                            # Create a pipeline with vectorization and model
                            pipeline = Pipeline([
                                ('vectorizer', TfidfVectorizer(max_features=3000)),
                                ('classifier', model_options[selected_model_type])
                            ])
                            
                            # Create parameter grid for the pipeline
                            pipeline_param_grid = {}
                            for param, values in param_grid.items():
                                pipeline_param_grid[f'classifier__{param}'] = values
                            
                            # Create and run grid search
                            grid_search = GridSearchCV(
                                pipeline, 
                                pipeline_param_grid, 
                                cv=cv_folds, 
                                scoring='f1',
                                n_jobs=-1,
                                verbose=1
                            )
                            
                            grid_search.fit(X_train, y_train)
                            
                            # Save results to session state
                            st.session_state['grid_search_results'] = {
                                'model_type': selected_model_type,
                                'best_params': grid_search.best_params_,
                                'best_score': grid_search.best_score_,
                                'cv_results': grid_search.cv_results_
                            }
                            
                            # Display best parameters and score
                            st.success("Hyperparameter tuning completed!")
                            
                            st.write("### Best Parameters:")
                            best_params = {}
                            for param, value in grid_search.best_params_.items():
                                # Extract parameter name without the 'classifier__' prefix
                                param_name = param.split('__')[1]
                                best_params[param_name] = value
                            
                            st.json(best_params)
                            
                            st.write(f"### Best F1 Score: {grid_search.best_score_:.2%}")
                            
                            # Visualize grid search results
                            st.write("### Grid Search Results:")
                            
                            # Extract and organize results
                            results = pd.DataFrame(grid_search.cv_results_)
                            
                            # Create a clear results table
                            results_table = pd.DataFrame({
                                'Parameters': results['params'],
                                'Mean F1 Score': results['mean_test_score'],
                                'Std Dev': results['std_test_score']
                            })
                            
                            # Sort by performance
                            results_table = results_table.sort_values('Mean F1 Score', ascending=False).reset_index(drop=True)
                            
                            # Format the parameters for display
                            results_table['Parameters'] = results_table['Parameters'].apply(
                                lambda x: ', '.join([f"{k.split('__')[1]}: {v}" for k, v in x.items()])
                            )
                            
                            # Display the top 10 parameter combinations
                            st.dataframe(results_table.head(10).style.format({
                                'Mean F1 Score': '{:.2%}',
                                'Std Dev': '{:.2%}'
                            }))
                            
                            # Visualization of top parameter combinations
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                top_results = results_table.head(10).copy()
                                
                                # Plot the results
                                bars = ax.barh(
                                    range(len(top_results)), 
                                    top_results['Mean F1 Score'], 
                                    xerr=top_results['Std Dev'],
                                    color='skyblue',
                                    alpha=0.8
                                )
                                
                                # Add labels
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax.text(
                                        width + 0.01, 
                                        bar.get_y() + bar.get_height()/2, 
                                        f"{top_results['Mean F1 Score'].iloc[i]:.2%}", 
                                        va='center'
                                    )
                                
                                # Customize the plot
                                ax.set_yticks(range(len(top_results)))
                                ax.set_yticklabels([f"Params {i+1}" for i in range(len(top_results))])
                                ax.set_xlabel('F1 Score')
                                ax.set_title(f'Top 10 Parameter Combinations for {selected_model_type}')
                                ax.set_xlim(0, 1.0)
                                ax.grid(axis='x', linestyle='--', alpha=0.7)
                                
                                # Use a try/except block to handle the tight_layout error
                                try:
                                    fig.tight_layout()
                                except:
                                    st.warning("Warning: Could not apply tight layout to the plot.")
                                    
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error plotting parameter combinations: {str(e)}")
                                st.write("Unable to visualize parameter combinations.")
                            
                            # Option to train the model with the best parameters
                            st.write("### Train Model with Best Parameters")
                            
                            if st.button("Train Model with Best Parameters", key="train_best_params_btn"):
                                with st.spinner(f"Training {selected_model_type} with the best parameters..."):
                                    # Create a new model with best parameters
                                    best_model = grid_search.best_estimator_
                                    
                                    # Make predictions on the test set
                                    X_test = st.session_state['X_test']
                                    y_test = st.session_state['y_test']
                                    
                                    y_pred = best_model.predict(X_test)
                                    
                                    # Calculate metrics
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                                    
                                    # Save the tuned model
                                    st.session_state['tuned_model'] = {
                                        'model_type': selected_model_type,
                                        'model': best_model,
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1': f1,
                                        'y_pred': y_pred
                                    }
                                    
                                    # Display performance
                                    st.success(f"Trained {selected_model_type} with the best parameters!")
                                    
                                    metrics_col1, metrics_col2 = st.columns(2)
                                    
                                    with metrics_col1:
                                        st.write("#### Performance Metrics")
                                        st.write(f"Accuracy: {accuracy:.2%}")
                                        st.write(f"Precision: {precision:.2%}")
                                        st.write(f"Recall: {recall:.2%}")
                                        st.write(f"F1 Score: {f1:.2%}")
                                    
                                    with metrics_col2:
                                        # Confusion Matrix
                                        cm = confusion_matrix(y_test, y_pred)
                                        plt.figure(figsize=(6, 4))
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                                  xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
                                        plt.xlabel('Predicted')
                                        plt.ylabel('Actual')
                                        plt.title('Confusion Matrix')
                                        st.pyplot(plt)
        
        with model_tabs[4]:
            st.header("Cross-Validation")
            
            # Check if data has been split
            if 'X_train' not in st.session_state:
                st.warning("Please split the data in the 'Data Split' tab first.")
            else:
                st.write("""
                Cross-validation is a technique for evaluating how well a model performs on unseen data.
                We'll use k-fold cross-validation to estimate the model's performance more robustly.
                """)
                
                # Let user select which model to evaluate with cross-validation
                model_options = {
                    'Naive Bayes': MultinomialNB(),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'SVM': LinearSVC(random_state=42),
                    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
                }
                
                selected_model_type = st.selectbox("Select a model type for cross-validation:", 
                                              list(model_options.keys()), key="cv_model_select")
                
                # Number of cross-validation folds
                cv_folds = st.slider("Number of cross-validation folds:", 3, 10, 5, key="cv_folds_slider")
                
                # Button to run cross-validation
                cv_btn = st.button("Run Cross-Validation", key="run_cv_btn")
                
                if cv_btn:
                    with st.spinner(f"Running {cv_folds}-fold cross-validation for {selected_model_type}..."):
                        # Get the training data
                        X = st.session_state['preprocessed_df']['processed_text']
                        y = (st.session_state['preprocessed_df']['v1'] == 'spam').astype(int)
                        
                        # Create a pipeline with vectorization and model
                        pipeline = Pipeline([
                            ('vectorizer', TfidfVectorizer(max_features=3000)),
                            ('classifier', model_options[selected_model_type])
                        ])
                        
                        # Perform cross-validation
                        scoring = {'accuracy': 'accuracy', 'precision': 'precision', 
                                 'recall': 'recall', 'f1': 'f1'}
                        
                        cv_results = {}
                        for scorer_name, scorer in scoring.items():
                            cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scorer)
                            cv_results[scorer_name] = cv_scores
                        
                        # Save results to session state
                        st.session_state['cv_results'] = {
                            'model_type': selected_model_type,
                            'results': cv_results
                        }
                        
                        # Display results
                        st.success("Cross-validation completed!")
                        
                        # Create a summary table
                        results_df = pd.DataFrame({
                            'Fold': list(range(1, cv_folds + 1)),
                            'Accuracy': cv_results['accuracy'],
                            'Precision': cv_results['precision'],
                            'Recall': cv_results['recall'],
                            'F1 Score': cv_results['f1']
                        })
                        
                        # Add mean and std dev
                        mean_row = pd.DataFrame({
                            'Fold': ['Mean'],
                            'Accuracy': [results_df['Accuracy'].mean()],
                            'Precision': [results_df['Precision'].mean()],
                            'Recall': [results_df['Recall'].mean()],
                            'F1 Score': [results_df['F1 Score'].mean()]
                        })
                        
                        std_row = pd.DataFrame({
                            'Fold': ['Std Dev'],
                            'Accuracy': [results_df['Accuracy'].std()],
                            'Precision': [results_df['Precision'].std()],
                            'Recall': [results_df['Recall'].std()],
                            'F1 Score': [results_df['F1 Score'].std()]
                        })
                        
                        results_df = pd.concat([results_df, mean_row, std_row], ignore_index=True)
                        
                        # Display the table
                        st.dataframe(results_df.style.format({
                            'Accuracy': '{:.2%}',
                            'Precision': '{:.2%}',
                            'Recall': '{:.2%}',
                            'F1 Score': '{:.2%}'
                        }))
                        
                        # # Visualization of cross-validation results
                        # st.write("### Cross-Validation Performance")
                        
                        # fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # # Extract fold data (without mean and std)
                        # fold_data = results_df.iloc[:-2]
                        
                        # # Plot the results
                        # metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                        # colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853']
                        
                        # for i, metric in enumerate(metrics):
                        #     ax.plot(fold_data['Fold'], fold_data[metric], 
                        #           marker='o', label=metric, color=colors[i])
                        
                        # # Add mean lines
                        # for i, metric in enumerate(metrics):
                        #     mean_val = results_df.loc[results_df['Fold'] == 'Mean', metric].values[0]
                        #     ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5)
                        #     ax.text(cv_folds + 0.1, mean_val, f"Mean: {mean_val:.2%}", va='center', color=colors[i])
                        
                        # # Customize the plot
                        # ax.set_xlabel('Fold')
                        # ax.set_ylabel('Score')
                        # ax.set_title(f'Cross-Validation Results for {selected_model_type}')
                        # ax.set_xticks(fold_data['Fold'])
                        # ax.set_ylim(0, 1.1)
                        # ax.legend()
                        # ax.grid(True, alpha=0.3)
                        
                        # # Format y-axis as percentage
                        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                        
                        # plt.tight_layout()
                        # st.pyplot(fig)
                        
                        # # Learning curves
                        # st.write("### Learning Curves")
                        # st.write("""
                        # Learning curves show how the model's performance changes as the training set size increases.
                        # This can help identify if the model would benefit from more data or is overfitting.
                        # """)
                        
                        # with st.spinner("Generating learning curves..."):
                        #     # Get the data
                        #     X = st.session_state['preprocessed_df']['processed_text']
                        #     y = (st.session_state['preprocessed_df']['v1'] == 'spam').astype(int)
                            
                        #     # Create a pipeline
                        #     pipeline = Pipeline([
                        #         ('vectorizer', TfidfVectorizer(max_features=3000)),
                        #         ('classifier', model_options[selected_model_type])
                        #     ])
                            
                        #     # Define the training sizes to evaluate
                        #     train_sizes = np.linspace(0.1, 1.0, 5)
                            
                        #     # Calculate learning curves
                        #     train_sizes, train_scores, test_scores = learning_curve(
                        #         pipeline, X, y, train_sizes=train_sizes, cv=cv_folds, scoring='f1', n_jobs=-1
                        #     )
                            
                        #     # Calculate means and standard deviations
                        #     train_mean = np.mean(train_scores, axis=1)
                        #     train_std = np.std(train_scores, axis=1)
                        #     test_mean = np.mean(test_scores, axis=1)
                        #     test_std = np.std(test_scores, axis=1)
                            
                        #     # Plot learning curves
                        #     try:
                        #         fig, ax = plt.subplots(figsize=(10, 6))
                                
                        #         # Convert train_sizes to percentages
                        #         train_sizes_pct = train_sizes * 100
                                
                        #         # Plot training scores
                        #         ax.plot(train_sizes_pct, train_mean, 'o-', color='blue', label='Training Score')
                        #         ax.fill_between(train_sizes_pct, train_mean - train_std, train_mean + train_std, 
                        #                        alpha=0.1, color='blue')
                                
                        #         # Plot test scores
                        #         ax.plot(train_sizes_pct, test_mean, 'o-', color='red', label='Cross-validation Score')
                        #         ax.fill_between(train_sizes_pct, test_mean - test_std, test_mean + test_std, 
                        #                       alpha=0.1, color='red')
                                
                        #         # Customize the plot
                        #         ax.set_xlabel('Percentage of Training Data')
                        #         ax.set_ylabel('F1 Score')
                        #         ax.set_title(f'Learning Curves for {selected_model_type}')
                        #         ax.legend(loc='best')
                        #         ax.grid(True, alpha=0.3)
                                
                        #         # Format y-axis as percentage - use a safer approach without lambda for compatibility
                        #         from matplotlib.ticker import PercentFormatter
                        #         ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                                
                        #         # Use a try/except block to handle the tight_layout error
                        #         try:
                        #             fig.tight_layout()
                        #         except:
                        #             st.warning("Warning: Could not apply tight layout to the plot.")
                                    
                        #         st.pyplot(fig)
                        #     except Exception as e:
                        #         st.error(f"Error plotting learning curves: {str(e)}")
                        #         st.write("Learning curve data:")
                        #         st.write({
                        #             "Train sizes": list(train_sizes_pct),
                        #             "Train mean scores": list(train_mean),
                        #             "Test mean scores": list(test_mean)
                        #         })
                            
                        #     # Interpretation
                        #     st.write("#### Learning Curve Interpretation")
                            
                        #     # Calculate the gap between train and test scores
                        #     final_train_score = train_mean[-1]
                        #     final_test_score = test_mean[-1]
                        #     score_gap = final_train_score - final_test_score
                            
                        #     if score_gap > 0.2:
                        #         st.write("""
                        #         **Overfitting detected**: There is a large gap between the training and validation scores, 
                        #         which suggests that the model is overfitting to the training data. Consider using regularization 
                        #         techniques or reducing model complexity.
                        #         """)
                        #     elif test_mean[-1] - test_mean[0] < 0.05:
                        #         st.write("""
                        #         **Underfitting detected**: The validation score doesn't improve much as training data increases, 
                        #         which suggests the model might be too simple to capture the patterns in the data. Consider using 
                        #         a more complex model or adding more features.
                        #         """)
                        #     elif test_mean[-1] - test_mean[-2] > 0.01:
                        #         st.write("""
                        #         **Could benefit from more data**: The validation score is still increasing with more training data, 
                        #         which suggests that adding more training examples could further improve model performance.
                        #         """)
                        #     else:
                        #         st.write("""
                        #         **Good fit**: The model shows a good balance between training and validation scores, and the 
                        #         validation score has reached a plateau. This suggests that the model is neither overfitting nor 
                        #         underfitting and has learned the patterns in the data well.
                        #         """)
                            
                            # # Performance metrics explanation
                            # train_test_ratio = final_train_score / final_test_score if final_test_score > 0 else float('inf')
                            
                            # st.write(f"""
                            # - Final training score: {final_train_score:.2%}
                            # - Final validation score: {final_test_score:.2%}
                            # - Training/validation ratio: {train_test_ratio:.2f} (closer to 1.0 is better)
                            # """)
                            
                            # if train_test_ratio > 1.2:
                            #     st.write("""
                            #     The ratio of training to validation score is high, which indicates potential overfitting. 
                            #     Ideally, this ratio should be close to 1.0.
                            #     """)
                                
                # # If we've run cross-validation, display a download button for the results
                # if 'cv_results' in st.session_state:
                #     # Create a CSV with the cross-validation results
                #     cv_results = st.session_state['cv_results']
                #     model_type = cv_results['model_type']
                #     results = cv_results['results']
                    
                #     # Create a DataFrame with the results
                #     cv_df = pd.DataFrame({
                #         'Fold': list(range(1, len(results['accuracy']) + 1)),
                #         'Accuracy': results['accuracy'],
                #         'Precision': results['precision'],
                #         'Recall': results['recall'],
                #         'F1 Score': results['f1']
                #     })
                    
                #     # Add CSV download button
                #     csv = cv_df.to_csv(index=False).encode('utf-8')
                #     st.download_button(
                #         label=f"Download {model_type} Cross-Validation Results",
                #         data=csv,
                #         file_name=f"{model_type.lower().replace(' ', '_')}_cv_results.csv",
                #         mime="text/csv",
                #     )
                    
# The rest of the original code for model evaluation would follow here

# Custom Message Classifier Section
if 'models' in st.session_state and st.session_state['models']:
    st.markdown("---")
    st.title("🔍 Custom Message Classifier")
    
    st.write("""
    Enter any message below to classify it as spam or legitimate (ham) using your trained model.
    The classifier will preprocess the text using the same steps as your dataset and then make a prediction.
    """)
    
    # Let user select which model to use for prediction
    model_names = list(st.session_state['models'].keys())
    
    # Default to the best model if available
    default_model = st.session_state.get('best_model_name', model_names[0]) if model_names else None
    
    if default_model:
        selected_model_name = st.selectbox(
            "Select model for prediction:", 
            model_names,
            index=model_names.index(default_model)
        )
        
        # Get the selected model
        selected_model = st.session_state['models'][selected_model_name]
        
        # Text area for user input
        user_message = st.text_area(
            "Enter a message to classify:", 
            height=100,
            placeholder="Type or paste a message here to classify it as spam or ham..."
        )
        
        # Add a "Classify" button
        classify_btn = st.button("Classify Message", key="classify_btn", use_container_width=True)
        
        if classify_btn and user_message:
            with st.spinner("Classifying message..."):
                # Get the preprocessing functions
                # We need to redefine them here to avoid scope issues
                def lowercase_text(text):
                    return text.lower() if isinstance(text, str) else ""
                
                def remove_punctuation(text):
                    if isinstance(text, str):
                        return text.translate(str.maketrans('', '', string.punctuation))
                    return ""
                
                def remove_stopwords(text):
                    if isinstance(text, str):
                        tokens = text.split()
                        stop_words = set(stopwords.words('english'))
                        filtered_tokens = [word for word in tokens if word not in stop_words]
                        return ' '.join(filtered_tokens)
                    return ""
                
                def remove_numbers(text):
                    if isinstance(text, str):
                        return ''.join([i for i in text if not i.isdigit()])
                    return ""
                
                def apply_stemming(text):
                    if isinstance(text, str):
                        from nltk.stem import PorterStemmer
                        stemmer = PorterStemmer()
                        tokens = text.split()
                        stemmed_tokens = [stemmer.stem(word) for word in tokens]
                        return ' '.join(stemmed_tokens)
                    return ""
                
                # Preprocess the user message - apply the same steps used during training
                processed_message = user_message
                processed_message = lowercase_text(processed_message)
                processed_message = remove_punctuation(processed_message)
                processed_message = remove_stopwords(processed_message)
                processed_message = remove_numbers(processed_message)
                processed_message = apply_stemming(processed_message)
                
                # Vectorize the message
                vectorizer = st.session_state['vectorizer']
                message_tfidf = vectorizer.transform([processed_message])
                
                # Make prediction
                prediction = selected_model.predict(message_tfidf)[0]
                
                # Get prediction probability if available
                prediction_proba = None
                if hasattr(selected_model, 'predict_proba'):
                    probabilities = selected_model.predict_proba(message_tfidf)[0]
                    prediction_proba = probabilities[1]  # Probability of being spam (class 1)
                
                # Display result with an attractive UI
                st.markdown("### Classification Result")
                
                # Create columns for displaying the result
                col1, col2 = st.columns([1, 3])
                
                # Display icon based on prediction
                with col1:
                    if prediction == 1:  # Spam
                        st.markdown("""
                        <div style="background-color: #FFEBEE; padding: 20px; border-radius: 10px; 
                                   text-align: center; font-size: 40px; color: #D32F2F;">
                        ⚠️
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Ham
                        st.markdown("""
                        <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; 
                                   text-align: center; font-size: 40px; color: #388E3C;">
                        ✅
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display prediction and details
                with col2:
                    if prediction == 1:  # Spam
                        st.markdown("""
                        <div style="background-color: #FFEBEE; padding: 20px; border-radius: 10px;">
                        <h3 style="color: #D32F2F; margin-top: 0;">SPAM DETECTED</h3>
                        <p>This message has been classified as spam.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Ham
                        st.markdown("""
                        <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px;">
                        <h3 style="color: #388E3C; margin-top: 0;">LEGITIMATE MESSAGE</h3>
                        <p>This message appears to be legitimate (ham).</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show confidence if probability is available
                if prediction_proba is not None:
                    st.markdown("### Confidence Level")
                    
                    confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
                    confidence_percentage = confidence * 100
                    
                    # Create a progress bar for the confidence
                    confidence_color = "#D32F2F" if prediction == 1 else "#388E3C"
                    st.markdown(f"""
                    <div style="margin-top: 10px;">
                        <div style="width: 100%; background-color: #f0f0f0; border-radius: 5px; height: 30px;">
                            <div style="width: {confidence_percentage}%; background-color: {confidence_color}; 
                                     height: 30px; border-radius: 5px; text-align: center; line-height: 30px; color: white;">
                                {confidence_percentage:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    The model is **{confidence_percentage:.1f}%** confident in its prediction
                    that this message is {"spam" if prediction == 1 else "legitimate"}.
                    """)
                
                # Display preprocessed message details
                with st.expander("View Preprocessing Details"):
                    st.markdown("#### Original Message")
                    st.markdown(f"```\n{user_message}\n```")
                    
                    st.markdown("#### Processed Message")
                    st.markdown(f"```\n{processed_message}\n```")
                    
                    st.markdown("#### Important Features")
                    
                    # Only extract feature importance for models that support it
                    if selected_model_name in ['Logistic Regression', 'SVM', 'Random Forest']:
                        # Get feature names
                        feature_names = vectorizer.get_feature_names_out()
                        
                        # Get feature importance based on model type
                        if selected_model_name in ['Logistic Regression', 'SVM']:
                            feature_importance = selected_model.coef_[0]
                        else:  # Random Forest
                            feature_importance = selected_model.feature_importances_
                        
                        # Get non-zero features from the vectorized message
                        message_features = message_tfidf.toarray()[0]
                        present_feature_indices = message_features.nonzero()[0]
                        
                        if len(present_feature_indices) > 0:
                            present_features = []
                            for idx in present_feature_indices:
                                feature_name = feature_names[idx]
                                importance = feature_importance[idx]
                                present_features.append((feature_name, importance))
                            
                            # Sort by absolute importance
                            present_features.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Display the top features
                            feature_df = pd.DataFrame(present_features[:10], columns=['Term', 'Importance'])
                            st.dataframe(feature_df)
                        else:
                            st.write("No features matched the vocabulary.")
                    else:
                        st.write("Feature importance not available for this model type.")
else:
    st.info("Please upload a dataset file to begin analysis.")