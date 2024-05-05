import streamlit as st
import pandas as pd
import datetime
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # For embeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from PIL import Image
import matplotlib.pyplot as plt


def main():
    image = Image.open("C:\\Users\\darve\\OneDrive\\Desktop\\agro.png")  # Replace with your image filename

    # Display the image with a caption (optional)
    st.image(image, use_column_width=False)
    st.title("!! AI- Agro Web Interface !!")
    choice = st.selectbox("Select an option:", ("View Weather Data", "Retrieve Advisories"))

    if choice == "View Weather Data":
        def filter_data(df, start_date, end_date):
            # Convert start_date to datetime object with the correct format
            start_date = pd.to_datetime(start_date, format='%d-%m-%Y')

            # If end_date is provided, convert it to datetime object
            if end_date:
                end_date = pd.to_datetime(end_date, format='%d-%m-%Y')
            else:
                # If end_date is not provided, set it as the same as start_date
                end_date = start_date

            # Convert the first column to datetime objects
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%d-%m-%Y')

            # Filter DataFrame based on date range
            filtered_df = df[(df.iloc[:, 0] >= start_date) & (df.iloc[:, 0] <= end_date)]

            return filtered_df

        def plot_data(data):
            # Select columns for plotting (replace with your desired columns)
            cols_to_plot = ["Rainfall(mm)", "Tmax(°C)", "Tmin(°C)"]

            # Convert "Parameter" column to datetime format
            data["Parameter"] = pd.to_datetime(data["Parameter"], format='%Y-%m-%d %H:%M:%S')

            # Extract day and month for DD/MM format
            data["Parameter_DDMM"] = data["Parameter"].dt.strftime('%d/%m')

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('grey')
            for col in cols_to_plot:
                ax.plot(data["Parameter_DDMM"], data[col])
            ax.set_xlabel("(DD/MM)", color='white')  # Update X-axis label
            ax.set_ylabel("Value", color='white')
            ax.set_title("visuals", color='white')
            ax.tick_params(color='grey')
            st.pyplot(fig)

        def update_start_date():
            st.session_state["start_date"] = st.session_state.get("start_date", None)  # Get existing value or None

        def update_end_date():
            st.session_state["end_date"] = st.session_state.get("end_date", None)
        # Path to your dataset file
        file_path = "C:\\Users\\darve\\OneDrive\\Desktop\\forecast.csv"

        # Read the dataset into a DataFrame with the first row as header
        df = pd.read_csv(file_path, encoding='ISO-8859-1', header=0)

        # State variables to hold user input (initialize as None)
        start_date = None
        end_date = None

        # Buttons and their actions
        submit_button = st.button("Submit", type="primary")
        reset_button = st.button("Reset", type="secondary")

        if submit_button:
            # Get user input for start date
            start_date = st.session_state.get("start_date", None)
            # Get user input for end date, or leave it empty if not provided
            end_date = st.session_state.get("end_date", None)

            # Filter the data only if both start and end dates are provided
            if start_date and end_date:
                filtered_data = filter_data(df.copy(), start_date, end_date)
                #st.write(filtered_data.to_string(index=False))
                st.table(filtered_data)
            if start_date and end_date:
                filtered_data = filter_data(df.copy(), start_date, end_date)
                # st.table(filtered_data)  # Display the table
                plot_data(filtered_data)

        # Update state variables on date input changes
        st.date_input("Enter Start Date:", key="start_date", format="DD/MM/YYYY", on_change=update_start_date)
        st.date_input("Enter End Date (Optional):", key="end_date", format="DD/MM/YYYY", on_change=update_end_date)

        # Helper functions to update state variables on date input changes


        # Reset functionality using reset button
        if reset_button:
            st.session_state = {}  # Clear all session state variables

    elif choice == "Retrieve Advisories":

        def calculate_week_number(year, month, day):
            try:
                # Create a datetime object
                date = datetime.datetime(year, month, day)
                # Calculate the week number using strftime
                week_number = date.strftime('%U')
                return int(week_number) + 1  # Add 1 to make weeks start from 1
            except ValueError:
                return "Invalid date. Please enter a valid date."

        def get_advisories(start_date, end_date, document_paths):
            """
            Retrieves advisories for a date range, handling multiple weeks if necessary.

            Args:
                start_date (datetime.date): The start date of the range (inclusive).
                end_date (datetime.date): The end date of the range (inclusive).
                document_paths (dict): A dictionary mapping week numbers (strings) to PDF paths.

            Returns:
                list: A list of strings, where each element represents the text for a week
                      within the date range.
            """

            advisories = []
            current_date = start_date
            selected_weeks = []
            while current_date <= end_date:
                week_number = calculate_week_number(current_date.year, current_date.month, current_date.day)
                selected_weeks.append(f"Week {week_number}")
                selected_week = "week " + str(week_number)
                if selected_week in document_paths:
                    pdf_path = document_paths[selected_week]
                    try:
                        # Read text from PDF
                        with open(pdf_path, 'rb') as pdf_file:
                            pdfreader = PdfReader(pdf_file)
                            raw_text = ''
                            for page in pdfreader.pages:
                                content = page.extract_text()
                                if content:
                                    raw_text += content
                        advisories.append(raw_text)
                    except FileNotFoundError:
                        advisories.append(f"PDF not found at: {pdf_path}")
                    except Exception as e:
                        advisories.append(f"An error occurred while reading the PDF: {e}")
                current_date += datetime.timedelta(days=7)  # Move to next week

            return advisories, selected_weeks
        def process_advisory_text(text):
            # You can customize this function to clean or preprocess the text
            # (e.g., remove headers/footers, apply NLP techniques)
            return text

        def answer_advisory_question(question, processed_text, openai_api_key):
            # Initialize OpenAI API (consider error handling for missing API key)
            llm = OpenAI(openai_api_key=openai_api_key)

            # Here, we're using RetrievalQA with basic text splitting
            # You can explore more advanced text-to-code or transformers for embeddings
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(processed_text)

            # Replace with your preferred embedding method (if needed)
            # embeddings = TransformerEmbeddings(model_name="allenai/juris-base")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # Initialize vector store
            docsearch = FAISS.from_texts(texts, embeddings)

            # Initialize question answering model
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

            # Answer the user's question
            return qa.invoke(question)

        # Set your OpenAI API key (consider environment variables for security)
        openai_api_key = os.getenv("OPENAI_API_KEY", "sk-proj-Zebvn843vJ64SCKfgB2fT3BlbkFJ4oZOzj7cWFh8PEcNEjOj")

        # Streamlit App Interface (continued)

        # Start and end date pickers
        start_date = st.date_input("Select Start Date:", value=None, format="YYYY/MM/DD")
        end_date = st.date_input("Select End Date:", value=None, format="YYYY/MM/DD")

        if start_date and end_date:  # Check if both dates are selected
            # Calculate week range based on start and end dates
            if start_date > end_date:
                st.error("Start date cannot be after end date.")
            else:
                # Replace with your actual document paths for each week
                document_paths = {
                    "week 1": "path/to/week1_advisories.pdf",
                    "week 2": "C:\\Users\\darve\\OneDrive\\Desktop\\week 2.pdf",
                    "week 3": "C:\\Users\\darve\\OneDrive\\Desktop\\week 3.pdf",
                    "week 4": "C:\\Users\\darve\\OneDrive\\Desktop\\week 4.pdf",
                    # ... add entries for other weeks
                }

                # Get advisories for all weeks in the date range
                advisory_texts, selected_weeks = get_advisories(start_date, end_date, document_paths)

                # Display selected week numbers
                st.write(f"Selected Weeks: {', '.join(selected_weeks)}")  # Join weeks with commas

                # Combine text from all retrieved advisories (if multiple weeks)
                processed_text = "\n\n".join(advisory_texts)  # Separate texts for clarity

                # Ask user's question about the advisory
                question = st.text_input("Ask your questions")

                if question:  # Check if a question is asked
                    answer = answer_advisory_question(question, processed_text, openai_api_key)

                    st.write("RECOMMENDED ADVICE : ")
                    st.write(answer["result"])


if __name__ == "__main__":
    main()
