import google.generativeai as genai
from crewai import Agent, Task, Crew
import os
import streamlit as st
from litellm.exceptions import RateLimitError
st.markdown("""
    **Note:** This Multi-Modal Agent can take multiple media files of any type. 
    For video processing, try uploading a lower-quality video for testing purposes, 
    as Google's API rate limit may exceed if the video is of high quality.
""")
page_bg_img="""
<style>
[data-testid="stAppViewContainer"]{
background-size: cover;
background-image: url("https://images.unsplash.com/photo-1517483000871-1dbf64a6e1c6?q=80&w=3869&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)
generation_config = {
    "temperature": 0.25,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 10000,
    "response_mime_type": "text/plain",
}

# Configure the Gemini API
API_KEY = "AIzaSyDW1ZDMrgdjgcXjICHVkVFZju2vOoFVOOk"
genai.configure(api_key=API_KEY)



# Define AI Agents
image_agent = Agent(
    name="ClaimVision AI",
    role="AI Image Forensics and Damage Verification Specialist",
    goal = """To detect fraudulent insurance claims by analyzing submitted images for inconsistencies, signs of tampering, and verifying the authenticity of damage based on image features, ensuring that genuine claims are processed accurately and fraudulent claims are flagged.""",
backstory = """ClaimGuard AI was developed to tackle fraud in the insurance sector by utilizing advanced computer vision and AI techniques to analyze submitted images of damaged items. It carefully evaluates factors such as the extent of damage and potential signs of image manipulation. ClaimGuard AI helps insurers differentiate between legitimate claims, where extensive damage is visible, and fraudulent claims, where damage is minimal or inconsistently presented. By integrating machine learning models trained on a wide range of genuine and fraudulent claims, ClaimGuard AI empowers insurers to make more informed decisions, reducing losses from fraudulent activities and ensuring fair claims processing for policyholders.""",
    llm="gemini-2.0-flash-thinking-exp-01-21",
    self_reflect=False
)

audio_agent = Agent(
    name="Speech Analysis AI",
    role="Analyzes audio files for inconsistencies and fraud in conjunction with other agents' data.",
    goal="Detect fraudulent indicators in speech recordings by analyzing vocal patterns, tone, inconsistencies in the conversation, and correlating with visual data for cross-verification of the claim's authenticity.",
    backstory="""Speech Analysis AI was developed to enhance fraud detection by focusing on vocal cues and speech patterns that often accompany deceptive behavior. By analyzing factors such as hesitation, tone shifts, and unnatural pauses, this AI provides critical insights to identify fraud in claims, interviews, and other scenarios. Integrating with other agents, such as image analysis, it cross-references the findings to provide a more comprehensive fraud assessment, ensuring that the evidence from both visual and audio sources is considered for more accurate detection of fraudulent activities.""",
    llm="gemini-2.0-flash-thinking-exp-01-21",
    self_reflect=False
)

document_agent = Agent(
    name="Document Analysis AI",
    role = "Analyzes documents for fraud indicators, including inconsistencies in text formatting, structure, and authenticity, while cross-referencing data with image and speech agents to identify contradictions and ensure a comprehensive fraud detection process.",
    goal="Detect fraudulent indicators in documents by analyzing inconsistencies in text, fonts, timestamps, structure, and cross-referencing the data with image and speech analysis results to identify any contradictions that may indicate fraudulent behavior.",
    backstory=""" Document Analysis AI was developed to ensure the integrity of submitted documents, such as receipts or contracts, by scrutinizing various aspects like text formatting, timestamps, and document structure. With the ability to identify signs of manipulation, it cross-references its findings with other agents, including image and speech analysis, to create a more accurate and holistic assessment of fraud.This AI-driven approach helps insurers and organizations detect fraudulent claims and document manipulation, providing them with the most reliable insights when validating claims or verifying authenticity.""",
    llm="gemini-2.0-flash-thinking-exp-01-21",
    self_reflect=False
) 

video_agent = Agent(
    name="Video Analysis AI",
    role="Analyze video footage for fraud detection and inconsistencies, and relate them with other agents' data (document, image, and speech) to check for any contradictory points that may indicate fraud.",
    goal="Detect fraudulent indicators in video footage by analyzing visual elements, object movement, lighting, and background elements, while comparing the results with data from other agents (document, image, and speech) to identify any contradictions. Adjust the confidence score accordingly based on the density of contradictions between the agents.",
    backstory="Video Analysis AI was developed to scrutinize video evidence for fraud by analyzing inconsistencies in movement, lighting, background elements, and damage patterns. It cross-references these findings with data from other agents (document, image, and speech) to provide a holistic view of the authenticity of the claim. This AI-driven approach enhances fraud detection accuracy and helps insurers make informed decisions about claim validity.",
    llm="gemini-2.0-flash-thinking-exp-01-21",
    self_reflect= False
)


# Define Tasks
def analyze_medical_report(image_path: str):
    description = """Analyze the submitted image of a damaged mobile phone for fraud detection. Identify anomalies, inconsistencies, and signs of tampering. Verify metadata, assess damage authenticity, detect digital alterations, and cross-check with fraud patterns. Provide AI-driven insights, recommendations for insurers, and protection strategies for policyholders."""
    model_instance = genai.GenerativeModel(image_agent.llm,generation_config=generation_config)
    input_data = [description, genai.upload_file(image_path)]
    response = model_instance.generate_content(input_data)
    return response.text

def analyze_audio(audio_path: str):
    description = """
Speech Analysis AI is an advanced AI agent designed to analyze audio files for inconsistencies and potential fraud indicators in speech. By leveraging cutting-edge natural language processing and audio forensics, this agent specializes in detecting subtle vocal patterns, tone shifts, and inconsistencies that may suggest deception or dishonesty. Whether it’s evaluating phone calls for insurance claims, interviews, or security assessments, Speech Analysis AI helps identify red flags by focusing on vocal cues such as hesitation, stress, unnatural pauses, and irregular speech tempo. Its goal is to enhance fraud detection capabilities and improve decision-making in situations where verbal communication is key.
"""
    model_instance = genai.GenerativeModel(audio_agent.llm)
    input_data = [description, genai.upload_file(audio_path)]
    response = model_instance.generate_content(input_data)
    return response.text

def analyze_document(document_path: str):
    description = """
Document Analysis AI is a highly specialized AI agent designed to verify the authenticity of documents submitted as part of claims. By leveraging advanced machine learning and computer vision techniques, it analyzes receipts, invoices, and other supporting documents for signs of tampering or forgery. The agent inspects factors like text font consistency, metadata, timestamp validation, and document layout to identify any alterations or irregularities. It plays a crucial role in fraud detection by ensuring that all documents associated with a claim are legitimate and have not been manipulated. This ensures the integrity of the claim submission process and helps prevent fraudulent activities in various sectors, including insurance, banking, and legal services.
"""
    model_instance = genai.GenerativeModel(document_agent.llm)
    input_data = [description, genai.upload_file(document_path)]
    response = model_instance.generate_content(input_data)
    return response.text 

def analyze_videomedical_report(video_path):
    description = """Video Analysis AI is an advanced AI agent focused on analyzing video submissions to detect fraudulent activities such as tampered footage, staged accidents, or manipulated evidence. The agent leverages cutting-edge computer vision and machine learning algorithms to inspect the video for inconsistencies, including frame alterations, unnatural damage patterns, lighting discrepancies, and the sequence of events. It also looks for any evidence of manipulation or falsification, ensuring the video is authentic and not altered to support a fraudulent claim. By providing a deep analysis of video content, this AI agent helps organizations and insurance companies validate the authenticity of submitted video evidence, reducing the risk of fraud and improving decision-making."""
    model_instance = genai.GenerativeModel(video_agent.llm)
    input_data = [description, genai.upload_file(video_path)]
    response = model_instance.generate_content(input_data)
    return response.text


def extract_frames(video_path, output_folder, frame_rate=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        frame_count += 1

    cap.release()
    return extracted_frames
def analyze_medical_video(video_path: str):
    description = "Analyze video evidence for insurance fraud detection. Identify and assess damages such as cracks, scratches, and missing parts. Provide a structured analysis."
    output_folder = "extracted_frames"
    frames = extract_frames(video_path, output_folder, frame_rate=10)

    results = []
    for frame in frames:
        result = analyze_videomedical_report(frame, description)
        results.append(result)

    return "\n".join(results)

# Define Crew and Execute Tasks
def multi_modal_analysis(image_path: str, audio_path: str, document_path: str, video_path: str):
    print("\n--- Running Multi-Modal Claim Analysis with CrewAI ---")
    
    task1 = Task(
        name="ClaimVision AI - Fraud Detection",
    agent=image_agent,
    description="""ClaimVision AI is an advanced AI agent designed to analyze images submitted for insurance claims, specifically focused on detecting fraudulent claims related to damaged mobile phones. It uses AI-driven image forensics, computer vision, and machine learning techniques to identify inconsistencies, anomalies, and signs of tampering in submitted images. By examining factors such as damage patterns, metadata, and historical fraud data, ClaimVision AI provides actionable insights to support fraud detection, enhance decision-making, and recommend appropriate actions to insurance companies. The agent helps improve the accuracy and efficiency of the claims process while reducing the risk of fraudulent payouts.""",

    expected_output = """Should be precise in 15 points plain text without any bold text and in paragraph. The output will be a detailed report highlighting:
    1. Identification of inconsistencies in the image, such as if the phone appears unbroken despite the claim, which suggests a potential fraud
    2. Verification of the damage severity in the image, with more cracks or breaks indicating a genuine claim, while minimal damage might suggest an attempt to exaggerate the issue
    3. Metadata analysis including timestamp, device model, and location to identify any inconsistencies with the reported claim
    4. Detection of any signs of image tampering, such as areas with unnatural damage patterns or suspicious edits to exaggerate the breakage
    5. A confidence score for the authenticity of the claim,(get the confidence score alwways on basis of visual appearance the bore breaks and cracks the less fraud mean less confidence vice versa) along with actionable insights on whether further proof or investigation is needed, considering the severity and authenticity of the damage in the image

    The report will help insurance companies make data-driven decisions, provide tailored recommendations for claim approval or denial, and suggest protective measures for claimants to avoid future fraud.""",
        func=analyze_medical_report,
        args=[image_path]
    )
    
    task2 = Task(
        name="Speech Analysis",
    agent=audio_agent,
    description="Analyze audio recordings for fraud indicators, inconsistencies, and speech patterns and relate them with other agents data and check for any contradictory points which proves fraud claim.",

        expected_output = """Should be precise in 15 points plain text without any bold text and in paragraph. The output will be a detailed report highlighting:

        1.Identification of hesitation, unnatural pauses, or inconsistencies in speech that may indicate deception.
        2.Analysis of tone and pitch for signs of stress, nervousness, or discomfort associated with fraud.
        3.Evaluation of speech tempo for irregularities such as speaking too fast or too slow, suggesting fabrication.
        4.Detection of inconsistencies in word choice, sentence structure, or vague responses, which may suggest evasion.
        5.Identification of vocal stress or shaky voice patterns, signaling potential dishonesty or discomfort.
        6.Assessment of clarity in speech to identify evasiveness or deliberate attempts to avoid clear answers.
        7.Analysis of the emotional tone to check for mismatches with the narrative being described in the claim.
        8.Detection of signs that the speaker is reading from a script, indicating a rehearsed or fabricated claim.
        9.Comparison of speech patterns with known fraud cases to identify recurring red flags or deceptive behavior.
        10.Identification of contradictions between the speech and visual data from the image agent, such as speech indicating stress while the image shows minimal damage.
        11.Adjustments to the fraud confidence score based on contradictory findings from speech and image analysis.
        12.If the image agent shows no fraud and the speech agent shows high fraud, a slight increase in the fraud confidence score may be warranted due to potential hidden deception.
        13.If both agents indicate high fraud, the confidence score should be increased, reinforcing the likelihood of fraudulent activity.
        14.If the image agent shows high fraud and the speech agent indicates low fraud, the confidence score will remain unchanged, relying on the stronger visual confirmation.
        15.A final fraud confidence score that synthesizes the results of both image and speech agents, with actionable recommendations for further investigation or claim approval.
    The report will help insurance companies make informed decisions, provide targeted recommendations for claim validation, and suggest strategies to improve fraud detection processes.""",
        func=analyze_audio,
        args=[audio_path]
    )
    
    task3 = Task(
        name="Document Analysis",
    agent=document_agent,
    description="Verify the authenticity of documents submitted for fraud detection, including receipts and other supporting files, and relate them with other agents' data (image and speech) to check for any contradictory points that may indicate fraud.",

    expected_output = """
Should be precise in 15 points plain text without any bold text and in paragraph. The output will be a detailed report highlighting:
1. Identification of discrepancies in text fonts, sizes, or styles that may suggest document manipulation.
2. Verification of timestamp metadata to ensure the document's creation time aligns with the claim submission.
3. Detection of alterations in document structure, such as missing sections or irregular formatting.
4. Analysis of document properties, including file creation and modification dates, for signs of tampering.
5. Cross-referencing receipt details with external data sources to verify purchase legitimacy.
6. Identification of any forged signatures or inconsistent handwriting patterns in scanned documents.
7. Assessment of document quality to identify potential scanning or digital editing inconsistencies.
8. Comparison of receipt details with industry standards to ensure that the document is realistic and plausible.
9. Detection of duplicate or cloned document elements, such as copied text or images, suggesting fraud.
10. Cross-referencing document information with visual data from the image agent to check for consistency in damage, purchase location, and other details.
11. Cross-checking document details against audio analysis from the speech agent to ensure that the narrative matches the information provided in the document.
12. Inconsistencies between the document and image or speech data will lead to an increase in the fraud confidence score, suggesting the need for further investigation.
13. A confidence score reflecting the document's authenticity, with actionable insights on whether additional verification or investigation is required.
14. Suggestions for further document review, including the need for expert verification or contacting the claimant for clarification.
15. A final fraud likelihood score based on the combined findings from document, image, and speech agents, with recommendations for claim approval, denial, or further scrutiny.
""",
    func=analyze_document,
        args=[document_path]
    )

    task4 = Task(
        name = "Video Analysis",
agent = video_agent,
description = "Analyze video footage for fraud detection and inconsistencies and relate them with other agents data and check for any contradictory points which proves fraud claim.",
expected_output = """
Should be precise in 15 points plain text without any bold text and in paragraph. The output will be a detailed report highlighting:

1. Identification of tampering signs in the video, such as frame alterations or inconsistent visual content.
2. Detection of video artifacts, such as compression artifacts or pixelation, which may indicate manipulation or editing.
3. Analysis of damage consistency, ensuring that the video footage matches typical accidental scenarios.
4. Evaluation of the sequence of events in the video to ensure it aligns with the claimed accident or damage.
5. Detection of inconsistencies in object movement, such as unnatural pauses or jerky motions suggesting editing.
6. Examination of background elements for signs of artificial additions or editing, such as inconsistent scenes.
7. Verification of video file metadata, including timestamps and camera details, to check for signs of manipulation.
8. Identification of audio-visual discrepancies, such as mismatched sound or unnatural voice patterns.
9. Comparison of damage patterns in the video with known patterns of accidental damage to ensure authenticity.
10. Detection of blurry or pixelated video segments that might indicate video enhancement or manipulation.
11. Detection of any cut or jump in the video timeline that may suggest deliberate omission or editing.
12. Checking for irregularities in the scale and size of objects in the video that may indicate tampering.
13. Cross-checking the video’s scene consistency with other data sources to ensure all elements align with the claimed context.
14. Ensuring that any background noise, visual anomalies, or reflections do not deviate from what is realistic in an accident scenario.
15. A confidence score for the authenticity of the video, with actionable insights on whether further verification or investigation is needed.

Finally, consider the data from all agents (video, document, image, and speech) and provide a second paragraph with forecasting how the condition of the mobile might evolve over the next 6 months. The device will naturally deteriorate over time due to electronic wear and tear, but the degree of deterioration will depend on the extent of the damage. Provide monthly predictions with confidence scores in a tabular format and this should majorly depend on looks of the mobile phone regardless the fraud percentage, if the mobile looks goods then its high change that it will come long time.example: if the phone have cracks regardless other AI agents results the prediction should straightaway start from 40-50 and decrese rapidly, this dependes on amount of wear andtear.
""",
        func=analyze_medical_video,
        args=[video_path]
    )

    crew = Crew(tasks=[task1])
    results = crew.kickoff()
    # Function to get AI-generated responses
    def query_gemini(prompt):
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use correct model
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "No response from AI."


    # Define prompts for LLM
    prompt_fraud_analysis = f"in 3-4lines in simple english,Based on the given analysis, determine whether the claim is fraudulent or not. Provide a clear 'Yes' or 'No' answer with reasoning.\n\nText: {results}"
    prompt_ai_insights = f"in 4-5 lines in simple english,Identify anomalies, trends, and patterns from the given analysis and summarize key AI-driven insights.\n\nText: {results}"
    prompt_actionable_recommendations = f"in 4-5 lines in simple english,Generate actionable insights and recommendations for the claimant and business decision-makers based on the given analysis.\n\nText: {results}"
    prompt_prediction_table = f"Extract and format the prediction table from the given analysis, keeping the format intact.\n\nText: {results}"

    # Query the LLM
    fraud_analysis = query_gemini(prompt_fraud_analysis)
    ai_insights = query_gemini(prompt_ai_insights)
    actionable_recommendations = query_gemini(prompt_actionable_recommendations)
    prediction_table = query_gemini(prompt_prediction_table)

    st.title("Claim Analysis Report")
    st.title("Claim Analysis Report")

    st.subheader("1. Fraudulent Claim Analysis")
    st.info(fraud_analysis)

    st.subheader("2. AI-Driven Analysis Summary")
    st.warning(ai_insights)

    st.subheader("3. Actionable Insights & Recommendations")
    st.success(actionable_recommendations)

    st.subheader("4. Prediction Table")
    st.text(prediction_table)

    print('end')
# Check if a page is set in session state, otherwise default to page 1
if "page" not in st.session_state:
    st.session_state.page = "page_1"  # Default to page_1

# Page 1: File Uploads
if st.session_state.page == "page_1":
    # Streamlit UI
    # Inject custom CSS to prevent title wrapping
    st.markdown("""
        <style>
            .custom-title {
                white-space: nowrap;  /* Prevent text from wrapping to the next line */
                overflow: hidden;     /* Hide overflow if text is too long */
                text-overflow: ellipsis;  /* Add ellipsis if the text overflows */
                font-size: 36px;       /* Adjust font size as needed */
                font-weight: bold;     /* Make the title bold
                text-align: center;    /* Center-align the title */
            }
        </style>
    """, unsafe_allow_html=True)

    # Use the custom title inside st.markdown
    st.markdown('<h1 class="custom-title">Multi-Modal Agent AI Mobile Insurance Fraud Detection</h1>', unsafe_allow_html=True)

    # Inject custom CSS to add space between the columns
    st.markdown("""
        <style>
            .css-1d391kg { 
                margin-right: 20px;  /* Adjust the distance between columns */
            }
            .css-1v0mbdj { 
                margin-left: 20px;  /* Adjust the distance between columns */
            }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns for the top row (2 uploaders)
    col1, col2 = st.columns(2)

    with col1:
        image_path = st.file_uploader("Upload Medical Report Image", type=["png", "jpg", "jpeg"])

    with col2:
        audio_path = st.file_uploader("Upload Audio File", type=["mp3", "wav"])

    # Add some space between the columns
    st.markdown("<br>", unsafe_allow_html=True)

    # Create two columns for the bottom row (2 uploaders)
    col3, col4 = st.columns(2)

    with col3:
        document_path = st.file_uploader("Upload Document", type=["pdf", "csv"])

    with col4:
        video_path = st.file_uploader("Upload Video File", type=["mp4", "avi"])

    if st.button("Run Analysis"):
        if image_path and audio_path and document_path and video_path:
            try:
                multi_modal_analysis(image_path, audio_path, document_path, video_path)
            except RateLimitError as e:
                # Handle RateLimitError when API quota is exceeded
                print("Rate limit exceeded. Please check the quota for Vertex AI.")
                print(f"Error details: {e}")
                
            except Exception as e:
                # Catch any other exceptions that may occur
                print("An error occurred during multi-modal analysis.")
                print(f"Error details: {e}")
        else:
            st.write("Please upload all required files.")