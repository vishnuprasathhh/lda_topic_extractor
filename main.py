from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile
import os
import logging
import sys

# Add the directory containing 'topic_lda' to Python's search path.
# This helps resolve ModuleNotFoundError if the current working directory isn't the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from topic_lda.extractor import LDAExtractor # Ensure this import path and class name are correct

# Configure logging for better visibility of application events and errors.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with metadata for documentation.
# This is the 'app' object that Uvicorn looks for.
app = FastAPI(
    title="NMF Topic Extractor API", # Updated title to reflect NMF
    description="Extracts descriptive topics from .docx files using NMF and an LLM.", # Updated description
    version="1.0.0"
)

@app.get("/")
async def root():
    """
    Provides a welcome message and directs users to the API documentation.
    This endpoint handles GET requests to the root URL.
    """
    logger.info("Accessed root endpoint.")
    return {
        "message": "Welcome to the NMF Topic Extractor API! Please visit /docs for interactive documentation.",
        "instructions": "Use the /extract-topics/ POST endpoint to upload a DOCX file and get topics."
    }

@app.post("/extract-topics/")
async def extract_topics_api(
    file: UploadFile = File(..., description="The DOCX file to extract topics from."),
    num_topics: int = Form(10, ge=1, description="The number of topics to extract (minimum 1).")
):
    """
    Extracts topics from an uploaded DOCX file using the NMF algorithm
    and generates descriptive topic titles using an LLM.

    Args:
        file (UploadFile): The uploaded .docx file.
        num_topics (int): The desired number of topics.

    Returns:
        dict: A dictionary containing the extracted topics.

    Raises:
        HTTPException: If the file type is incorrect, or if any error occurs
                       during the topic extraction pipeline.
    """
    tmp_path = None # Initialize temporary file path to None
    try:
        # Validate that the uploaded file is a .docx
        if not file.filename.endswith(".docx"):
            logger.warning(f"Invalid file type uploaded: {file.filename}. Only .docx is supported.")
            raise HTTPException(status_code=400, detail="Only .docx files are supported.")
        
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            contents = await file.read() # Read the content of the uploaded file asynchronously
            tmp.write(contents)          # Write the content to the temporary file
            tmp_path = tmp.name          # Store the path for later cleanup
        
        logger.info(f"Temporary file created at: {tmp_path} for uploaded file: {file.filename}")

        # Initialize the LDAExtractor (which now uses NMF and LLM internally)
        extractor = LDAExtractor(tmp_path, num_topics)
        
        # CRITICAL FIX: Await the asynchronous run_pipeline method
        topics = await extractor.run_pipeline()

        # Format the extracted topics for the API response
        formatted_topics = [f"{i+1}) {t}" for i, t in enumerate(topics)]
        
        logger.info(f"Successfully extracted {len(topics)} topics from {file.filename}.")
        return {
            "message": "✅ Topics extracted successfully!",
            "topics": formatted_topics
        }

    except FileNotFoundError as e:
        logger.error(f"File not found during processing: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"File not found or accessible: {e}")
    except ValueError as e:
        logger.error(f"Data processing error in extractor pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error during topic extraction: {e}")
    except HTTPException as e:
        # Re-raise explicit HTTPExceptions (e.g., 400 for wrong file type)
        raise
    except Exception as e:
        # Catch any other unexpected errors and log them comprehensively
        logger.exception(f"An unexpected error occurred during topic extraction for {file.filename}.")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # Ensure the temporary file is deleted, regardless of success or failure
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Temporary file deleted: {tmp_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary file {tmp_path}: {e}")

