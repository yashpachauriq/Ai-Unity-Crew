import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import fs from "fs";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(bodyParser.json());

// Define the directory to save and load the vector store
const directory = "./faiss-store";

let vectorStore;

// Set up multer for file handling
const upload = multer({ dest: "uploads/" });

// Utility function to clean up uploaded files
const cleanupFile = (filePath) => {
  fs.unlink(filePath, (err) => {
    if (err) console.error("Error deleting file:", err);
  });
};

// Endpoint 1: Process and store a PDF resume
app.post("/process-pdf-resume", upload.single("resume"), async (req, res) => {
  const { file, body } = req;

  if (!file) {
    return res.status(400).json({ error: "Resume file is required." });
  }

  const { status, role, feedback } = body;

  if (!status || !role) {
    return res.status(400).json({ error: "Status and role are required." });
  }

  try {
    // Load and process the uploaded PDF document
    const loader = new PDFLoader(file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 10,
    });

    const chunks = await splitter.splitDocuments(docs);
    const resumeText = chunks.map(chunk => chunk.pageContent).join("\n");

    if (!vectorStore) {
      vectorStore = await FaissStore.fromDocuments(
        [
          {
            pageContent: resumeText,
            metadata: { status, role, feedback },
          },
        ],
        new OpenAIEmbeddings()
      );
    } else {
      await vectorStore.addDocuments([
        {
          pageContent: resumeText,
          metadata: { status, role, feedback },
        },
      ]);
    }

    // Save the vector store to the directory
    await vectorStore.save(directory);

    // Optionally, delete the uploaded file after processing
    cleanupFile(file.path);

    return res
      .status(200)
      .json({ message: "PDF resume processed and stored successfully." });
  } catch (error) {
    console.error("Error storing resume:", error);
    return res.status(500).json({ error: "Failed to store resume." });
  }
});

// Endpoint 2: Review a new resume
app.post("/review-resume", upload.single("resume"), async (req, res) => {
  const { file, body } = req;

  if (!file) {
    return res.status(400).json({ error: "Resume file is required." });
  }

  const { requirements } = body;

  if (!requirements) {
    return res.status(400).json({ error: "Evaluation requirements are required." });
  }

  try {
    if (!vectorStore) {
      vectorStore = await FaissStore.load(directory, new OpenAIEmbeddings());
    }

    // Load and process the uploaded PDF document for review
    const loader = new PDFLoader(file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 10,
    });

    const chunks = await splitter.splitDocuments(docs);
    const resumeText = chunks.map((chunk) => chunk.pageContent).join("\n");

    // Use the vector store to review the resume
    const matchingDocs = await vectorStore.similaritySearch(resumeText);
    const matchingText = matchingDocs
      .map((doc) => `${doc.pageContent}\nMetadata: ${JSON.stringify(doc.metadata)}`)
      .join("\n");

    const llm = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    // Create the prompt with context and requirements
    const prompt = `
      You are a recruiter evaluating resumes for a Product Manager position. 

      **Requirements: These are requirements which are needed in the evaluation**
      ${requirements}

      **Resume to Evaluate:**
      ${resumeText}

      **Context:**
      Context: Below are the previously screened resumes which resemble the Resume to Evaluate, so take this as a small consideration while evaluating the Resume to Evaluate.
      ${matchingText}

      **Instructions for HR:**
      Perform an initial screening of the resume based on the Requirements and Context provided. Assign a score between 0 and 100 reflecting the candidate’s fit for the Product Manager role. Provide reasoning for the score, detailing how the candidate meets or falls short of the given criteria. Focus exclusively on the provided resume and requirements, and use the context to align with our specific needs.
    `;

    // Generate response using the correct method
    const response = await llm.invoke({
      prompt
    });

    // Optionally, delete the uploaded file after processing
    cleanupFile(file.path);

    console.log(response);
    return res.status(200).json({
      message: "Resume reviewed successfully.",
      review: response.choices[0].text, // Extracting the text response
    });
  } catch (error) {
    console.error("Error reviewing resume:", error);
    return res.status(500).json({ error: "Failed to review resume." });
  }
});


// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
