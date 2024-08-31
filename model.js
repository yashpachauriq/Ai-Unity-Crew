import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import fs from "fs";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAI } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import dotenv from "dotenv";
import path from "path";
import cors from 'cors';
dotenv.config();


// Load environment variables

// Create an Express app
const app = express();
app.use(bodyParser.json());
app.use(cors()); 


// Set up multer for file handling
const upload = multer({ dest: "uploads/" });

// Define the directory to save and load the vector store
const directory = "./faiss-store";

let vectorStore;

// Utility function to clean up uploaded files
const cleanupFile = (filePath) => {
  fs.unlink(filePath, (err) => {
    if (err) console.error("Error deleting file:", err);
  });
};

app.use(cors({
  origin: '*', // Ensure this matches your frontend origin
  methods: ['GET', 'POST'], // Specify the methods your API will support
  allowedHeaders: ['Content-Type', 'Authorization'], // Include necessary headers
}));

// Handle preflight requests


// Endpoint 1: Process and store a PDF resume
app.post("/process-pdf-resume", upload.single("resume"), async (req, res) => {
  const { status, reason } = req.body;
  const { file } = req;

  if (!file || !status || !reason) {
    return res.status(400).json({ error: "Resume file, status, and requirements are required." });
  }

  try {
    const loader = new PDFLoader(file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 10,
    });

    const chunks = await splitter.splitDocuments(docs);

    // Add status and requirements to each chunk
    const chunksWithStatus = chunks.map(chunk => ({
      ...chunk,
      metadata: { status, reason }
    }));

    // Initialize or update the vector store
    if (!vectorStore) {
      vectorStore = await FaissStore.fromDocuments(chunksWithStatus, new OpenAIEmbeddings());
    } else {
      await vectorStore.addDocuments(chunksWithStatus);
    }

    // Save the vector store to the directory
    await vectorStore.save(directory);

    // Clean up the uploaded file
    cleanupFile(file.path);

    return res.status(200).json({ message: "PDF resume processed and stored successfully." });
  } catch (error) {
    console.error("Error processing PDF resume:", error);
    return res.status(500).json({ error: "Failed to process PDF resume." });
  }
});

app.post("/review-resume", upload.single("resume"), async (req, res) => {
  const { file } = req;
  const { requirements } = req.body;

  if (!file) {
    return res.status(400).json({ error: "Resume file is required." });
  }

  if (!requirements) {
    return res.status(400).json({ error: "Evaluation requirements are required." });
  }

  try {
    if (!vectorStore) {
      // Load the vector store if not already loaded
      vectorStore = await FaissStore.load(directory, new OpenAIEmbeddings());
    }

    const reviewLoader = new PDFLoader(file.path);
    const reviewDocs = await reviewLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 10,
    });

    const reviewChunks = await splitter.splitDocuments(reviewDocs);
    const reviewResumeText = reviewChunks.map(chunk => chunk.pageContent).join("\n");

    // Hardcoding the number of similar documents to retrieve
    const numSimilarDocs = 1;

    // Use the vector store to review the resume
    const matchingDocs = await vectorStore.similaritySearch(reviewResumeText, numSimilarDocs);
    const matchingText = matchingDocs.map(doc => doc.pageContent).join("\n");

    const llm = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    // Incorporate the requirements into the prompt
    const prompt = PromptTemplate.fromTemplate(
     `
      You are a recruiter evaluating resumes for a Product Manager position.
      *Requirements: These are requirements which are needed in the evaluation*
      ${requirements}
      *Resume to Evaluate:*
      ${reviewResumeText}
      *Context:*
      Below are the previously screened resumes which resemble the Resume to Evaluate, so take this as a small consideration while evaluating the Resume to Evaluate.
      ${matchingText}
      *Instructions for HR:*
      Perform an initial screening of the resume based on the Requirements and Context provided. Assign a score between 0 and 100 reflecting the candidate’s fit for the Product Manager role. Provide reasoning for the score, detailing how the candidate meets or falls short of the given criteria.
      If the candidate has less than 2 years of experience ensure them them a lower rating as well and not hire
     `
    );

    const chain = RunnableSequence.from([
      async () => ({
        context: matchingText,
        resumeText: reviewResumeText,
        requirements,
      }),
      prompt,
      llm,
    ]);

    const response = await chain.invoke({
      context: matchingText,
      resumeText: reviewResumeText,
      requirements,
    });

    // Clean up the uploaded file
    cleanupFile(file.path);

    return res.status(200).json({ message: "Resume reviewed successfully.", review: response });
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
