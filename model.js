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

// Load environment variables
dotenv.config();

// Create an Express app
const app = express();
app.use(bodyParser.json());

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

// Endpoint 1: Process and store a PDF resume
app.post("/process-pdf-resume", upload.single("resume"), async (req, res) => {
  const { status } = req.body;
  const { file } = req;

  if (!file || !status) {
    return res.status(400).json({ error: "Resume file and status are required." });
  }

  try {
    const loader = new PDFLoader(file.path);
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 10,
    });

    const chunks = await splitter.splitDocuments(docs);

    // Add status to each chunk
    const chunksWithStatus = chunks.map(chunk => ({
      ...chunk,
      metadata: { status }
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
  const { requirements } = req.body; // Removed numSimilarDocs since it will be hardcoded

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
      `You are a recruiter evaluating resumes. 
      Here are the specific requirements for this evaluation: {requirements}
      Consider the following criteria when scoring: 
      1. Quality of projects and relevant experience.
      2. Proficiency in multiple programming languages.
      3. Strength of coding profiles (e.g., GitHub, LeetCode).
      4. Experience in core backend technologies.
      Assign a score between 0 and 100 based on these criteria and the requirements provided, and provide reasoning for the score.`
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

    // Extract the score using a regex
    const scoreMatch = response.match(/Score:\s*(\d+)/);
    const score = scoreMatch ? parseInt(scoreMatch[1], 10) : 0;

    // Determine the status based on the score
    const status = score >= 75 ? "hired" : "not hired";

    // Save the reviewed resume back to the vector store with the status
    const reviewChunksWithMetadata = reviewChunks.map(chunk => ({
      ...chunk,
      metadata: { status }
    }));

    await vectorStore.addDocuments(reviewChunksWithMetadata);
    await vectorStore.save(directory);

    // Clean up the uploaded file
    cleanupFile(file.path);

    return res.status(200).json({ message: "Resume reviewed and stored successfully.", review: response, status });
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
