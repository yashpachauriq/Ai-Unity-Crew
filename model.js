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

// Endpoint 1: Process and store a PDF resume
app.post("/process-pdf-resume", upload.single("resume"), async (req, res) => {
  const { status } = req.body;
  const { file } = req;

  if (!file || !status) {
    return res.status(400).json({ error: "Resume file and status are required." });
  }

  try {
    // Load and process the uploaded PDF document
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

    return res.status(200).json({ message: "PDF resume processed and stored successfully." });
  } catch (error) {
    console.error("Error processing PDF resume:", error);
    return res.status(500).json({ error: "Failed to process PDF resume." });
  }
});

app.post("/review-resume", upload.single("resume"), async (req, res) => {
  const { file } = req;

  if (!file) {
    return res.status(400).json({ error: "Resume file is required." });
  }

  if (!vectorStore) {
    // Attempt to load the vector store if not already loaded
    try {
      vectorStore = await FaissStore.load(directory, new OpenAIEmbeddings());
    } catch (error) {
      console.error("Error loading vector store:", error);
      return res.status(500).json({ error: "Failed to load vector store." });
    }
  }

  try {
    // Load and process the uploaded PDF document for review
    const reviewLoader = new PDFLoader(file.path);
    const reviewDocs = await reviewLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 10,
    });

    const reviewChunks = await splitter.splitDocuments(reviewDocs);
    const reviewResumeText = reviewChunks.map(chunk => chunk.pageContent).join("\n");

    // Use the vector store to review the resume
    const matchingDocs = await vectorStore.similaritySearch(reviewResumeText);
    console.log(matchingDocs)

    const llm = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const prompt = PromptTemplate.fromTemplate(
      `Evaluate the following resume against the stored resumes. Give a higher score if they have good projects, multiple languages, and good coding profiles.
      Also, consider if they have worked in core backend languages. 
      Provide a score between 0 and 100, and your reasoning:`
    );

    const chain = RunnableSequence.from([
      async () => ({
        context: matchingDocs.map(doc => doc.pageContent).join("\n"),
        resumeText: reviewResumeText,
      }),
      prompt,
      llm,
    ]);

    const response = await chain.invoke({
      context: matchingDocs.map(doc => doc.pageContent).join("\n"),
      resumeText: reviewResumeText,
    });

    console.log("Review Response:", response);

    // Parse the score from the response (assuming it's in a structured format)
    const score = parseInt(response.match(/Score: (\d+)/)[1], 10);
    let status;

    // Determine the status based on the score
    if (score >= 75) {
      status = "hired";
    } else {
      status = "not hired";
    }

    // Save the reviewed resume back to the vector store with the status
    const reviewChunksWithMetadata = reviewChunks.map(chunk => ({
      ...chunk,
      metadata: { status }
    }));

    await vectorStore.addDocuments(reviewChunksWithMetadata);
    await vectorStore.save(directory);

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
