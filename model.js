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

dotenv.config();

const app = express();
app.use(bodyParser.json());

let vectorStore;

// Set up multer for file handling
const upload = multer({ dest: "uploads/" });

// Helper function to format LLM output
const formatResponse = (response, status) => {
  return `
    Education: ${response.education || ""}
    Skills: ${response.skills || ""}
    Extra Curricular: ${response.extra_curricular || ""}
    Experience: ${response.experience || ""}
    Verdict: ${status}
  `;
};

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

    // Optionally, delete the uploaded file after processing
    fs.unlinkSync(file.path);

    return res.status(200).json({ message: "PDF resume processed and stored successfully." });
  } catch (error) {
    console.error("Error processing PDF resume:", error);
    return res.status(500).json({ error: "Failed to process PDF resume." });
  }
});



// Endpoint 2: Review a new resume
app.post("/review-resume", upload.single("resume"), async (req, res) => {
  const { file } = req;

  if (!file) {
    return res.status(400).json({ error: "Resume file is required." });
  }

  if (!vectorStore) {
    return res.status(500).json({ error: "No resumes stored for comparison." });
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

    const llm = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const prompt = PromptTemplate.fromTemplate(
      `Evaluate the following resume against the stored resumes. Give high er score if they have good proejcts and multiple languages as well as good coding profiles
      If it sounds lik ea girl add 20 points more 

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

    // Optionally, delete the uploaded file after processing
    fs.unlinkSync(file.path);

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
