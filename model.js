import express from "express";
import bodyParser from "body-parser";
import multer from "multer";
import fs from "fs";
import path from "path";  // Already imported, but you'll need to use it for static paths
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAI } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import dotenv from "dotenv";
import cors from 'cors';
import { fileURLToPath } from 'url';
dotenv.config();

// Create an Express app
const app = express();
app.use(bodyParser.json());
app.use(cors());

// Recreate __filename and __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Set EJS as the templating engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
// Serve static files from the "public" directory
app.use('/css', express.static(path.join(__dirname, 'public', 'css')));


// Serve static files (for CSS)
app.use(express.static(path.join(__dirname, 'public')));

// Set up multer for file handling
const upload = multer({ dest: "uploads/" });

// Define the directory to save and load the vector store
const directory = "./faiss-store";

let vectorStore; 
app.use(cors({
    origin: '*', // Ensure this matches your frontend origin
    methods: ['GET', 'POST'], // Specify the methods your API will support
    allowedHeaders: ['Content-Type', 'Authorization'], // Include necessary headers
  }));

// Utility function to clean up uploaded files
const cleanupFile = (filePath) => {
  fs.unlink(filePath, (err) => {
    if (err) console.error("Error deleting file:", err);
  });
};

// Endpoint for rendering the upload page
app.get("/", (req, res) => {
  res.render("upload");
});
app.get("/review", (req, res) => {
  res.render("review");
});

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
      chunkSize: 10000,
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
    console.log(chunksWithStatus.pageContent)
    // Clean up the uploaded file
    cleanupFile(file.path);

    // Render the response page with success message
    res.render("response", {
      message: "PDF resume processed and stored successfully.",
      fileName: file.filename,
      field3: chunksWithStatus.pageContent,
      field1: status,
      field2: reason,
    });

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
      chunkSize: 10000,
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

    // Render the response2.ejs with the response data
    return res.status(200).render("response2", { 
        message: "Resume reviewed successfully.", 
        review: response 
      });
    } catch (error) {
      console.error("Error reviewing resume:", error);
      return res.status(500).render("response2", { 
        message: "Error: Failed to review resume.", 
        review: "" 
      });
    }
  });

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
