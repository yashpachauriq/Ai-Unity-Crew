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

    // Store the processed chunks in the vector database
    if (!vectorStore) {
      vectorStore = await FaissStore.fromDocuments(chunksWithStatus, new OpenAIEmbeddings());
      console.log("Success in vector store here");
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
  const { requirements } = req.body;
  const { file } = req;

  if (!file || !requirements) {
    return res.status(400).json({ error: "Resume file and requirements are required." });
  }

  if (!vectorStore) {
    return res.status(500).json({ error: "No resumes stored for comparison." });
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
    const resumeText = chunks.map(chunk => chunk.pageContent).join("\n");

    // Retrieve matching resumes
    const retriever = vectorStore.asRetriever();
    const matchingResumes = await retriever.retrieve(resumeText);

    // Format the retrieved resumes
    const formatDocs = (docs) => {
      return docs.map((doc) => doc.pageContent).join("\n");
    };

    // Create a custom prompt for scoring
    const customPrompt = PromptTemplate.fromTemplate(
      `You are a hiring assistant. Based on the context provided below, evaluate the candidate's resume.

      Requirements: {requirements}

      Matching Resumes:
      {context}

      Candidate Resume:
      {resumeText}

      Provide a score between 0 and 100 and your reasoning on whether to consider this candidate or not:`
    );

    const llm = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const evaluateResumeChain = RunnableSequence.from([
      {
        context: matchingResumes.pipe(formatDocs),
        requirements: new RunnablePassthrough(),
        resumeText: new RunnablePassthrough(),
      },
      customPrompt,
      llm,
    ]);

    const response = await evaluateResumeChain.invoke({
      requirements,
      resumeText,
    });

    // Optionally, delete the uploaded file after processing
    fs.unlinkSync(file.path);

    return res.status(200).json({ score: response });
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
