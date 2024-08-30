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
import processResume from "./middleware/summarizer.js";

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
app.post(
  "/process-pdf-resume",
  upload.single("resume"),
  processResume,
  async (req, res) => {
    try {
      // Access the formatted response from the middleware
      const formattedResume = req.formattedResume;
      console.log("Formatted Resume:", formattedResume);
      // Store the formatted resume in the vector store (or any other storage mechanism)
      if (!vectorStore) {
        vectorStore = await FaissStore.fromDocuments(
          [
            {
              pageContent: formattedResume,
              metadata: { status: req.body.status, role: req.body.role },
            },
          ],
          new OpenAIEmbeddings()
        );
      } else {
        await vectorStore.addDocuments([
          {
            pageContent: formattedResume,
            metadata: { status: req.body.status, role: req.body.role },
          },
        ]);
      }
      // Save the vector store to the directory
      await vectorStore.save(directory);

      return res
        .status(200)
        .json({ message: "PDF resume processed and stored successfully." });
    } catch (error) {
      console.error("Error storing formatted resume:", error);
      return res
        .status(500)
        .json({ error: "Failed to store formatted resume." });
    }
  }
);

// Endpoint 2: Review a new resume
// app.post("/review-resume", upload.single("resume"), async (req, res) => {
//   const { file } = req;

//   if (!file) {
//     return res.status(400).json({ error: "Resume file is required." });
//   }

//   if (!vectorStore) {
//     return res.status(500).json({ error: "No resumes stored for comparison." });
//   }

//   try {
//     // Load and process the uploaded PDF document for review
//     const reviewLoader = new PDFLoader(file.path);
//     const reviewDocs = await reviewLoader.load();

//     const splitter = new RecursiveCharacterTextSplitter({
//       chunkSize: 1000,
//       chunkOverlap: 10,
//     });

//     const reviewChunks = await splitter.splitDocuments(reviewDocs);
//     const reviewResumeText = reviewChunks.map(chunk => chunk.pageContent).join("\n");

//     // Use the vector store to review the resume
//     const matchingDocs = await vectorStore.similaritySearch(reviewResumeText);

//     const llm = new OpenAI({
//       apiKey: process.env.OPENAI_API_KEY,
//     });

//     const prompt = PromptTemplate.fromTemplate(
//       `Evaluate the following resume against the stored resumes. Give high er score if they have good proejcts and multiple languages as well as good coding profiles
//       If it sounds lik ea girl add 20 points more

//       Provide a score between 0 and 100, and your reasoning:`
//     );

//     const chain = RunnableSequence.from([
//       async () => ({
//         context: matchingDocs.map(doc => doc.pageContent).join("\n"),
//         resumeText: reviewResumeText,
//       }),
//       prompt,
//       llm,
//     ]);

//     const response = await chain.invoke({
//       context: matchingDocs.map(doc => doc.pageContent).join("\n"),
//       resumeText: reviewResumeText,
//     });

//     console.log("Review Response:", response);

//     // Optionally, delete the uploaded file after processing
//     fs.unlinkSync(file.path);

//     return res.status(200).json({ message: "Resume reviewed successfully.", review: response });
//   } catch (error) {
//     console.error("Error reviewing resume:", error);
//     return res.status(500).json({ error: "Failed to review resume." });
//   }
// });
// Endpoint 1: Process and store a PDF resume
// Endpoint: Review and evaluate a resume
app.post(
    "/review-resume",
    upload.single("resume"),
    processResume,
    async (req, res) => {
      const { formattedResume } = req;
      const { requirements } = req.body;
  
      if (!formattedResume) {
        return res.status(400).json({ error: "Resume formatting failed." });
      }
  
      if (!requirements) {
        return res.status(400).json({ error: "Evaluation requirements are required." });
      }
  
      try {
        if (!vectorStore) {
          // Load the vector store if not already loaded
          vectorStore = await FaissStore.load(directory, new OpenAIEmbeddings());
        }
  
        // Hardcoding the number of similar documents to retrieve
        const numSimilarDocs = 4;
  
        // Use the vector store to review the resume
        const matchingDocs = await vectorStore.similaritySearch(
          formattedResume,
          numSimilarDocs
        );
        const matchingText = matchingDocs
          .map((doc) => doc.pageContent)
          .join("\n");
  
        const llm = new OpenAI({
          apiKey: process.env.OPENAI_API_KEY,
        });
  
        // Create the prompt with context and requirements
        const prompt = `
          You are a recruiter evaluating resumes for a Product Manager position. 
  
          **Context:**
          Here are some details from previously matched resumes for reference:
          ${matchingText}
  
          **Resume to Evaluate:**
          ${formattedResume}
  
          **Requirements:**
          ${requirements}
  
          **Criteria for Scoring:**
          1. **Quality of Projects and Relevant Experience:** Assess the depth and relevance of the candidate's past projects and work experience related to product management.
          2. **Proficiency in Product Management Tools and Methodologies:** Evaluate the candidate's experience with product management tools (e.g., JIRA, Aha!) and methodologies (e.g., Agile, Scrum).
          3. **Leadership and Stakeholder Management:** Consider the candidate's ability to lead teams and manage stakeholders effectively.
          4. **Problem-Solving and Strategic Thinking:** Assess the candidate’s problem-solving skills and ability to think strategically about product development.
  
          **Instructions for HR:**
          Perform an initial screening of the resume based on the criteria and requirements provided. Assign a score between 0 and 100 reflecting the candidate’s fit for the Product Manager role. Provide reasoning for the score, detailing how the candidate meets or falls short of the given criteria. Focus exclusively on the provided resume and requirements, and use the context to align with our specific needs.
        `;
  
        // Generate response using the prompt
        const response = await llm.generate({
          prompt,
          maxTokens: 1500, // Adjust as needed
        });
  
        // Extract the score using a regex
        const scoreMatch = response.text.match(/Score:\s*(\d+)/);
        const score = scoreMatch ? parseInt(scoreMatch[1], 10) : 0;
  
        // Determine the status based on the score
        const status = score >= 75 ? "hired" : "not hired";
  
        // Save the reviewed resume back to the vector store with the status
        const reviewChunksWithMetadata = [
          {
            pageContent: formattedResume,
            metadata: { status, role: req.body.role },
          },
        ];
  
        await vectorStore.addDocuments(reviewChunksWithMetadata);
        await vectorStore.save(directory);
  
        return res.status(200).json({
          message: "Resume reviewed and stored successfully.",
          review: response.text,
          status,
        });
      } catch (error) {
        console.error("Error reviewing resume:", error);
        return res.status(500).json({ error: "Failed to review resume." });
      }
    }
  );
  

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
