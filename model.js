import express from "express";
import bodyParser from "body-parser";
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

// Endpoint 1: Process and store a resume
app.post("/process-resume", async (req, res) => {
  const { resumeText, status } = req.body;

  if (!resumeText || !status) {
    return res.status(400).json({ error: "Resume text and status are required." });
  }

  // LLM Prompt to extract details
  const promptTemplate = PromptTemplate.fromTemplate(
    `Extract the following details from the resume:

    Education:
    Skills:
    Extra Curricular:
    Experience:

    Resume:
    {resumeText}`
  );

  const llm = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const chain = RunnableSequence.from([
    { resumeText },
    promptTemplate,
    llm,
  ]);

  const response = await chain.invoke(req.body);
  const formattedResponse = formatResponse(response, status);

  // Split the formatted response into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 10,
  });

  const chunks = await splitter.splitDocuments([{ pageContent: formattedResponse }]);

  // Store the processed resume in the vector database
  if (!vectorStore) {
    vectorStore = await FaissStore.fromDocuments(chunks, new OpenAIEmbeddings());
  } else {
    await vectorStore.addDocuments(chunks);
  }

  return res.status(200).json({ message: "Resume processed and stored successfully." });
});

// Endpoint 2: Review a new resume
app.post("/review-resume", async (req, res) => {
  const { resumeText, requirements } = req.body;

  if (!resumeText || !requirements) {
    return res.status(400).json({ error: "Resume text and requirements are required." });
  }

  if (!vectorStore) {
    return res.status(500).json({ error: "No resumes stored for comparison." });
  }

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

  return res.status(200).json({ score: response });
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
