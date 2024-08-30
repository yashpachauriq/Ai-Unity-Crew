import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { ChatOpenAI } from "@langchain/openai";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import pdfParse from "pdf-parse";

dotenv.config();

const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});
// Manually define __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const processResume = async (req, res, next) => {
  const { role } = req.body;
  const { file } = req;

  if (!file || !role) {
    return res.status(400).json({ error: "Resume file, status, and role are required." });
  }

  try {
    // Load and process the uploaded PDF document
    const loader = new PDFLoader(file.path);
    const docs = await loader.load();

    // Combine document text into a single string
    const resumeText = docs.map(doc => doc.pageContent).join("\n");

    // Load the role-specific template
    const templatePath = path.join(__dirname, `../templates/${role}-template.pdf`);
    if (!fs.existsSync(templatePath)) {
      return res.status(404).json({ error: "Template not found for the specified role." });
    }

    // Read the PDF template file as a buffer
  const templateBuffer = fs.readFileSync(templatePath);

  // Parse the PDF buffer to extract text
  const templateData = await pdfParse(templateBuffer);
  const templateText = templateData.text;

    // Construct the prompt for the LLM
    const prompt = `
      Use the provided template to format the following information from the resume text:
      
      Template:
      ${templateText}
      
      Resume Text:
      ${resumeText}
    `;

    // Invoke ChatGPT with the constructed prompt
    const response = await llm.invoke(prompt);

    // Attach the response to the request object for further use in the route
    req.formattedResume = response.content;

    // Optionally, delete the uploaded file after processing
    fs.unlinkSync(file.path);

    // Proceed to the next middleware or route handler
    next();
  } catch (error) {
    console.error("Error processing resume:", error);
    return res.status(500).json({ error: "Failed to process resume." });
  }
};

export default processResume;
