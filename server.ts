import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Initialize Gemini Client Lazily/Safely
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is missing in Secrets manager.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

// Health Check API
app.get("/api/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

// AI Customer Churn Retention Playbook API
app.post("/api/retention-playbook", async (req, res) => {
  try {
    const { 
      name, 
      tenure, 
      monthlyCharges, 
      usageFactor, 
      supportCalls, 
      contractType, 
      billingIssues, 
      predictedRisk 
    } = req.body;

    const client = getGeminiClient();

    const systemInstruction = 
      "You are a world-class Customer Success and Retention Strategist. " +
      "Analyze the high-risk customer account details and provide a professional, specific, immediately actionable retention playbook. " +
      "Frame suggestions around the customer's actual metrics (billing issues, high support calls, tenure, short/long-term contracts, drops in active usage). " +
      "Use clear professional headings with bullet points. Avoid filler introductions or fluff. Make instructions direct, tactical, and empathetic.";

    const prompt = `
Generate an enterprise customer retention playbook and personalized outreach approach for:
- **Customer Name**: ${name || 'Valued Subscriber'}
- **Predicted Churn Risk**: ${(predictedRisk * 100).toFixed(1)}%
- **Subscription Tenure**: ${tenure} months
- **Monthly Spend**: $${monthlyCharges}
- **Usage Frequency**: ${usageFactor}x (Ratio compared to historical average, values < 1.0 indicate usage is dropping)
- **Customer Support Cases (Last 30 Days)**: ${supportCalls} calls
- **Contract Commitment**: ${contractType} contract
- **Billing Complications/Errors**: ${billingIssues === 1 ? 'Yes, payment failure or billing dispute active' : 'No active complications'}

Structure your strategy clearly:
1. **Account Diagnostics**: Highlight the top 2-3 severe risk factors initiating their churn path.
2. **Personalized Outreach Script (Email / Phone Pitch)**: Write a direct, empathetic message designed representing high-touch customer support. Avoid generic templates, reference their actual tenure and issues (e.g. support calls or billing issues).
3. **Custom Retention Trigger Deal**: Offer a tailored discount, priority technical call, or contract benefit that leverages their contract style to secure another 12-month loyalty window. Give precise incentive pricing details (e.g., matching their monthly charges).
`;

    const response = await client.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        systemInstruction,
        temperature: 0.7,
      }
    });

    res.json({ 
      success: true, 
      playbook: response.text || "Failed to generate playbook output from Gemini." 
    });
  } catch (error: any) {
    console.error("AI Retention Playbook Error:", error);
    res.status(500).json({ 
      success: false, 
      error: error.message || "An unexpected error occurred during AI compilation." 
    });
  }
});

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server is running on port ${PORT}`);
  });
}

startServer();
