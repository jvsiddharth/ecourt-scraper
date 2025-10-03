import express from "express";
import dotenv from "dotenv";
import fetch from "node-fetch";
import { google } from "googleapis";
import fs from "fs";

dotenv.config();

const app = express();
app.use(express.json());

let drive;

// --- Initialize Google Drive ---
async function initDrive() {
  console.log("Initializing Google Drive...");
  const credentials = JSON.parse(fs.readFileSync("credentials.json"));
  const tokens = JSON.parse(fs.readFileSync("token.json"));
  const { client_secret, client_id, redirect_uris } = credentials.installed || credentials.web;
  const oAuth2Client = new google.auth.OAuth2(client_id, client_secret, redirect_uris[0]);
  oAuth2Client.setCredentials(tokens);
  drive = google.drive({ version: "v3", auth: oAuth2Client });
  console.log("✅ Google Drive ready");
}

// --- Helper Functions ---
async function listFiles(folderName) {
  try {
    const folderRes = await drive.files.list({
      q: `name='${folderName}' and mimeType='application/vnd.google-apps.folder'`,
      fields: "files(id)",
    });
    if (!folderRes.data.files.length) return "❌ Folder not found";
    
    const filesRes = await drive.files.list({
      q: `'${folderRes.data.files[0].id}' in parents`,
      fields: "files(name)",
    });
    if (!filesRes.data.files.length) return "📂 Empty folder";
    return filesRes.data.files.map((f) => "📄 " + f.name).join("\n");
  } catch (err) {
    return "❌ Error: " + err.message;
  }
}

async function deleteFile(folderName, fileName) {
  try {
    const folderRes = await drive.files.list({
      q: `name='${folderName}' and mimeType='application/vnd.google-apps.folder'`,
      fields: "files(id)",
    });
    if (!folderRes.data.files.length) return "❌ Folder not found";
    
    const fileRes = await drive.files.list({
      q: `name='${fileName}' and '${folderRes.data.files[0].id}' in parents`,
      fields: "files(id)",
    });
    if (!fileRes.data.files.length) return "❌ File not found";
    
    await drive.files.delete({ fileId: fileRes.data.files[0].id });
    return `✅ Deleted: ${fileName}`;
  } catch (err) {
    return "❌ Error: " + err.message;
  }
}

async function moveFile(fileName, destFolderName) {
  try {
    const fileRes = await drive.files.list({
      q: `name='${fileName}'`,
      fields: "files(id, parents)",
    });
    if (!fileRes.data.files.length) return "❌ File not found";
    
    const destFolderRes = await drive.files.list({
      q: `name='${destFolderName}' and mimeType='application/vnd.google-apps.folder'`,
      fields: "files(id)",
    });
    if (!destFolderRes.data.files.length) return "❌ Destination folder not found";
    
    const fileId = fileRes.data.files[0].id;
    const previousParents = fileRes.data.files[0].parents.join(",");
    
    await drive.files.update({
      fileId: fileId,
      addParents: destFolderRes.data.files[0].id,
      removeParents: previousParents,
      fields: "id, parents",
    });
    
    return `✅ Moved ${fileName} to ${destFolderName}`;
  } catch (err) {
    return "❌ Error: " + err.message;
  }
}

async function renameFile(oldName, newName) {
  try {
    const fileRes = await drive.files.list({
      q: `name='${oldName}'`,
      fields: "files(id)",
    });
    if (!fileRes.data.files.length) return "❌ File not found";
    
    await drive.files.update({
      fileId: fileRes.data.files[0].id,
      resource: { name: newName },
    });
    
    return `✅ Renamed to: ${newName}`;
  } catch (err) {
    return "❌ Error: " + err.message;
  }
}

async function summaryFolder(folderName) {
  try {
    const folderRes = await drive.files.list({
      q: `name='${folderName}' and mimeType='application/vnd.google-apps.folder'`,
      fields: "files(id)",
    });
    if (!folderRes.data.files.length) return "❌ Folder not found";
    
    const filesRes = await drive.files.list({
      q: `'${folderRes.data.files[0].id}' in parents`,
      fields: "files(name, mimeType)",
    });
    
    if (!filesRes.data.files.length) return "📂 Empty folder";
    
    const summary = `📊 ${folderName} Summary:\nTotal files: ${filesRes.data.files.length}\n` +
      filesRes.data.files.map(f => `• ${f.name}`).join("\n");
    
    return summary;
  } catch (err) {
    return "❌ Error: " + err.message;
  }
}

// --- Mock Testing Endpoint ---
app.post("/test", async (req, res) => {
  try {
    const text = (req.body.text || "").trim();
    const parts = text.split(/\s+/);
    const command = parts[0].toUpperCase();

    let result = "❓ Unknown command. Try: LIST, DELETE, MOVE, RENAME, SUMMARY";

    if (command === "LIST" && parts[1]) {
      result = await listFiles(parts[1].replace("/", ""));
    } else if (command === "DELETE" && parts[1]) {
      const pathParts = parts[1].replace(/\//g, " ").trim().split(" ");
      const folder = pathParts[0];
      const file = pathParts[1];
      result = await deleteFile(folder, file);
    } else if (command === "MOVE" && parts[1] && parts[2]) {
      const file = parts[1].replace(/^\/.*\//, "");
      const destFolder = parts[2].replace("/", "");
      result = await moveFile(file, destFolder);
    } else if (command === "RENAME" && parts[1] && parts[2]) {
      result = await renameFile(parts[1], parts[2]);
    } else if (command === "SUMMARY" && parts[1]) {
      result = await summaryFolder(parts[1].replace("/", ""));
    }

    console.log(`🧪 Test Command: "${text}" → Result:\n${result}`);
    return res.json({ input: text, output: result });
  } catch (err) {
    console.error("❌ Test Error:", err);
    return res.status(500).json({ error: err.message });
  }
});

// --- Health Check ---
app.get("/", (req, res) => res.send("WhatsApp Drive Bot Running ✅"));
app.get("/health", (req, res) => res.status(200).send("ok"));

// --- Start Server ---
const PORT = process.env.PORT || 3000;

async function start() {
  try {
    await initDrive();
    app.listen(PORT, () => {
      console.log(`🚀 Server running on port ${PORT}`);
      console.log(`🧪 Test endpoint: http://localhost:${PORT}/test`);
    });
  } catch (err) {
    console.error("❌ Startup error:", err);
    process.exit(1);
  }
}

start();
