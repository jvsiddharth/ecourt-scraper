import uuid
import time
import io
import json
import os
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, send_file, Response, send_from_directory
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import traceback
from weasyprint import HTML, CSS
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import easyocr
import base64


app = Flask(__name__)


# File to store search history
HISTORY_FILE = "search_history.json"
# Directory to store PDFs
PDF_STORAGE_DIR = "saved_pdfs"

# Create PDF storage directory if it doesn't exist
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)


# In-memory sessions: session_id -> {"driver": WebDriver, "last_active": float, "scraped_results": list}
sessions = {}



def load_history():
    """Load search history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading history: {e}")
    return []



def save_history(history):
    """Save search history to file with explicit flush"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            f.flush()  # Force write to disk
            os.fsync(f.fileno())  # Ensure OS writes to disk
    except IOError as e:
        print(f"Error saving history: {e}")



def add_to_history(session_id, url, court_name, case_type, reg_no, reg_year, results_count):
    """Add a search to history"""
    try:
        history = load_history()
        history.append({
            "session_id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": url,
            "court_name": court_name,
            "case_type": case_type,
            "reg_no": reg_no,
            "reg_year": reg_year,
            "results_count": results_count,
            "case_details_pdfs": []  # Will store CNO -> PDF filename mappings
        })
        save_history(history)
        print(f"History saved: {len(history)} total entries")  # Debug log
    except Exception as e:
        print(f"Error adding to history: {e}")


def update_history_with_pdf(session_id, data_cno, pdf_filename):
    """Update history entry with saved PDF filename"""
    try:
        history = load_history()
        for entry in history:
            if entry["session_id"] == session_id:
                if "case_details_pdfs" not in entry:
                    entry["case_details_pdfs"] = []
                entry["case_details_pdfs"].append({
                    "cno": data_cno,
                    "filename": pdf_filename
                })
                break
        save_history(history)
    except Exception as e:
        print(f"Error updating history with PDF: {e}")

def preprocess_captcha_image(image_data):
    """
    Enhanced preprocessing for difficult captchas
    """
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Try multiple thresholding techniques
        # Method 1: Otsu's thresholding
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(closing, None, 10, 7, 21)
        
        # Upscale image for better OCR
        scale_percent = 200  # percent of original size
        width = int(denoised.shape[1] * scale_percent / 100)
        height = int(denoised.shape[0] * scale_percent / 100)
        upscaled = cv2.resize(denoised, (width, height), interpolation=cv2.INTER_CUBIC)
        
        return upscaled
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def solve_captcha_tesseract(image_data):
    """
    Try multiple Tesseract configurations
    """
    try:
        processed_img = preprocess_captcha_image(image_data)
        
        if processed_img is None:
            return None
        
        pil_img = Image.fromarray(processed_img)
        
        # Try multiple configurations
        configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ]
        
        best_result = None
        max_length = 0
        
        for config in configs:
            captcha_text = pytesseract.image_to_string(pil_img, config=config).strip()
            captcha_text = ''.join(filter(str.isalnum, captcha_text))
            
            # Keep the longest valid result
            if len(captcha_text) > max_length:
                max_length = len(captcha_text)
                best_result = captcha_text
        
        print(f"Tesseract OCR result: {best_result}")
        return best_result if best_result else None
        
    except Exception as e:
        print(f"Error in Tesseract OCR: {e}")
        return None


def solve_captcha_easyocr(image_data):
    """
    Solve captcha using EasyOCR (more accurate for complex captchas)
    """
    try:
        # Initialize EasyOCR reader (only once, can be global)
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Preprocess the image
        processed_img = preprocess_captcha_image(image_data)
        
        if processed_img is None:
            return None
        
        # Read text from image
        results = reader.readtext(processed_img, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        
        if results:
            captcha_text = ''.join(results).strip()
            # Clean the result
            captcha_text = ''.join(filter(str.isalnum, captcha_text))
            print(f"EasyOCR result: {captcha_text}")
            return captcha_text
        
        return None
        
    except Exception as e:
        print(f"Error in EasyOCR: {e}")
        return None


def get_captcha_image_from_driver(driver):
    """
    Extract captcha image from Selenium driver
    """
    try:
        captcha_img = driver.find_element("css selector", "#siwp_captcha_image_0")
        # Get image as base64 or screenshot
        captcha_screenshot = captcha_img.screenshot_as_png
        return captcha_screenshot
    except Exception as e:
        print(f"Error extracting captcha image: {e}")
        return None


def validate_captcha_result(captcha_text, expected_length=5):
    """
    Validate CAPTCHA result
    """
    if not captcha_text:
        return False
    
    # Check if result has reasonable length
    if len(captcha_text) < expected_length - 1 or len(captcha_text) > expected_length + 2:
        return False
    
    # Check if result contains only alphanumeric characters
    if not captcha_text.isalnum():
        return False
    
    return True


def auto_solve_captcha(driver, method='easyocr'):
    """
    Automatically solve captcha - try EasyOCR first
    """
    try:
        captcha_image_data = get_captcha_image_from_driver(driver)
        
        if captcha_image_data is None:
            return None
        
        # Try EasyOCR first (better for distorted images)
        captcha_solution = solve_captcha_easyocr(captcha_image_data)
        
        # Fallback to Tesseract if EasyOCR fails
        if not captcha_solution or len(captcha_solution) < 4:
            print("EasyOCR failed or result too short, trying Tesseract...")
            captcha_solution = solve_captcha_tesseract(captcha_image_data)
        
        return captcha_solution
        
    except Exception as e:
        print(f"Error in auto_solve_captcha: {e}")
        return None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dynamic Court Scraper with Selenium Session</title>
    <style>
        body { background: #f0f2f5; font-family: Arial; padding: 20px; }
        .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        label { display: block; margin-top: 10px; }
        select, input { width: 100%; padding: 8px; margin-top: 5px; }
        button { margin-top: 15px; padding: 10px; width: 100%; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .download-btn { background: #28a745; margin-top: 5px; }
        .download-btn:hover { background: #218838; }
        img { margin-top: 10px; display: block; max-width: 100%; }
        #saved_results_container div { border: 1px solid #ccc; margin-top: 10px; padding: 5px; border-radius: 4px; max-height: 300px; overflow: auto; }
        a.viewCnrDetails.btn.accent-color { cursor: pointer; color: blue; text-decoration: underline; }
        /* modal styles */
        #detailModal { display: none; position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.5); justify-content:center; align-items:center; z-index: 1000; }
        #modalContent { background: white; padding: 20px; max-width: 90vw; max-height: 90vh; overflow: auto; border-radius: 8px; position: relative; }
        #closeModal { float:right; cursor:pointer; font-weight: bold; font-size: 20px; }
        .history-item { border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 4px; background: #f9f9f9; cursor: pointer; transition: all 0.3s; }
        .history-item:hover { background: #e0e7ff; transform: translateX(5px); box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .history-title { font-weight: bold; color: #007bff; margin-bottom: 5px; }
        .history-details { font-size: 0.9em; color: #666; margin-top: 5px; }
        .history-badge { display: inline-block; background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 5px; }
        #historySection { margin-top: 20px; max-height: 400px; overflow-y: auto; }
        .loading { color: #666; font-style: italic; }
        .pdf-link { display: block; padding: 5px; margin: 5px 0; background: #e3f2fd; border-radius: 3px; text-decoration: none; color: #1976d2; }
        .pdf-link:hover { background: #bbdefb; }
    </style>
</head>
<body>
<div class="container">
    <h2>Court Form Scraper with Selenium Session</h2>


    <label>Target URL:</label>
    <input type="text" id="target_url" placeholder="Enter Court Page URL">
    <button id="startSessionBtn">Start Session & Load Form</button>


    <div id="historySection">
        <h3>Previous Searches <span style="font-size: 0.8em; color: #666;">(Click to view details)</span></h3>
        <div id="historyContainer"></div>
    </div>


    <div id="session_area" style="display:none;">
        <p><strong>Session ID:</strong> <span id="session_id_display"></span></p>


        <div align="center" style="margin-top:10px; vertical-align: middle;">
            <label style="display:inline-block; margin-right:10px;">
                <input type="radio" name="court_type_choice" value="complex" checked> Complex
            </label>
            <label style="display:inline-block;">
                <input type="radio" name="court_type_choice" value="establishment"> Establishment
            </label>
        </div>


        <label>Court Complex/Establishment:</label>
        <select id="court_select"></select>


        <label>Case Type:</label>
        <select id="case_type_select"></select>


        <label>Case Number:</label>
        <input type="text" id="reg_no">


        <label>Case Year:</label>
        <input type="text" id="reg_year">


        <label>Captcha:</label>
        <img id="captcha_img" src="">
        <input type="text" id="captcha_value" placeholder="Enter Captcha">
        <button id="autoSolveCaptchaBtn" style="background: #17a2b8; margin-top: 10px;">Auto Solve Captcha</button>


        <button id="submitBtn">Search</button>


        <h3>Saved Results</h3>
        <button class="download-btn" id="downloadAllResultsBtn">Download All Results as PDF</button>
        <div id="saved_results_container"></div>
    </div>
</div>


<!-- Modal for case details -->
<div id="detailModal">
    <div id="modalContent">
        <span id="closeModal">&times;</span>
        <button class="download-btn" id="downloadDetailBtn" style="margin-bottom: 10px;">Download & Save Details as PDF</button>
        <div id="modalBody"></div>
    </div>
</div>


<script>
let sessionId = null;
let currentDetailCno = null;


// Load history on page load
window.onload = async () => {
    await loadHistory();
};


async function loadHistory() {
    const resp = await fetch("/get_history");
    const data = await resp.json();
    const container = document.getElementById("historyContainer");
    container.innerHTML = "";
    
    if(data.history.length === 0) {
        container.innerHTML = "<p style='color: #666;'>No previous searches yet.</p>";
        return;
    }
    
    data.history.reverse().forEach((item, idx) => {
        const div = document.createElement("div");
        div.className = "history-item";
        div.onclick = () => viewHistoryDetails(item.session_id);
        
        let pdfBadge = "";
        if(item.case_details_pdfs && item.case_details_pdfs.length > 0) {
            pdfBadge = `<span class="history-badge" style="background: #ff9800;">📄 ${item.case_details_pdfs.length} PDFs</span>`;
        }
        
        div.innerHTML = `
            <div class="history-title">
                ${item.court_name} - Case ${item.reg_no}/${item.reg_year}
                <span class="history-badge">${item.results_count || 0} results</span>
                ${pdfBadge}
            </div>
            <div class="history-details">
                📅 ${item.timestamp}<br>
                📋 ${item.case_type}<br>
                🆔 Session: ${item.session_id.substring(0, 8)}...
            </div>
        `;
        container.appendChild(div);
    });
}


async function viewHistoryDetails(historySessionId) {
    const modal = document.getElementById("detailModal");
    const modalBody = document.getElementById("modalBody");
    
    modalBody.innerHTML = "<p class='loading'>Loading search details...</p>";
    modal.style.display = "flex";
    
    try {
        const resp = await fetch(`/get_history_details?session_id=${historySessionId}`);
        const data = await resp.json();
        
        if(data.error) {
            modalBody.innerHTML = `<p style="color: red;">${data.error}</p>`;
            return;
        }
        
        let html = `
            <h2>Search Details</h2>
            <p><strong>Court:</strong> ${data.court_name}</p>
            <p><strong>Case:</strong> ${data.reg_no}/${data.reg_year}</p>
            <p><strong>Type:</strong> ${data.case_type}</p>
            <p><strong>Date:</strong> ${data.timestamp}</p>
            <hr>
        `;
        
        // Display saved PDFs
        if(data.case_details_pdfs && data.case_details_pdfs.length > 0) {
            html += `<h3>Saved Case Detail PDFs (${data.case_details_pdfs.length})</h3><div style="margin: 10px 0;">`;
            data.case_details_pdfs.forEach((pdf, idx) => {
                html += `<a class="pdf-link" href="/download_saved_pdf?filename=${pdf.filename}" target="_blank">
                    📄 CNO: ${pdf.cno} (Saved: ${pdf.filename})
                </a>`;
            });
            html += `</div><hr>`;
        }
        
        html += `<h3>Results (${data.results_count})</h3>`;
        
        if(data.results && data.results.length > 0) {
            data.results.forEach((result, idx) => {
                html += `<div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 4px;">
                    <h4>Result ${idx + 1}</h4>
                    ${result}
                </div>`;
            });
            
            // Add download button for this history
            html += `<button class="download-btn" onclick="downloadHistoryPDF('${historySessionId}')" style="width: 100%; margin-top: 10px;">Download This Search as PDF</button>`;
        } else {
            html += "<p>No results available for this search.</p>";
        }
        
        modalBody.innerHTML = html;
        
    } catch(error) {
        modalBody.innerHTML = `<p style="color: red;">Error loading details: ${error.message}</p>`;
    }
}


function downloadHistoryPDF(historySessionId) {
    window.open(`/download_history_pdf?session_id=${historySessionId}`, '_blank');
}


document.getElementById("startSessionBtn").onclick = async () => {
    const url = document.getElementById("target_url").value.trim();
    if(!url) {
        alert("Please enter a URL");
        return;
    }
    const resp = await fetch("/start_session", {
        method: "POST",
        body: new URLSearchParams({url}),
    });
    const data = await resp.json();
    if(data.error){
        alert("Error: " + data.error);
        return;
    }
    sessionId = data.session_id;
    document.getElementById("session_id_display").textContent = sessionId;
    document.getElementById("session_area").style.display = "block";
    await loadForm();
    await loadSavedResults();
};


async function loadForm(){
    const resp = await fetch("/load_form", {
        method: "POST",
        body: new URLSearchParams({session_id: sessionId})
    });
    const data = await resp.json();
    document.getElementById("court_select").innerHTML = data.options_html;
    document.getElementById("captcha_img").src = data.captcha_url;
    document.getElementById("case_type_select").innerHTML = "";
    setupListeners();
}


function getCurrentChoice(){
    return document.querySelector('input[name="court_type_choice"]:checked').value;
}


function setupListeners(){
    document.getElementById("court_select").addEventListener("change", async () => {
        const court_value = document.getElementById("court_select").value;
        const choice = getCurrentChoice();
        const resp = await fetch("/load_case_type", {
            method: "POST",
            body: new URLSearchParams({session_id: sessionId, court_value, choice})
        });
        const data = await resp.json();
        document.getElementById("case_type_select").innerHTML = data.options_html;
        document.getElementById("captcha_img").src = data.captcha_url;
    });


    document.querySelectorAll('input[name="court_type_choice"]').forEach(radio =>
        radio.addEventListener("change", async () => {
            const choice = getCurrentChoice();
            const resp = await fetch("/reload_options", {
                method: "POST",
                body: new URLSearchParams({session_id: sessionId, choice})
            });
            const data = await resp.json();
            document.getElementById("court_select").innerHTML = data.options_html;
            document.getElementById("captcha_img").src = data.captcha_url;
            document.getElementById("case_type_select").innerHTML = "";
        })
    );
}


document.getElementById("submitBtn").onclick = async () => {
    const body = new URLSearchParams({
        session_id: sessionId,
        court_value: document.getElementById("court_select").value,
        case_type_value: document.getElementById("case_type_select").value,
        reg_no: document.getElementById("reg_no").value,
        reg_year: document.getElementById("reg_year").value,
        captcha_value: document.getElementById("captcha_value").value,
        choice: getCurrentChoice()
    });
    const resp = await fetch("/submit_form", {method:"POST", body});
    const content = await resp.text();
    document.getElementById("saved_results_container").innerHTML = "";
    await loadSavedResults();
    await loadHistory(); // Refresh history after new search
};


async function loadSavedResults(){
    if(!sessionId) return;
    const resp = await fetch(`/saved_results?session_id=${sessionId}`);
    const data = await resp.json();
    const container = document.getElementById("saved_results_container");
    container.innerHTML = "";
    data.results.forEach((html, idx) => {
        const div = document.createElement("div");
        div.innerHTML = html;


        div.querySelectorAll("a.viewCnrDetails.btn.accent-color").forEach(link => {
            link.onclick = async (e) => {
                e.preventDefault();
                const data_cno = link.getAttribute("data-cno");
                await viewCaseDetails(data_cno);
            };
        });


        container.appendChild(div);
    });
}

document.getElementById("autoSolveCaptchaBtn").onclick = async () => {
    if(!sessionId) {
        alert("No session active");
        return;
    }
    
    const autoSolveBtn = document.getElementById("autoSolveCaptchaBtn");
    autoSolveBtn.disabled = true;
    autoSolveBtn.textContent = "Solving...";
    
    try {
        const resp = await fetch("/auto_solve_captcha", {
            method: "POST",
            body: new URLSearchParams({
                session_id: sessionId,
                method: "tesseract"  // or "easyocr"
            })
        });
        
        const data = await resp.json();
        
        if(data.success) {
            document.getElementById("captcha_value").value = data.captcha_text;
            alert("Captcha solved: " + data.captcha_text);
        } else {
            alert("Failed to solve captcha: " + (data.error || "Unknown error"));
        }
    } catch(error) {
        alert("Error: " + error.message);
    } finally {
        autoSolveBtn.disabled = false;
        autoSolveBtn.textContent = "Auto Solve Captcha";
    }
};

async function loadForm(){
    const resp = await fetch("/load_form", {
        method: "POST",
        body: new URLSearchParams({session_id: sessionId})
    });
    const data = await resp.json();
    document.getElementById("court_select").innerHTML = data.options_html;
    document.getElementById("captcha_img").src = data.captcha_url;
    document.getElementById("case_type_select").innerHTML = "";
    setupListeners();
    
    // Auto-solve captcha
    await autoSolveCaptchaOnLoad();
}

async function autoSolveCaptchaOnLoad() {
    // Wait a bit for image to load
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const resp = await fetch("/auto_solve_captcha", {
        method: "POST",
        body: new URLSearchParams({
            session_id: sessionId,
            method: "tesseract"
        })
    });
    
    const data = await resp.json();
    if(data.success) {
        document.getElementById("captcha_value").value = data.captcha_text;
        console.log("Auto-solved captcha:", data.captcha_text);
    }
}


document.getElementById("downloadAllResultsBtn").onclick = async () => {
    if(!sessionId) {
        alert("No session active");
        return;
    }
    window.open(`/download_results_pdf?session_id=${sessionId}`, '_blank');
};


async function viewCaseDetails(data_cno){
    currentDetailCno = data_cno;
    const resp = await fetch("/view_case_details", {
        method: "POST",
        body: new URLSearchParams({session_id: sessionId, data_cno})
    });
    const contentType = resp.headers.get("Content-Type") || "";
    if(contentType.includes("application/pdf")){
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        window.open(url);
    } else {
        const html = await resp.text();
        showModal(html);
    }
}


document.getElementById("downloadDetailBtn").onclick = async () => {
    if(!currentDetailCno || !sessionId) {
        alert("No case details to download");
        return;
    }
    
    // Download and save the PDF
    const resp = await fetch(`/download_case_detail_pdf?session_id=${sessionId}&data_cno=${currentDetailCno}`);
    if(resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `case_details_${currentDetailCno}_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        alert("PDF saved successfully!");
        await loadHistory(); // Refresh history to show new PDF
    } else {
        alert("Failed to download PDF");
    }
};


function showModal(content){
    const modal = document.getElementById("detailModal");
    document.getElementById("modalBody").innerHTML = content;
    modal.style.display = "flex";
}


document.getElementById("closeModal").onclick = () => {
    document.getElementById("detailModal").style.display = "none";
    currentDetailCno = null;
};


window.onclick = function(event) {
    const modal = document.getElementById("detailModal");
    if(event.target == modal){
        modal.style.display = "none";
        currentDetailCno = null;
    }
};
</script>
</body>
</html>
"""



def create_driver(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver



def get_options_and_captcha(driver, selector):
    try:
        select_elem = driver.find_element("css selector", selector)
        select = Select(select_elem)
        options = [{"value": option.get_attribute("value"), "text": option.text} for option in select.options if option.get_attribute("value")]
    except Exception:
        options = []
    try:
        img = driver.find_element("css selector", "#siwp_captcha_image_0")
        captcha_url = img.get_attribute("src")
    except Exception:
        captcha_url = ""
    return options, captcha_url



@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)



@app.route("/get_history", methods=["GET"])
def get_history():
    history = load_history()
    return jsonify({"history": history})



@app.route("/get_history_details", methods=["GET"])
def get_history_details():
    session_id = request.args.get("session_id")
    history = load_history()
    
    # Find the specific history entry
    history_entry = next((h for h in history if h["session_id"] == session_id), None)
    
    if not history_entry:
        return jsonify({"error": "History not found"}), 404
    
    # Check if session still exists
    session = sessions.get(session_id)
    if session and "scraped_results" in session:
        history_entry["results"] = session["scraped_results"]
    else:
        history_entry["results"] = []
    
    return jsonify(history_entry)



@app.route("/start_session", methods=["POST"])
def start_session():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    session_id = str(uuid.uuid4())
    try:
        driver = create_driver(url)
    except Exception as e:
        return jsonify({"error": f"Failed to create browser: {str(e)}"}), 500
    sessions[session_id] = {
        "driver": driver, 
        "last_active": time.time(), 
        "scraped_results": [],
        "url": url
    }
    return jsonify({"session_id": session_id})



@app.route("/load_form", methods=["POST"])
def load_form():
    session_id = request.form.get("session_id")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session"}), 400
    driver = session["driver"]
    session["last_active"] = time.time()


    options, captcha_url = get_options_and_captcha(driver, "select#est_code")
    return jsonify({
        "options_html": "".join([f"<option value='{o['value']}'>{o['text']}</option>" for o in options]),
        "captcha_url": captcha_url
    })



@app.route("/load_case_type", methods=["POST"])
def load_case_type():
    session_id = request.form.get("session_id")
    court_value = request.form.get("court_value")
    choice = request.form.get("choice", "complex")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session"}), 400
    driver = session["driver"]
    session["last_active"] = time.time()


    sel = "select#est_code" if choice == "complex" else "select#court_establishment"


    try:
        court_select = Select(driver.find_element("css selector", sel))
        court_select.select_by_value(court_value)
        time.sleep(1)
        case_type_select = Select(driver.find_element("css selector", "select#case_type"))
        options = [{"value": o.get_attribute("value"), "text": o.text} for o in case_type_select.options if o.get_attribute("value")]
        img = driver.find_element("css selector", "#siwp_captcha_image_0")
        captcha_url = img.get_attribute("src")
    except Exception as e:
        return jsonify({"error": f"Error loading case types: {str(e)}"}), 500


    return jsonify({
        "options_html": "".join([f"<option value='{o['value']}'>{o['text']}</option>" for o in options]),
        "captcha_url": captcha_url
    })


@app.route("/reload_options", methods=["POST"])
def reload_options():
    session_id = request.form.get("session_id")
    choice = request.form.get("choice")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session"}), 400
    driver = session["driver"]
    session["last_active"] = time.time()


    sel = "select#est_code" if choice == "complex" else "select#court_establishment"


    try:
        select_elem = driver.find_element("css selector", sel)
        select = Select(select_elem)
        options = [{"value": o.get_attribute("value"), "text": o.text} for o in select.options if o.get_attribute("value")]
        img = driver.find_element("css selector", "#siwp_captcha_image_0")
        captcha_url = img.get_attribute("src")
    except Exception as e:
        return jsonify({"error": f"Error reloading options: {str(e)}"}), 500


    return jsonify({
        "options_html": "".join([f"<option value='{o['value']}'>{o['text']}</option>" for o in options]),
        "captcha_url": captcha_url
    })

@app.route("/auto_solve_captcha", methods=["POST"])
def auto_solve_captcha_endpoint():
    """
    Endpoint to automatically solve the captcha using OCR
    """
    session_id = request.form.get("session_id")
    method = request.form.get("method", "tesseract")  # 'tesseract' or 'easyocr'
    
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session"}), 400
    
    driver = session["driver"]
    session["last_active"] = time.time()
    
    try:
        captcha_solution = auto_solve_captcha(driver, method)
        
        if captcha_solution:
            return jsonify({
                "success": True,
                "captcha_text": captcha_solution
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to solve captcha"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error solving captcha: {str(e)}"
        }), 500


@app.route("/submit_form", methods=["POST"])
def submit_form():
    session_id = request.form.get("session_id")
    court_value = request.form.get("court_value")
    case_type_value = request.form.get("case_type_value")
    reg_no = request.form.get("reg_no")
    reg_year = request.form.get("reg_year")
    captcha_value = request.form.get("captcha_value")
    choice = request.form.get("choice", "complex")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session"}), 400
    driver = session["driver"]
    session["last_active"] = time.time()


    sel = "select#est_code" if choice == "complex" else "select#court_establishment"


    try:
        court_select = Select(driver.find_element("css selector", sel))
        court_select.select_by_value(court_value)
        
        # Get court name for history
        court_name = court_select.first_selected_option.text
        
        time.sleep(1)
        case_type_select = Select(driver.find_element("css selector", "select#case_type"))
        case_type_select.select_by_value(case_type_value)
        
        # Get case type name for history
        case_type_name = case_type_select.first_selected_option.text
        
        driver.find_element("css selector", "input#reg_no").clear()
        driver.find_element("css selector", "input#reg_no").send_keys(reg_no)
        driver.find_element("css selector", "input#reg_year").clear()
        driver.find_element("css selector", "input#reg_year").send_keys(reg_year)
        driver.find_element("css selector", "input#siwp_captcha_value_0").clear()
        driver.find_element("css selector", "input#siwp_captcha_value_0").send_keys(captcha_value)
        driver.find_element("css selector", "input[type=submit]").click()
        time.sleep(3)


        results_div = driver.find_element("css selector", "div.resultsHolder.servicesResultsContainer")
        scraped_html = results_div.get_attribute("innerHTML")


        if "scraped_results" not in session:
            session["scraped_results"] = []
        session["scraped_results"].append(scraped_html)
        
        # Add to history with results count
        add_to_history(session_id, session.get("url", ""), court_name, case_type_name, 
                      reg_no, reg_year, len(session["scraped_results"]))


    except Exception as e:
        return jsonify({"error": f"Error submitting form: {str(e)}"}), 500


    return scraped_html



@app.route("/saved_results", methods=["GET"])
def saved_results():
    session_id = request.args.get("session_id")
    session = sessions.get(session_id)
    if not session or "scraped_results" not in session:
        return jsonify({"results": []})
    return jsonify({"results": session["scraped_results"]})


@app.route("/view_case_details", methods=["POST"])
def view_case_details():
    session_id = request.form.get("session_id")
    data_cno = request.form.get("data_cno")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session"}), 400
    driver = session["driver"]


    try:
        selector = f'a.viewCnrDetails.btn.accent-color[data-cno="{data_cno}"]'
        wait = WebDriverWait(driver, 15)
        view_link = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))


        driver.execute_script("arguments[0].scrollIntoView(true);", view_link)
        driver.execute_script("arguments[0].click();", view_link)


        # Wait for the table to load
        wait.until(EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'data-table-1') and caption[text()='Case Details']]")))
        time.sleep(2)  # Extra wait for dynamic content


        scraped_data = {}


        tables = driver.find_elements(By.CSS_SELECTOR, "table.data-table-1")
        
        for table_idx, table in enumerate(tables):
            try:
                caption = table.find_element(By.TAG_NAME, "caption").text.strip()
            except:
                caption = f"Table {table_idx + 1}"
            
            # Get the entire HTML of the table for debugging
            table_html = table.get_attribute("outerHTML")
            
            # Extract all rows (both thead and tbody)
            thead_rows = []
            tbody_rows = []
            
            try:
                thead = table.find_element(By.TAG_NAME, "thead")
                thead_rows = thead.find_elements(By.TAG_NAME, "tr")
            except:
                pass
            
            try:
                tbody = table.find_element(By.TAG_NAME, "tbody")
                tbody_rows = tbody.find_elements(By.TAG_NAME, "tr")
            except:
                pass
            
            # If no thead/tbody, get all tr elements directly
            if not thead_rows and not tbody_rows:
                all_rows = table.find_elements(By.TAG_NAME, "tr")
            else:
                all_rows = thead_rows + tbody_rows
            
            if not all_rows:
                continue
            
            # Parse each row
            table_data = []
            
            for row_idx, row in enumerate(all_rows):
                # Get all cells (th and td)
                cells = row.find_elements(By.CSS_SELECTOR, "th, td")
                
                row_data = {
                    "row_index": row_idx,
                    "cells": []
                }
                
                for cell_idx, cell in enumerate(cells):
                    # Try multiple ways to get cell content
                    cell_text = ""
                    
                    # Method 1: Try span.bt-content
                    try:
                        span = cell.find_element(By.CSS_SELECTOR, "span.bt-content")
                        cell_text = span.text.strip()
                    except:
                        pass
                    
                    # Method 2: If no span, get cell text directly
                    if not cell_text:
                        cell_text = cell.text.strip()
                    
                    # Method 3: If still no text, try innerHTML
                    if not cell_text:
                        cell_text = cell.get_attribute("innerHTML").strip()
                    
                    row_data["cells"].append({
                        "index": cell_idx,
                        "tag": cell.tag_name,
                        "text": cell_text,
                        "colspan": cell.get_attribute("colspan"),
                        "rowspan": cell.get_attribute("rowspan")
                    })
                
                table_data.append(row_data)
            
            # Now process the data into a usable format
            if len(table_data) >= 2:
                # Check if first row is headers
                first_row_cells = table_data[0]["cells"]
                second_row_cells = table_data[1]["cells"]
                
                # If first row has th tags and second row has td tags, it's a header-value table
                first_is_headers = all(c["tag"] == "th" for c in first_row_cells)
                second_is_data = all(c["tag"] == "td" for c in second_row_cells)
                
                if first_is_headers and second_is_data and len(first_row_cells) == len(second_row_cells):
                    # Two-row header-value structure
                    headers = [c["text"] for c in first_row_cells]
                    values = [c["text"] for c in second_row_cells]
                    
                    result = dict(zip(headers, values))
                    scraped_data[caption] = [result]
                    
                elif first_is_headers and len(table_data) > 1:
                    # Multi-row data table
                    headers = [c["text"] for c in first_row_cells]
                    rows_list = []
                    
                    for row_data in table_data[1:]:
                        values = [c["text"] for c in row_data["cells"]]
                        if len(headers) == len(values):
                            row_dict = dict(zip(headers, values))
                        else:
                            row_dict = {f"col_{i}": v for i, v in enumerate(values)}
                        rows_list.append(row_dict)
                    
                    scraped_data[caption] = rows_list
                else:
                    # Unknown structure, save raw data
                    rows_list = []
                    for row_data in table_data:
                        row_dict = {f"col_{i}": c["text"] for i, c in enumerate(row_data["cells"])}
                        rows_list.append(row_dict)
                    scraped_data[caption] = rows_list
            else:
                # Single row or empty table
                rows_list = []
                for row_data in table_data:
                    row_dict = {f"col_{i}": c["text"] for i, c in enumerate(row_data["cells"])}
                    rows_list.append(row_dict)
                scraped_data[caption] = rows_list


        # Extract Petitioner and Respondent information
        try:
            petitioner_div = driver.find_element(By.CSS_SELECTOR, "div.border.box.bg-white div.Petitioner")
            petitioner = [li.text.strip() for li in petitioner_div.find_elements(By.TAG_NAME, "li")]
            scraped_data["Petitioner and Advocate"] = petitioner
        except:
            scraped_data["Petitioner and Advocate"] = []


        try:
            respondent_div = driver.find_element(By.CSS_SELECTOR, "div.border.box.bg-white div.respondent")
            respondent = [li.text.strip() for li in respondent_div.find_elements(By.TAG_NAME, "li")]
            scraped_data["Respondent and Advocate"] = respondent
        except:
            scraped_data["Respondent and Advocate"] = []


        if "scraped_case_details" not in session:
            session["scraped_case_details"] = {}
        session["scraped_case_details"][data_cno] = scraped_data


        return jsonify(scraped_data)


    except Exception:
        err_trace = traceback.format_exc()
        return jsonify({"error": f"Failed to scrape case details: {err_trace}"}), 500



@app.route("/download_results_pdf", methods=["GET"])
def download_results_pdf():
    session_id = request.args.get("session_id")
    session = sessions.get(session_id)
    if not session or "scraped_results" not in session:
        return jsonify({"error": "No results to download"}), 400


    try:
        # Combine all results
        combined_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .result-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; page-break-inside: avoid; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Court Case Search Results</h1>
            <p>Session ID: {session_id}</p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        for idx, result in enumerate(session["scraped_results"], 1):
            combined_html += f'<div class="result-section"><h2>Result {idx}</h2>{result}</div>'
        
        combined_html += "</body></html>"


        # Convert to PDF using WeasyPrint
        pdf = HTML(string=combined_html).write_pdf()


        filename = f"court_results_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )


    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500



@app.route("/download_history_pdf", methods=["GET"])
def download_history_pdf():
    session_id = request.args.get("session_id")
    history = load_history()
    
    # Find the history entry
    history_entry = next((h for h in history if h["session_id"] == session_id), None)
    
    if not history_entry:
        return jsonify({"error": "History not found"}), 404
    
    # Get results if session still exists
    session = sessions.get(session_id)
    results = session.get("scraped_results", []) if session else []
    
    if not results:
        return jsonify({"error": "No results available for this search"}), 400


    try:
        combined_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .meta-info {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .result-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; page-break-inside: avoid; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Court Case Search Results</h1>
            <div class="meta-info">
                <p><strong>Court:</strong> {history_entry.get('court_name', 'N/A')}</p>
                <p><strong>Case:</strong> {history_entry.get('reg_no', 'N/A')}/{history_entry.get('reg_year', 'N/A')}</p>
                <p><strong>Type:</strong> {history_entry.get('case_type', 'N/A')}</p>
                <p><strong>Search Date:</strong> {history_entry.get('timestamp', 'N/A')}</p>
                <p><strong>Results Count:</strong> {len(results)}</p>
            </div>
        """
        
        for idx, result in enumerate(results, 1):
            combined_html += f'<div class="result-section"><h2>Result {idx}</h2>{result}</div>'
        
        combined_html += "</body></html>"


        # Convert to PDF using WeasyPrint
        pdf = HTML(string=combined_html).write_pdf()


        filename = f"court_search_{history_entry.get('reg_no', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )


    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500



@app.route("/download_case_detail_pdf", methods=["GET"])
def download_case_detail_pdf():
    session_id = request.args.get("session_id")
    data_cno = request.args.get("data_cno")
    session = sessions.get(session_id)
    
    if not session or "scraped_case_details" not in session or data_cno not in session["scraped_case_details"]:
        return jsonify({"error": "No case details to download"}), 400


    try:
        details = session["scraped_case_details"][data_cno]
        
        # Build HTML for case details
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; page-break-inside: avoid; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ padding: 5px 0; border-bottom: 1px solid #eee; }}
            </style>
        </head>
        <body>
            <h1>Case Details Report</h1>
            <p><strong>CNO:</strong> {data_cno}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        for section_name, section_data in details.items():
            html += f"<h2>{section_name}</h2>"
            
            if isinstance(section_data, list):
                if len(section_data) > 0 and isinstance(section_data[0], dict):
                    # Table format
                    html += "<table><thead><tr>"
                    for key in section_data[0].keys():
                        html += f"<th>{key}</th>"
                    html += "</tr></thead><tbody>"
                    
                    for row in section_data:
                        html += "<tr>"
                        for value in row.values():
                            html += f"<td>{value}</td>"
                        html += "</tr>"
                    html += "</tbody></table>"
                else:
                    # List format
                    html += "<ul>"
                    for item in section_data:
                        html += f"<li>{item}</li>"
                    html += "</ul>"
        
        html += "</body></html>"


        # Convert to PDF using WeasyPrint
        pdf = HTML(string=html).write_pdf()


        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"case_details_{data_cno}_{timestamp}.pdf"
        
        # Save PDF locally to the storage directory
        pdf_path = os.path.join(PDF_STORAGE_DIR, filename)
        with open(pdf_path, 'wb') as f:
            f.write(pdf)
        
        # Update history with the saved PDF filename
        update_history_with_pdf(session_id, data_cno, filename)
        
        print(f"PDF saved locally: {pdf_path}")
        
        return Response(
            pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )


    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500



@app.route("/download_saved_pdf", methods=["GET"])
def download_saved_pdf():
    """Serve a previously saved PDF file"""
    filename = request.args.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    # Security check: ensure filename doesn't contain path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    
    pdf_path = os.path.join(PDF_STORAGE_DIR, filename)
    
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF file not found"}), 404
    
    try:
        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": f"Failed to serve PDF: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)
