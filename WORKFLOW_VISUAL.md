# 🎯 API WORKFLOW - VISUAL GUIDE

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR FRONTEND                             │
│  (React / Vue / Angular / HTML / Any Framework)                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ 1. Upload CSV
                 │    POST /run-csv/
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR API SERVER                             │
│                   (api_server.py)                                │
│                                                                   │
│  Step 1: Receive CSV file ✅                                     │
│  Step 2: Load problems   ✅                                      │
│  Step 3: Process with reasoning engine ✅                        │
│  Step 4: Generate results ✅                                     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ 2. Returns BOTH:
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌─────────┐              ┌─────────┐
│  JSON   │              │   CSV   │
│ Results │              │  File   │
└─────────┘              └─────────┘
    │                         │
    │ Display in UI           │ Download link
    ▼                         ▼
┌─────────────────────────────────────┐
│  FRONTEND GETS BOTH:                │
│                                     │
│  ✅ JSON for displaying in UI      │
│  ✅ CSV for downloading            │
└─────────────────────────────────────┘
```

---

## 🔄 Flow Example

### 1️⃣ Frontend Uploads CSV
```javascript
const formData = new FormData();
formData.append('file', csvFile);

const response = await fetch('http://localhost:8080/run-csv/', {
  method: 'POST',
  body: formData
});
```

**Input CSV:**
```csv
topic,problem_statement,answer_options
Math,"What is 2+2?","['3','4','5']"
Science,"What is H2O?","['Water','Oxygen','Hydrogen']"
```

### 2️⃣ API Processes
```
Processing problem 1/2 (50%)...
Processing problem 2/2 (100%)...
✅ Complete!
```

### 3️⃣ API Returns JSON
```javascript
const result = await response.json();

// Result contains:
{
  "success": true,
  "total_problems": 2,
  "download_csv": "/download/output_abc123.csv",
  "results": [
    {
      "topic": "Math",
      "problem_statement": "What is 2+2?",
      "solution": "Step 1: Add 2 and 2...",
      "correct_option": "4"
    },
    {
      "topic": "Science", 
      "problem_statement": "What is H2O?",
      "solution": "H2O is water...",
      "correct_option": "Water"
    }
  ]
}
```

### 4️⃣ Frontend Uses Results

**Option A: Display in UI**
```javascript
// Show results in your interface
result.results.forEach(item => {
  displayProblem(item.problem_statement);
  displayAnswer(item.correct_option);
  displayReasoning(item.solution);
});
```

**Option B: Download CSV**
```javascript
// Let user download CSV
window.location.href = `http://localhost:8080${result.download_csv}`;
```

**Option C: Both!**
```javascript
// Display AND provide download option
showResults(result.results);
showDownloadButton(result.download_csv);
```

---

## 📊 Data Format Examples

### Input CSV (What Frontend Sends)
```csv
topic,problem_statement,answer_options
Math,"Calculate 15 + 27","['40', '42', '44']"
Science,"What element is O?","['Oxygen', 'Gold', 'Silver']"
```

### Output JSON (For UI Display)
```json
{
  "total_problems": 2,
  "results": [
    {
      "topic": "Math",
      "problem_statement": "Calculate 15 + 27",
      "correct_option": "42",
      "solution": "Adding 15 and 27 gives 42"
    },
    {
      "topic": "Science",
      "problem_statement": "What element is O?",
      "correct_option": "Oxygen",
      "solution": "O is the symbol for Oxygen"
    }
  ],
  "download_csv": "/download/output_abc123.csv"
}
```

### Output CSV (For Download)
```csv
topic,problem_statement,solution,correct option
Math,"Calculate 15 + 27","Adding 15 and 27 gives 42","42"
Science,"What element is O?","O is the symbol for Oxygen","Oxygen"
```

---

## 🎨 Frontend Implementation Examples

### Example 1: Simple Upload Form
```html
<form id="uploadForm">
  <input type="file" id="csvFile" accept=".csv" required>
  <button type="submit">Process CSV</button>
</form>

<div id="results"></div>

<script>
document.getElementById('uploadForm').onsubmit = async (e) => {
  e.preventDefault();
  
  const file = document.getElementById('csvFile').files[0];
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8080/run-csv/', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  // Show results
  document.getElementById('results').innerHTML = `
    <h3>Processed ${result.total_problems} problems</h3>
    <a href="http://localhost:8080${result.download_csv}">Download Results</a>
  `;
};
</script>
```

### Example 2: React Component
```jsx
function CSVProcessor() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8080/run-csv/', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    setResults(data);
    setLoading(false);
  };
  
  return (
    <div>
      <input type="file" onChange={handleUpload} accept=".csv" />
      
      {loading && <p>Processing...</p>}
      
      {results && (
        <>
          <h3>Results: {results.total_problems} problems</h3>
          
          {results.results.map((r, i) => (
            <div key={i}>
              <strong>Q:</strong> {r.problem_statement}<br/>
              <strong>A:</strong> {r.correct_option}
            </div>
          ))}
          
          <a href={`http://localhost:8080${results.download_csv}`}>
            Download CSV
          </a>
        </>
      )}
    </div>
  );
}
```

### Example 3: With Progress (Async)
```javascript
async function uploadWithProgress(file) {
  // 1. Upload for async processing
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8080/batch-async/', {
    method: 'POST',
    body: formData
  });
  
  const { job_id } = await response.json();
  
  // 2. Connect WebSocket for progress
  const ws = new WebSocket(`ws://localhost:8080/ws/${job_id}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Update progress bar
    updateProgress(data.progress);
    
    // Check if completed
    if (data.status === 'completed') {
      showResults(data.result);
      ws.close();
    }
  };
}
```

---

## ✅ Confirmation Checklist

Your API supports:

- ✅ **Upload CSV** via POST /run-csv/
- ✅ **Automatic Processing** (reasoning engine)
- ✅ **JSON Results** (in response body)
- ✅ **CSV Results** (downloadable file)
- ✅ **CORS Enabled** (frontend integration)
- ✅ **Error Handling** (400, 413, 500 codes)
- ✅ **File Validation** (CSV format, size limits)
- ✅ **Progress Tracking** (async endpoint)
- ✅ **Real-time Updates** (WebSocket)

---

## 🚀 Get Started

1. **Start API:**
   ```bash
   python api_server.py
   ```

2. **Test it:**
   ```bash
   python test_core_functionality.py
   ```

3. **Connect Frontend:**
   - Use examples above
   - Point to: http://localhost:8080
   - Upload CSV, get results!

---

## 📚 Documentation Files

- **FUNCTIONALITY_CONFIRMATION.md** - This file!
- **ENDPOINTS_GUIDE.md** - Detailed API reference
- **ENDPOINTS_QUICK_REF.md** - Quick reference
- **test_core_functionality.py** - Automated tests
- **test_frontend.html** - Visual testing tool

---

## 💡 Pro Tips

1. **Small CSVs (< 100 rows)**: Use `/run-csv/` (sync)
2. **Large CSVs (> 100 rows)**: Use `/batch-async/` (with progress)
3. **Single Problems**: Use `/run-single/` (fastest)
4. **Caching**: Install Redis for 10x speed boost
5. **Testing**: Use `/docs` for interactive testing

---

**Your API is ready for frontend integration! 🎉**
