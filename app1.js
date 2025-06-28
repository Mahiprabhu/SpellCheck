document.addEventListener('DOMContentLoaded', function() {
    const checkBtn = document.getElementById('check-btn');
    const fileUpload = document.getElementById('file-upload');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    // Model descriptions
    // Update model descriptions to set proper expectations
    const modelDescriptions = {
        'bert': 'Transformer model (best for context-aware corrections)',
        'lstm': 'Neural network (better for technical term patterns)',
        'ensemble': 'Combines BERT and LSTM (balanced approach)'
    };
    
    // Domain descriptions
    const domainDescriptions = {
        'physics': 'Physics terminology checker (quantum mechanics, relativity, etc.)',
        'medical': 'Medical terminology checker (anatomy, diseases, medications)',
        'legal': 'Legal terminology checker (contracts, statutes, case law)'
    };
    
    checkBtn.addEventListener('click', processText);
    fileUpload.addEventListener('change', handleFileUpload);
    
    async function processText() {
        const text = document.getElementById('input-text').value.trim();
        const domain = document.getElementById('domain').value;
        const model = document.getElementById('model').value;
        
        if (!text) {
            alert('Please enter some text');
            return;
        }
        
        checkBtn.disabled = true;
        checkBtn.textContent = 'Processing...';
        progressContainer.style.display = 'none';
        
        try {
            // Show progress for large texts
            if (text.length > 1000) {
                progressContainer.style.display = 'flex';
                simulateProgress();
            }
            
            const response = await fetch('/api/spellcheck', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ 
                    text, 
                    domain,
                    model 
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }
            
            displayResults(data, domain, model);
        } catch (error) {
            console.error('Error:', error);
            alert(`Spellcheck failed: ${error.message}`);
        } finally {
            checkBtn.disabled = false;
            checkBtn.textContent = 'Check Spelling';
            progressContainer.style.display = 'none';
        }
    }
    
    function simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
            }
            progressBar.value = progress;
            progressText.textContent = `${Math.round(progress)}%`;
        }, 200);
        
        // Return a function to stop the simulation if needed
        return () => clearInterval(interval);
    }
    
    function displayResults(data, domain, model) {
        const output = document.getElementById('output');
        output.innerHTML = '';
        
        // Debug output
        console.log("Raw API response:", data);
    
        // Create results header
        output.innerHTML = `
            <div class="results-header">
                <h2>Results for ${domain} domain</h2>
                <div class="model-info">
                    Using ${model} model: ${modelDescriptions[model] || ''}
                </div>
            </div>
        `;

        if (model !== 'bert') {
            output.innerHTML += `
                <div class="model-note">
                    Note: ${model.toUpperCase()} may be less accurate than BERT for some words.
                    <br>For best results, try the BERT model.
                </div>
            `;
        }
    
        // Process and display errors
        if (data.general_errors?.length > 0 || data.domain_errors?.length > 0) {
            // Display general errors if any exist
            if (data.general_errors?.length > 0) {
                const generalDiv = document.createElement('div');
                generalDiv.className = 'error-section';
                generalDiv.innerHTML = '<h3>General Spelling Errors</h3>';
                
                const grouped = groupBySentence(data.general_errors);
                for (const [context, errors] of Object.entries(grouped)) {
                    generalDiv.innerHTML += `
                        <div class="sentence-group general">
                            <p class="context">"${context}"</p>
                            <div class="errors">
                                ${errors.map(error => `
                                    <div class="error-item">
                                        <span class="original">${error.original}</span> → 
                                        <strong>${error.correction}</strong>
                                        <span class="confidence">[${error.confidence}%]</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }
                output.appendChild(generalDiv);
            }
    
            // Display domain errors if any exist
            if (data.domain_errors?.length > 0) {
                const domainDiv = document.createElement('div');
                domainDiv.className = 'error-section';
                domainDiv.innerHTML = `<h3>${capitalizeFirstLetter(domain)} Term Suggestions</h3>`;
                
                const grouped = groupBySentence(data.domain_errors);
                for (const [context, errors] of Object.entries(grouped)) {
                    domainDiv.innerHTML += `
                        <div class="sentence-group domain">
                            <p class="context">"${context}"</p>
                            <div class="errors">
                                ${errors.map(error => `
                                    <div class="error-item">
                                        <span class="original">${error.original}</span> → 
                                        <strong>${error.correction}</strong>
                                        <span class="confidence">[${error.confidence}%]</span>
                                        ${error.ontology?.length ? `
                                        <div class="ontology">
                                            Related: ${error.ontology.map(x => 
                                                `${x.term} (${x.relation})`).join(', ')}
                                        </div>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }
                output.appendChild(domainDiv);
            }
        } else {
            output.innerHTML += `
                <div class="success">
                    No spelling errors detected! ✅
                    <div class="subtext">The text appears to be correctly spelled for the ${domain} domain.</div>
                </div>
            `;
        }
    }
    
    function showDebugInfo() {
        const debug = document.getElementById('debug-info');
        debug.style.display = debug.style.display === 'none' ? 'block' : 'none';
    }
    
    function groupBySentence(errors) {
        return errors.reduce((groups, error) => {
            if (!groups[error.context]) groups[error.context] = [];
            groups[error.context].push(error);
            return groups;
        }, {});
    }
    
    function createErrorGroup(context, errors, errorType) {
        return `
            <div class="sentence-group ${errorType}">
                <p class="context">"${context}"</p>
                <div class="errors">
                    ${errors.map(error => createErrorItem(error, errorType)).join('')}
                </div>
            </div>
        `;
    }
    
    function createErrorItem(error, errorType) {
        const ontologyInfo = error.ontology?.length ? `
            <div class="ontology">
                <strong>Related terms:</strong>
                ${error.ontology.map(item => 
                    `<span class="ontology-item">${item.term} (${item.relation})</span>`
                ).join(', ')}
            </div>
        ` : '';
        
        return `
            <div class="error-item">
                <div class="error-main">
                    <span class="original">${error.original}</span> → 
                    <strong>${error.correction}</strong>
                    <span class="confidence">[confidence: ${error.confidence}%]</span>
                    <span class="error-type ${errorType}-error">${errorType}</span>
                </div>
                ${ontologyInfo}
                <div class="feedback-buttons" data-original="${escapeHtml(error.original)}" data-correction="${escapeHtml(error.correction)}">
                    <button class="feedback-btn correct">✅ Correct</button>
                    <button class="feedback-btn incorrect">❌ Incorrect</button>
                </div>
            </div>
        `;
    }
    
    function addFeedbackButtons() {
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', async function() {
                const buttonsDiv = this.parentElement;
                const original = buttonsDiv.getAttribute('data-original');
                const correction = buttonsDiv.getAttribute('data-correction');
                const isCorrect = this.classList.contains('correct');
                
                try {
                    const response = await fetch('/api/feedback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            original,
                            correction,
                            is_correct: isCorrect
                        })
                    });
                    
                    if (response.ok) {
                        this.textContent = isCorrect ? 'Thanks!' : 'Noted';
                        this.style.backgroundColor = isCorrect ? '#4CAF50' : '#f44336';
                        this.nextElementSibling?.remove();
                    }
                } catch (error) {
                    console.error('Feedback error:', error);
                }
            });
        });
    }
    
    async function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const domain = document.getElementById('domain').value;
        const model = document.getElementById('model').value;
        
        checkBtn.disabled = true;
        checkBtn.textContent = 'Processing File...';
        progressContainer.style.display = 'flex';
        progressBar.value = 0;
        progressText.textContent = '0%';
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('domain', domain);
            formData.append('model', model);
            
            const response = await fetch('/api/spellcheck-file', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }
            
            displayResults(data, domain, model);
            document.getElementById('input-text').value = `File: ${file.name}`;
        } catch (error) {
            console.error('Error:', error);
            alert(`File processing failed: ${error.message}`);
        } finally {
            checkBtn.disabled = false;
            checkBtn.textContent = 'Check Spelling';
            progressContainer.style.display = 'none';
        }
    }
    
    // Helper functions
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});