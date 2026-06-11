/**
 * Fitify ML — Frontend Application
 * Handles video upload, API communication, and results rendering.
 */

// ============================================================
// Configuration
// ============================================================

const API_BASE = window.location.origin;
const API_ENDPOINTS = {
    analyze: `${API_BASE}/api/analyze`,
    exercises: `${API_BASE}/api/exercises`,
    health: `${API_BASE}/api/health`,
};

// ============================================================
// DOM Elements
// ============================================================

const elements = {
    apiStatus: document.getElementById('api-status'),
    dropzone: document.getElementById('dropzone'),
    videoInput: document.getElementById('video-input'),
    dropzoneContent: document.getElementById('dropzone-content'),
    videoPreview: document.getElementById('video-preview'),
    previewPlayer: document.getElementById('preview-player'),
    previewInfo: document.getElementById('preview-info'),
    clearVideo: document.getElementById('clear-video'),
    analyzeBtn: document.getElementById('analyze-btn'),
    uploadArea: document.getElementById('upload-area'),
    loading: document.getElementById('loading'),
    loadingText: document.getElementById('loading-text'),
    results: document.getElementById('results'),
    scoreRingFill: document.getElementById('score-ring-fill'),
    scoreValue: document.getElementById('score-value'),
    scoreGrade: document.getElementById('score-grade'),
    resultsTitle: document.getElementById('results-title'),
    resultsSubtitle: document.getElementById('results-subtitle'),
    exerciseName: document.getElementById('exercise-name'),
    exerciseConfidence: document.getElementById('exercise-confidence'),
    repCount: document.getElementById('rep-count'),
    videoDuration: document.getElementById('video-duration'),
    feedbackGrid: document.getElementById('feedback-grid'),
    timelineChart: document.getElementById('timeline-chart'),
    timelineSection: document.getElementById('timeline-section'),
    metaGrid: document.getElementById('meta-grid'),
    analyzeAnother: document.getElementById('analyze-another'),
};

let selectedFile = null;

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    setupDragDrop();
    setupFileInput();
    setupAnalyzeButton();
    setupAnalyzeAnother();
    injectSVGDefs();
});

// ============================================================
// API Health Check
// ============================================================

async function checkApiHealth() {
    const statusEl = elements.apiStatus;
    const textEl = statusEl.querySelector('.status-text');

    try {
        const res = await fetch(API_ENDPOINTS.health, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            statusEl.className = 'nav-status online';
            textEl.textContent = 'API Online';
        } else {
            statusEl.className = 'nav-status offline';
            textEl.textContent = 'API Error';
        }
    } catch {
        statusEl.className = 'nav-status offline';
        textEl.textContent = 'API Offline';
    }
}

// ============================================================
// SVG Gradient Definitions (for score ring)
// ============================================================

function injectSVGDefs() {
    const svg = document.querySelector('.score-ring');
    if (!svg) return;

    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
        <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#6C63FF"/>
            <stop offset="100%" style="stop-color:#00D4AA"/>
        </linearGradient>
    `;
    svg.insertBefore(defs, svg.firstChild);
}

// ============================================================
// Drag & Drop
// ============================================================

function setupDragDrop() {
    const dropzone = elements.dropzone;

    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            handleFileSelect(files[0]);
        } else {
            showToast('Please drop a video file (MP4, AVI, MOV, etc.)', 'error');
        }
    });

    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('.btn') || e.target.closest('video')) return;
        elements.videoInput.click();
    });
}

// ============================================================
// File Input
// ============================================================

function setupFileInput() {
    elements.videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    elements.clearVideo.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });
}

function handleFileSelect(file) {
    const maxSize = 100 * 1024 * 1024; // 100MB

    if (!file.type.startsWith('video/')) {
        showToast('Please select a video file.', 'error');
        return;
    }

    if (file.size > maxSize) {
        showToast(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max: 100 MB.`, 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const url = URL.createObjectURL(file);
    elements.previewPlayer.src = url;
    elements.previewInfo.textContent = `${file.name} • ${(file.size / 1024 / 1024).toFixed(1)} MB`;

    elements.dropzoneContent.style.display = 'none';
    elements.videoPreview.style.display = 'flex';
    elements.analyzeBtn.disabled = false;
}

function clearSelectedFile() {
    selectedFile = null;
    elements.videoInput.value = '';

    if (elements.previewPlayer.src) {
        URL.revokeObjectURL(elements.previewPlayer.src);
        elements.previewPlayer.src = '';
    }

    elements.dropzoneContent.style.display = '';
    elements.videoPreview.style.display = 'none';
    elements.analyzeBtn.disabled = true;
}

// ============================================================
// Analyze Button
// ============================================================

function setupAnalyzeButton() {
    elements.analyzeBtn.addEventListener('click', () => {
        if (selectedFile) {
            analyzeVideo(selectedFile);
        }
    });
}

function setupAnalyzeAnother() {
    elements.analyzeAnother.addEventListener('click', () => {
        elements.results.style.display = 'none';
        elements.uploadArea.style.display = '';
        document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
        clearSelectedFile();
    });
}

// ============================================================
// Video Analysis
// ============================================================

async function analyzeVideo(file) {
    // Show loading state
    elements.uploadArea.style.display = 'none';
    elements.loading.style.display = '';
    elements.results.style.display = 'none';

    // Animate loading steps
    animateLoadingSteps();

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch(API_ENDPOINTS.analyze, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        if (data.error) {
            throw new Error(data.error);
        }

        // Show results
        elements.loading.style.display = 'none';
        renderResults(data);

    } catch (error) {
        elements.loading.style.display = 'none';
        elements.uploadArea.style.display = '';
        showToast(`Analysis failed: ${error.message}`, 'error');
    }
}

function animateLoadingSteps() {
    const steps = ['step-1', 'step-2', 'step-3', 'step-4'];
    const texts = [
        'Classifying exercise type...',
        'Extracting pose landmarks...',
        'Analyzing form quality...',
        'Generating your report...',
    ];

    let current = 0;

    const interval = setInterval(() => {
        if (current > 0) {
            document.getElementById(steps[current - 1]).classList.remove('active');
            document.getElementById(steps[current - 1]).classList.add('done');
        }

        if (current < steps.length) {
            document.getElementById(steps[current]).classList.add('active');
            elements.loadingText.textContent = texts[current];
            current++;
        } else {
            clearInterval(interval);
        }
    }, 1500);

    // Store interval so we can clear on completion
    window._loadingInterval = interval;
}

// ============================================================
// Render Results
// ============================================================

function renderResults(data) {
    if (window._loadingInterval) clearInterval(window._loadingInterval);

    // Reset loading steps
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById(`step-${i}`);
        step.classList.remove('active', 'done');
    }

    elements.results.style.display = '';
    elements.results.scrollIntoView({ behavior: 'smooth' });

    // Exercise info
    const exerciseDisplay = (data.exercise || 'Unknown').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    elements.exerciseName.textContent = exerciseDisplay;
    elements.exerciseConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    elements.repCount.textContent = data.rep_count || '0';
    elements.videoDuration.textContent = data.video_info ? `${data.video_info.duration}s` : '-';

    elements.resultsTitle.textContent = `${exerciseDisplay} Analysis`;
    elements.resultsSubtitle.textContent = data.grade_message || '';

    // Score animation
    animateScore(data.overall_score || 0, data.grade || '-');

    // Feedback cards
    renderFeedbackCards(data.form_feedback || []);

    // Timeline
    renderTimeline(data.timeline || []);

    // Meta cards
    renderMetaCards(data);
}

// ============================================================
// Score Animation
// ============================================================

function animateScore(score, grade) {
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (score / 100) * circumference;

    // Reset
    elements.scoreRingFill.style.strokeDashoffset = circumference;
    elements.scoreValue.textContent = '0';

    // Trigger animation after brief delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            elements.scoreRingFill.style.strokeDashoffset = offset;

            // Animate number
            animateCounter(elements.scoreValue, 0, Math.round(score), 1200);

            // Grade
            const gradeEl = elements.scoreGrade;
            gradeEl.textContent = `Grade: ${grade}`;
            gradeEl.className = `score-grade grade-${grade}`;
        }, 100);
    });
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ============================================================
// Feedback Cards
// ============================================================

function renderFeedbackCards(feedbackList) {
    const grid = elements.feedbackGrid;
    grid.innerHTML = '';

    if (!feedbackList.length) {
        grid.innerHTML = '<p style="color: var(--text-muted); padding: 20px;">No specific form feedback available.</p>';
        return;
    }

    feedbackList.forEach((fb, index) => {
        const statusIcon = fb.status === 'good' ? '✅' : fb.status === 'warning' ? '⚠️' : '❌';
        const scoreClass = fb.status === 'good' ? 'score-good' : fb.status === 'warning' ? 'score-warning' : 'score-error';

        const card = document.createElement('div');
        card.className = `feedback-card status-${fb.status}`;
        card.style.animationDelay = `${index * 0.08}s`;

        card.innerHTML = `
            <div class="feedback-header">
                <span class="feedback-aspect">
                    <span class="feedback-status-icon">${statusIcon}</span>
                    ${fb.aspect}
                </span>
                <span class="feedback-score ${scoreClass}">${Math.round(fb.score)}</span>
            </div>
            <p class="feedback-message">${fb.message}</p>
            <div class="feedback-suggestion">${fb.suggestion}</div>
        `;

        grid.appendChild(card);
    });
}

// ============================================================
// Timeline Chart
// ============================================================

function renderTimeline(timeline) {
    const container = elements.timelineChart;
    const section = elements.timelineSection;

    if (!timeline.length) {
        section.style.display = 'none';
        return;
    }

    section.style.display = '';
    container.innerHTML = '';

    const barsDiv = document.createElement('div');
    barsDiv.className = 'timeline-bars';

    const labelsDiv = document.createElement('div');
    labelsDiv.className = 'timeline-labels';

    const maxScore = 100;

    timeline.forEach((point) => {
        const bar = document.createElement('div');
        bar.className = 'timeline-bar';
        const height = Math.max(5, (point.score / maxScore) * 100);
        bar.style.height = `${height}%`;
        bar.setAttribute('data-score', Math.round(point.score));

        // Color based on score
        if (point.score >= 75) {
            bar.style.background = 'var(--status-good)';
        } else if (point.score >= 50) {
            bar.style.background = 'var(--status-warning)';
        } else {
            bar.style.background = 'var(--status-error)';
        }

        barsDiv.appendChild(bar);

        const label = document.createElement('div');
        label.className = 'timeline-label';
        label.textContent = `${point.timestamp}s`;
        labelsDiv.appendChild(label);
    });

    container.appendChild(barsDiv);
    container.appendChild(labelsDiv);
}

// ============================================================
// Meta Cards
// ============================================================

function renderMetaCards(data) {
    const grid = elements.metaGrid;
    grid.innerHTML = '';

    const meta = data.analysis_meta || {};

    const cards = [
        { label: 'Frames Analyzed', value: meta.frames_analyzed || '-' },
        { label: 'Pose Detection', value: `${meta.pose_detection_rate || 0}%` },
        { label: 'Processing Time', value: `${meta.processing_time_seconds || 0}s` },
        { label: 'Model', value: data.classification?.model_used === 'videomae' ? 'VideoMAE' : 'Fallback' },
    ];

    cards.forEach(card => {
        const el = document.createElement('div');
        el.className = 'meta-card';
        el.innerHTML = `
            <div class="meta-card-value">${card.value}</div>
            <div class="meta-card-label">${card.label}</div>
        `;
        grid.appendChild(el);
    });
}

// ============================================================
// Toast Notifications
// ============================================================

function showToast(message, type = 'error') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast ${type === 'success' ? 'toast-success' : ''}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
