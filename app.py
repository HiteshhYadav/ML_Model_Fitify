"""
Fitify ML — Flask Web Application
REST API for exercise video analysis with a premium web UI.
"""

import os
import sys
import uuid
import time
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Configuration
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Lazy-load analyzer to speed up startup
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from analysis.analyzer import ExerciseAnalyzer
        _analyzer = ExerciseAnalyzer()
    return _analyzer


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Serve the main web UI."""
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Fitify ML",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/exercises", methods=["GET"])
def list_exercises():
    """List all supported exercises."""
    analyzer = get_analyzer()
    exercises = analyzer.classifier.get_supported_exercises()

    # Format exercise names for display
    formatted = []
    for ex in exercises:
        formatted.append({
            "id": ex,
            "name": ex.replace("_", " ").title(),
        })

    return jsonify({
        "exercises": formatted,
        "count": len(formatted),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    Upload and analyze a workout video.
    Returns a comprehensive form analysis report.
    """
    # Check for video file
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Use 'video' form field."}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(video_file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save uploaded file
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        video_file.save(save_path)

        # Check file size
        file_size = os.path.getsize(save_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(save_path)
            return jsonify({
                "error": f"File too large ({file_size / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
            }), 413

        if file_size == 0:
            os.remove(save_path)
            return jsonify({"error": "Uploaded file is empty."}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Run analysis
    try:
        analyzer = get_analyzer()
        report = analyzer.analyze(save_path)

        # Add file metadata
        report["file_info"] = {
            "original_name": video_file.filename,
            "size_mb": round(file_size / 1024 / 1024, 2),
            "analyzed_at": datetime.now().isoformat(),
        }

        return jsonify(report)

    except Exception as e:
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "details": "Please ensure the video contains a person performing an exercise."
        }), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print()
    print("=" * 60)
    print("  Fitify ML - AI Exercise Analyzer")
    print("=" * 60)
    print()
    print("  Starting server...")
    print("  Open http://localhost:5000 in your browser")
    print()
    print("  API Endpoints:")
    print("    POST /api/analyze    - Analyze a workout video")
    print("    GET  /api/exercises  - List supported exercises")
    print("    GET  /api/health     - Health check")
    print()
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
