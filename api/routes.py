"""
API routes for the Market Research SMOL Agents system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
import json

from ..pipelines.collection_pipeline import run_collection_pipeline
from ..pipelines.analysis_pipeline import run_analysis_pipeline
from ..agents.report_agent import ReportAgent
from ..config.settings import DATA_DIR, REPORT_DIR

# Initialize FastAPI app
app = FastAPI(
    title="Market Research SMOL Agents API",
    description="API for running market research analysis with SMOL agents",
    version="1.0.0"
)

logger = logging.getLogger("api.routes")

# Models
class ResearchRequest(BaseModel):
    market_domain: str
    search_terms: List[str]
    include_visualizations: bool = True
    output_format: str = "markdown"

class AnalysisRequest(BaseModel):
    market_domain: str
    data_files: List[str]
    analysis_types: Optional[List[str]] = None

class ReportRequest(BaseModel):
    market_domain: str
    analysis_files: List[str]
    output_format: str = "markdown"
    include_visualizations: bool = True

class ResearchStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    message: Optional[str] = None

# In-memory task storage
tasks = {}

# Background task handler
def run_research_task(task_id: str, request: ResearchRequest):
    """Run a complete research task in the background"""
    try:
        # Update task status
        tasks[task_id] = {
            "status": "collecting",
            "progress": 0.1,
            "message": "Collecting data..."
        }
        
        # Run data collection
        collection_results = run_collection_pipeline(
            market_domain=request.market_domain,
            search_terms=request.search_terms
        )
        
        if collection_results.get("status") != "success":
            tasks[task_id] = {
                "status": "failed",
                "progress": 0.0,
                "message": "Data collection failed"
            }
            return
        
        # Get data files
        data_files = [file for source in collection_results.get("sources", {}).values() 
                    for file in [source.get("data_file")] if file]
        
        # Update task status
        tasks[task_id] = {
            "status": "analyzing",
            "progress": 0.4,
            "message": "Analyzing data..."
        }
        
        # Run analysis
        analysis_results = run_analysis_pipeline(
            market_domain=request.market_domain,
            data_files=data_files
        )
        
        # Get analysis files
        analysis_files = []
        for analysis_type, results in analysis_results.get("analyses", {}).items():
            if isinstance(results, dict) and "analysis" in results:
                for analysis_name, analysis_data in results["analysis"].items():
                    if "output_file" in analysis_data:
                        analysis_files.append(analysis_data["output_file"])
            elif isinstance(results, dict) and "sentiment_analysis" in results:
                if "output_file" in results["sentiment_analysis"]:
                    analysis_files.append(results["sentiment_analysis"]["output_file"])
        
        # Update task status
        tasks[task_id] = {
            "status": "reporting",
            "progress": 0.7,
            "message": "Generating report..."
        }
        
        # Generate report
        report_agent = ReportAgent()
        report_results = report_agent.act({
            "analysis_files": analysis_files,
            "market_domain": request.market_domain,
            "format": request.output_format,
            "include_sections": {
                "visualizations": request.include_visualizations
            }
        })
        
        # Check if report was generated
        if report_results.get("status") != "success":
            tasks[task_id] = {
                "status": "failed",
                "progress": 0.0,
                "message": "Report generation failed"
            }
            return
        
        # Get report file
        report_file = report_results.get("report", {}).get("file")
        
        # Update task status
        tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Research completed",
            "report_file": report_file,
            "analysis_files": analysis_files,
            "data_files": data_files
        }
        
    except Exception as e:
        logger.error(f"Error in research task: {str(e)}")
        tasks[task_id] = {
            "status": "failed",
            "progress": 0.0,
            "message": f"Error: {str(e)}"
        }

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Market Research SMOL Agents API"}

@app.post("/research", response_model=Dict[str, str])
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new market research task"""
    task_id = f"task_{len(tasks) + 1}"
    
    # Initialize task
    tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Task queued"
    }
    
    # Add task to background
    background_tasks.add_task(run_research_task, task_id, request)
    
    return {"task_id": task_id}

@app.get("/research/{task_id}/status", response_model=ResearchStatus)
async def get_research_status(task_id: str):
    """Get the status of a research task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0.0),
        "message": task.get("message")
    }

@app.get("/research/{task_id}/report")
async def get_research_report(task_id: str):
    """Get the report from a completed research task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Research not completed yet")
    
    report_file = task.get("report_file")
    if not report_file or not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    # Read the report file
    with open(report_file, "r") as f:
        content = f.read()
    
    # Determine content type
    content_type = "text/markdown" if report_file.endswith(".md") else "text/html"
    
    return {"content": content, "content_type": content_type}

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_data(request: AnalysisRequest):
    """Run analysis on existing data files"""
    # Verify that data files exist
    for file in request.data_files:
        if not os.path.exists(file):
            raise HTTPException(status_code=404, detail=f"Data file not found: {file}")
    
    # Run analysis pipeline
    analysis_results = run_analysis_pipeline(
        market_domain=request.market_domain,
        data_files=request.data_files
    )
    
    return analysis_results

@app.post("/report", response_model=Dict[str, Any])
async def generate_report(request: ReportRequest):
    """Generate a report from existing analysis files"""
    # Verify that analysis files exist
    for file in request.analysis_files:
        if not os.path.exists(file):
            raise HTTPException(status_code=404, detail=f"Analysis file not found: {file}")
    
    # Initialize report agent
    report_agent = ReportAgent()
    
    # Generate report
    report_results = report_agent.act({
        "analysis_files": request.analysis_files,
        "market_domain": request.market_domain,
        "format": request.output_format,
        "include_sections": {
            "visualizations": request.include_visualizations
        }
    })
    
    return report_results

@app.get("/files/data", response_model=List[Dict[str, Any]])
async def list_data_files(limit: int = Query(10, gt=0, le=100)):
    """List available data files"""
    data_path = DATA_DIR / "raw"
    
    if not data_path.exists():
        return []
    
    files = []
    for file in data_path.glob("**/*.json"):
        if len(files) >= limit:
            break
            
        # Get basic file info
        stat = file.stat()
        files.append({
            "path": str(file),
            "name": file.name,
            "size": stat.st_size,
            "created": stat.st_ctime
        })
    
    return files

@app.get("/files/reports", response_model=List[Dict[str, Any]])
async def list_report_files(limit: int = Query(10, gt=0, le=100)):
    """List available report files"""
    if not REPORT_DIR.exists():
        return []
    
    files = []
    for file in REPORT_DIR.glob("**/*.*"):
        if len(files) >= limit:
            break
            
        # Get basic file info
        stat = file.stat()
        files.append({
            "path": str(file),
            "name": file.name,
            "format": file.suffix.lstrip('.'),
            "size": stat.st_size,
            "created": stat.st_ctime
        })
    
    return files