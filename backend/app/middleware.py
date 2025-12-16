# LLM Assistant - Middleware
# Created by Gustav Christensen
# Date: December 2025
# Description: Quart middleware for logging, error handling, and request processing

import time
import logging
import traceback
from quart import Request, Response, jsonify, g
from werkzeug.exceptions import HTTPException

try:
    from .apimonitor import monitor
except ImportError:
    import apimonitor
    monitor = apimonitor.monitor

log = logging.getLogger(__name__)

def setup_middleware(app):
    
    @app.before_request
    async def before_request():
        # Attach start time to global request context
        g.start_time = time.time()

    @app.after_request
    async def after_request(response: Response):
        from quart import request
        # Check if start_time is set in g
        start_time = getattr(g, 'start_time', None)
        if start_time:
            duration_ms = (time.time() - start_time) * 1000
            status_code = response.status_code
            
            # Log basic info
            log.info(f"{request.method} {request.path} {status_code} - {duration_ms:.2f}ms")
            
            # Record metrics
            is_success = 200 <= status_code < 400
            monitor.record_request(latency_ms=duration_ms, success=is_success)
            
        return response

    @app.errorhandler(Exception)
    async def handle_exception(e):
        # Pass through HTTP exceptions
        if isinstance(e, HTTPException):
            return e
            
        # Log unexpected errors
        log.error(f"Unhandled exception: {str(e)}")
        log.debug(traceback.format_exc())
        
        # Record as failed request in monitor (latency might be skewed here but we count the error)
        # Note: after_request might not run if error handler creates response directly without bubbling? 
        # Actually in Quart error handlers return response, after_request runs on that.
        
        return jsonify({
            "error": "Internal Server Error", 
            "details": str(e),
            "type": type(e).__name__
        }), 500

    @app.errorhandler(404)
    async def not_found(e):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(401)
    async def unauthorized(e):
        return jsonify({"error": "Unauthorized"}), 401
