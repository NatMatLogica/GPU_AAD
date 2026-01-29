#!/usr/bin/env python3
"""
Simple HTTP server for the SIMM optimization demo.

This serves the HTML page and handles optimization API requests.

Usage:
    python serve_demo.py [--port 8080]

Then open http://localhost:8080/optimization_demo.html in your browser.
"""

import http.server
import socketserver
import json
import urllib.parse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

PORT = 8080


class OptimizationHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with optimization API endpoint."""

    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/run_optimization':
            self.handle_optimization(parsed.query)
        elif parsed.path == '/optimization_demo.html' or parsed.path == '/':
            # Serve the main visualization page
            self.path = '/visualization/index.html'
            super().do_GET()
        elif parsed.path.startswith('/data/'):
            # Serve data files directly from project root
            super().do_GET()
        else:
            # Serve static files
            super().do_GET()

    def handle_optimization(self, query_string):
        """Handle optimization API request."""
        try:
            params = urllib.parse.parse_qs(query_string)
            num_trades = int(params.get('trades', [10])[0])
            num_portfolios = int(params.get('portfolios', [3])[0])
            trade_types = params.get('types', ['ir_swap'])[0]
            avg_maturity = float(params.get('avgMaturity', [5.0])[0])
            maturity_spread = float(params.get('maturitySpread', [1.0])[0])

            # No hard limit - UI handles large datasets with summary view
            num_trades = min(num_trades, 500)  # Soft limit for server performance

            print(f"Running optimization: {num_trades} trades, {num_portfolios} portfolios, {trade_types}, maturity={avg_maturity}±{maturity_spread}Y")

            # Import and run optimization
            from run_optimization_demo import run_optimization_for_demo
            result = run_optimization_for_demo(num_trades, num_portfolios, trade_types, avg_maturity, maturity_spread)

            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            print(f"Optimization error: {e}")
            import traceback
            traceback.print_exc()

            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def get_local_ip():
    """Get the local IP address for LAN access."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def main():
    port = PORT
    host = "0.0.0.0"  # Bind to all interfaces for LAN access

    if len(sys.argv) > 1:
        if sys.argv[1] == '--port' and len(sys.argv) > 2:
            port = int(sys.argv[2])
        else:
            try:
                port = int(sys.argv[1])
            except ValueError:
                pass

    local_ip = get_local_ip()

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer((host, port), OptimizationHandler) as httpd:
        print(f"""
================================================================================
    SIMM Optimization Demo Server
================================================================================

    Local access:      http://localhost:{port}/optimization_demo.html

    LAN access:        http://{local_ip}:{port}/optimization_demo.html
                       (use this URL from other devices on your WiFi)

    Press Ctrl+C to stop
================================================================================
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
