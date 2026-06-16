import http.server
import socketserver
import json
import os
import urllib.parse
import pandas as pd
import mimetypes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

PORT = 5000

class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_index()
        elif parsed_path.path == '/api/dashboard':
            self.serve_api()
        elif parsed_path.path.startswith('/static/'):
            self.serve_static(parsed_path.path)
        else:
            self.send_error(404, "Not Found")
            
    def serve_index(self):
        index_path = os.path.join(TEMPLATES_DIR, "index.html")
        if os.path.exists(index_path):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open(index_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "Index not found")
            
    def serve_static(self, path):
        filepath = os.path.join(BASE_DIR, path.lstrip('/'))
        if os.path.exists(filepath) and os.path.isfile(filepath):
            self.send_response(200)
            mimetype, _ = mimetypes.guess_type(filepath)
            self.send_header('Content-type', mimetype or 'application/octet-stream')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")

    def serve_api(self):
        try:
            csv_path = os.path.join(OUTPUTS_DIR, "team_submission.csv")
            json_path = os.path.join(OUTPUTS_DIR, "explainability_logs.json")

            if not os.path.exists(csv_path) or not os.path.exists(json_path):
                response_data = {
                    "table": [],
                    "logs": {},
                    "summary": {"processed": "50,000+", "top_candidates": 0, "avg_tech": 0, "avg_founding": 0, "avg_hiring": 0}
                }
            else:
                df = pd.read_csv(csv_path)
                with open(json_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)

                # Calculate metrics dynamically
                if logs:
                    avg_tech = sum(log["scores"]["technical_fit"] for log in logs.values()) / len(logs)
                    avg_founding = sum(log["scores"]["founding_fit"] for log in logs.values()) / len(logs)
                    avg_hiring = sum(log["scores"]["hiring_probability"] for log in logs.values()) / len(logs)
                else:
                    avg_tech = avg_founding = avg_hiring = 0

                response_data = {
                    "table": df.to_dict(orient="records"),
                    "logs": logs,
                    "summary": {
                        "processed": "50,000+",
                        "top_candidates": len(df),
                        "avg_tech": round(avg_tech * 100, 1),
                        "avg_founding": round(avg_founding * 100, 1),
                        "avg_hiring": round(avg_hiring * 100, 1)
                    }
                }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), DashboardRequestHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()