"""Serve the grading page AND write every verdict straight to disk.
   POST /verdict  -> appends one JSON line to verdicts_<grader>.jsonl
   Run:  python serve.py      then open http://localhost:8765/grade.html
"""
import json, os, http.server, socketserver
PORT = 8765
class H(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/verdict": self.send_error(404); return
        n = int(self.headers.get("Content-Length", 0))
        v = json.loads(self.rfile.read(n))
        fn = f"verdicts_{v.get('grader','unknown')}.jsonl"
        with open(fn, "a") as f: f.write(json.dumps(v) + "\n")
        self.send_response(204); self.end_headers()
    def log_message(self, *a): pass          # quiet
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with socketserver.TCPServer(("127.0.0.1", PORT), H) as s:
    print(f"grading server on http://localhost:{PORT}/grade.html  — verdicts append to verdicts_<grader>.jsonl")
    s.serve_forever()
