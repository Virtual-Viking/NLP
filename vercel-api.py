from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import os
import sys

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Redirect to Streamlit app
    return """
    <html>
        <head>
            <title>Redirecting to Wordsmith SMS Analysis Suite</title>
            <meta http-equiv="refresh" content="0;url=https://wordsmith-sms-analysis.streamlit.app/">
            <script type="text/javascript">
                window.location.href = "https://wordsmith-sms-analysis.streamlit.app/"
            </script>
        </head>
        <body>
            <p>If you are not redirected, <a href="https://wordsmith-sms-analysis.streamlit.app/">click here</a>.</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
