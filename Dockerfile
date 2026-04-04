# ── Email Triage Environment ─────────────────────────────────────────────────
# HF Spaces (Docker SDK) — port 7860
# Standalone: docker build -t email-triage-env .
#             docker run -p 7860:7860 email-triage-env
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--log-level", "info"]