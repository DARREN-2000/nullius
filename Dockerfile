# Distroless-adjacent, dependency-free image. The application deliberately has no
# third-party runtime requirements, so there is nothing to pip install and no
# lockfile to drift. Runs as a non-root user because a clinical service should
# never need root.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY nullius/ ./nullius/
COPY corpus/ ./corpus/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN useradd --create-home --uid 10001 nullius \
 && mkdir -p /app/out && chown -R nullius:nullius /app
USER nullius

# Fail the build if the safety gates regress.
RUN python3 -W ignore::ResourceWarning -m unittest discover -s tests -q

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"

CMD ["python3", "-m", "nullius.api", "--host", "0.0.0.0", "--port", "8080"]
