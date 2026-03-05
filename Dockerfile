# ─────────────────────────────────────────────────────────────
# SharpInfer API — Multi-stage Docker Build
# ─────────────────────────────────────────────────────────────
# Build:   docker build -t sharpinfer .
# Run:     docker run -p 3512:3512 -v ./models:/models sharpinfer
# Pre-load: docker run -p 3512:3512 -v ./models:/models sharpinfer --model /models/your-model.gguf
# GPU:     docker run --gpus all -p 3512:3512 -v ./models:/models sharpinfer --gpu
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Build ───────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy solution and project files first (layer caching for restore)
COPY SharpInfer.sln ./
COPY src/SharpInfer.Core/SharpInfer.Core.csproj         src/SharpInfer.Core/
COPY src/SharpInfer.Gpu/SharpInfer.Gpu.csproj           src/SharpInfer.Gpu/
COPY src/SharpInfer.Api/SharpInfer.Api.csproj           src/SharpInfer.Api/
COPY src/SharpInfer.Cli/SharpInfer.Cli.csproj           src/SharpInfer.Cli/
COPY src/SharpInfer.VsCode/SharpInfer.VsCode.csproj     src/SharpInfer.VsCode/

# Restore dependencies (cached unless .csproj files change)
RUN dotnet restore src/SharpInfer.Api/SharpInfer.Api.csproj

# Copy everything else and publish
COPY . .
RUN dotnet publish src/SharpInfer.Api/SharpInfer.Api.csproj \
    -c Release \
    -o /app/publish \
    --no-restore

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app

# Create a non-root user for security
RUN groupadd -r sharpinfer && useradd -r -g sharpinfer -m sharpinfer

# Create the default models directory
RUN mkdir -p /models && chown sharpinfer:sharpinfer /models

# Copy published application
COPY --from=build /app/publish .

# Expose the default API port
EXPOSE 3512

# Switch to non-root user
USER sharpinfer

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3512/health || exit 1

# Default entrypoint — pass additional args via docker run or docker-compose
# Example: docker run sharpinfer --model /models/llama3.gguf --gpu
ENTRYPOINT ["dotnet", "SharpInfer.Api.dll"]
CMD ["--port", "3512", "--models-dir", "/models"]
