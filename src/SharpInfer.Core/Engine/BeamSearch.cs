using SharpInfer.Core.Layers;
using SharpInfer.Core.Tensors;

namespace SharpInfer.Core.Engine;

/// <summary>
/// Beam search with optional reranking.
///
/// Maintains N parallel hypotheses (beams) through the generation process,
/// expanding each beam at every step and pruning to keep the top N.
///
/// Each beam has its own KV cache, so memory cost is N x standard generation.
///
/// Reranking strategies:
/// - Log probability (default): sum of log probabilities, optionally length-normalized
/// - Length penalty: penalize very short or very long responses
/// - Diversity penalty: penalize beams that are too similar to each other
///
/// Use cases:
/// - Translation (where the best output may not be the greedily decoded one)
/// - Structured output (JSON, code) where correctness matters more than diversity
/// - When you need the single best response rather than creative variety
///
/// Note: Beam search tends to produce less diverse, more "boring" outputs.
/// For creative tasks, sampling-based methods are usually preferred.
/// </summary>
public class BeamSearch
{
    private readonly Transformer _model;

    /// <summary>Number of beams (parallel hypotheses).</summary>
    public int NumBeams { get; set; } = 4;

    /// <summary>
    /// Length penalty. >1.0 favors longer sequences, lower 1.0 favors shorter.
    /// Applied as: score / (length ^ penalty)
    /// </summary>
    public float LengthPenalty { get; set; } = 1.0f;

    /// <summary>
    /// Diversity penalty. Subtracted from beam scores when a token has already
    /// been selected by a previous beam at the same step.
    /// 0 = no diversity encouragement.
    /// </summary>
    public float DiversityPenalty { get; set; } = 0f;

    /// <summary>Stop early if all beams have finished (hit EOS).</summary>
    public bool EarlyStopping { get; set; } = true;

    /// <summary>Number of final results to return (must be &lt;= NumBeams).</summary>
    public int NumReturn { get; set; } = 1;

    // Single pre-allocated logit buffer — beams are processed sequentially so one buffer suffices.
    private readonly float[] _logitBuf;

    public BeamSearch(Transformer model)
    {
        _model = model;
        _logitBuf = new float[model.Config.VocabSize];
    }

    /// <summary>
    /// Run beam search and return the top results.
    /// </summary>
    public async Task<List<BeamResult>> SearchAsync(
        List<int> promptTokens, int maxNewTokens, KVCache templateCache)
    {
        int vocabSize = _model.Config.VocabSize;
        int eosId = _model.Config.EosTokenId;

        // Initialize beams — all start from the same prompt
        var activeBeams = new List<Beam>();
        var finishedBeams = new List<Beam>();

        // Create initial beam with prefilled cache
        var initialBeam = new Beam
        {
            Tokens = new List<int>(promptTokens),
            LogProb = 0f,
            Cache = CloneCache(templateCache),
            IsFinished = false,
        };
        activeBeams.Add(initialBeam);

        // Prefill the initial cache
        for (int i = 0; i < promptTokens.Count; i++)
            _model.Forward(promptTokens[i], i, initialBeam.Cache);

        // Expand to NumBeams after first step
        int position = promptTokens.Count;

        for (int step = 0; step < maxNewTokens; step++)
        {
            var candidates = new List<BeamCandidate>();

            foreach (var beam in activeBeams)
            {
                if (beam.IsFinished)
                {
                    // Finished beams carry forward unchanged
                    candidates.Add(new BeamCandidate
                    {
                        ParentBeam = beam,
                        TokenId = -1,
                        CumulativeLogProb = beam.LogProb,
                    });
                    continue;
                }

                int lastToken = beam.Tokens[^1];
                // Copy into pre-allocated buffer — no ToArray() allocation
                _model.Forward(lastToken, position + step, beam.Cache).CopyTo(_logitBuf);

                // Convert to log probabilities
                float maxLogit = _logitBuf[0];
                for (int i = 1; i < vocabSize; i++) if (_logitBuf[i] > maxLogit) maxLogit = _logitBuf[i];
                float logSumExp = 0f;
                for (int i = 0; i < vocabSize; i++)
                    logSumExp += MathF.Exp(_logitBuf[i] - maxLogit);
                logSumExp = maxLogit + MathF.Log(logSumExp);

                // Take top-K tokens per beam (K = 2 * NumBeams for diversity)
                int topK = NumBeams * 2;
                var topIndices = _logitBuf
                    .Select((val, idx) => (val, idx))
                    .OrderByDescending(x => x.val)
                    .Take(topK)
                    .ToList();

                foreach (var (logit, tokenId) in topIndices)
                {
                    float logProb = logit - logSumExp;
                    candidates.Add(new BeamCandidate
                    {
                        ParentBeam = beam,
                        TokenId = tokenId,
                        CumulativeLogProb = beam.LogProb + logProb,
                    });
                }
            }

            // Apply diversity penalty
            if (DiversityPenalty > 0f)
            {
                var selectedTokens = new HashSet<int>();
                foreach (var cand in candidates.OrderByDescending(c => c.CumulativeLogProb))
                {
                    if (cand.TokenId >= 0 && selectedTokens.Contains(cand.TokenId))
                        cand.CumulativeLogProb -= DiversityPenalty;
                    if (cand.TokenId >= 0)
                        selectedTokens.Add(cand.TokenId);
                }
            }

            // Apply length penalty to score
            var scoredCandidates = candidates.Select(c =>
            {
                int length = c.ParentBeam.Tokens.Count - promptTokens.Count + 1;
                float score = c.CumulativeLogProb / MathF.Pow(Math.Max(length, 1), LengthPenalty);
                return (Candidate: c, Score: score);
            })
            .OrderByDescending(x => x.Score)
            .Take(NumBeams)
            .ToList();

            // Expand selected candidates into new beams
            var newBeams = new List<Beam>();
            foreach (var (cand, score) in scoredCandidates)
            {
                if (cand.TokenId < 0)
                {
                    // Finished beam carried forward
                    newBeams.Add(cand.ParentBeam);
                    continue;
                }

                var newBeam = new Beam
                {
                    Tokens = new List<int>(cand.ParentBeam.Tokens) { cand.TokenId },
                    LogProb = cand.CumulativeLogProb,
                    Cache = step == 0 && activeBeams.Count == 1
                        ? CloneCache(cand.ParentBeam.Cache) // Clone for first expansion
                        : cand.ParentBeam.Cache,            // Reuse parent cache
                    IsFinished = cand.TokenId == eosId,
                };

                if (newBeam.IsFinished)
                    finishedBeams.Add(newBeam);
                else
                    newBeams.Add(newBeam);
            }

            activeBeams = newBeams;

            // Early stopping
            if (EarlyStopping && activeBeams.All(b => b.IsFinished))
                break;

            if (activeBeams.Count == 0)
                break;

            await Task.Yield();
        }

        // Combine finished and active beams, select top results
        var allBeams = finishedBeams.Concat(activeBeams)
            .OrderByDescending(b => b.LogProb / MathF.Pow(
                Math.Max(b.Tokens.Count - promptTokens.Count, 1), LengthPenalty))
            .Take(NumReturn)
            .ToList();

        return allBeams.Select(b => new BeamResult
        {
            Tokens = b.Tokens.Skip(promptTokens.Count).ToList(),
            LogProb = b.LogProb,
            NormalizedScore = b.LogProb / MathF.Pow(
                Math.Max(b.Tokens.Count - promptTokens.Count, 1), LengthPenalty),
            IsComplete = b.IsFinished,
        }).ToList();
    }

    private KVCache CloneCache(KVCache source)
    {
        // Create a new cache and copy data from source
        // This is memory-expensive but necessary for independent beam exploration
        var clone = new KVCache(_model.Config);
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        source.WriteRawData(writer);
        writer.Flush();

        ms.Seek(0, SeekOrigin.Begin);
        using var reader = new BinaryReader(ms);
        clone.ReadRawData(reader, source.CurrentLength);

        return clone;
    }

    private class Beam
    {
        public required List<int> Tokens { get; init; }
        public float LogProb { get; set; }
        public required KVCache Cache { get; init; }
        public bool IsFinished { get; set; }
    }

    private class BeamCandidate
    {
        public required Beam ParentBeam { get; init; }
        public int TokenId { get; init; }
        public float CumulativeLogProb { get; set; }
    }
}

public class BeamResult
{
    public required List<int> Tokens { get; init; }
    public float LogProb { get; init; }
    public float NormalizedScore { get; init; }
    public bool IsComplete { get; init; }
}
