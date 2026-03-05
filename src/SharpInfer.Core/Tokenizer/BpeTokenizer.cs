using System.Text;
using System.Text.RegularExpressions;
using SharpInfer.Core.Models;

namespace SharpInfer.Core.Tokenizer;

/// <summary>
/// Byte-Pair Encoding (BPE) tokenizer compatible with LLaMA/SentencePiece and
/// HuggingFace BPE models. Handles encoding (text -> token IDs) and decoding
/// (token IDs -> text).
///
/// Supports two modes:
/// 1. SentencePiece-style (score-based merges from GGUF)
/// 2. HuggingFace-style (explicit merge rules from tokenizer.json)
/// </summary>
public class BpeTokenizer
{
    private readonly string[] _vocab;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly float[] _scores;
    private readonly Dictionary<(string, string), int> _mergeRanks;
    private readonly int _bosId;
    private readonly int _eosId;

    public int VocabSize => _vocab.Length;
    public int BosId => _bosId;
    public int EosId => _eosId;

    public BpeTokenizer(TokenizerData data, int bosId = 1, int eosId = 2)
    {
        _vocab = data.Vocabulary;
        _scores = data.Scores;
        _bosId = bosId;
        _eosId = eosId;

        _tokenToId = new Dictionary<string, int>(_vocab.Length);
        for (int i = 0; i < _vocab.Length; i++)
        {
            if (_vocab[i] != null)
                _tokenToId[_vocab[i]] = i;
        }

        _mergeRanks = new Dictionary<(string, string), int>();
        if (data.Merges != null)
        {
            for (int i = 0; i < data.Merges.Length; i++)
                _mergeRanks[data.Merges[i]] = i;
        }
    }

    /// <summary>
    /// Encode a string into a list of token IDs.
    /// Optionally prepend BOS and/or append EOS tokens.
    /// </summary>
    public List<int> Encode(string text, bool addBos = true, bool addEos = false)
    {
        var tokens = new List<int>();

        if (addBos)
            tokens.Add(_bosId);

        if (string.IsNullOrEmpty(text))
            return tokens;

        // SentencePiece convention: leading space becomes special underscore char
        text = "\u2581" + text.Replace(" ", "\u2581");

        if (_mergeRanks.Count > 0)
        {
            // HuggingFace BPE: split into characters, then apply merge rules
            tokens.AddRange(EncodeBpeMerges(text));
        }
        else
        {
            // SentencePiece-style: greedy longest-match then merge by score
            tokens.AddRange(EncodeSentencePiece(text));
        }

        if (addEos)
            tokens.Add(_eosId);

        return tokens;
    }

    /// <summary>Decode a sequence of token IDs back to a string.</summary>
    public string Decode(IEnumerable<int> tokenIds)
    {
        var sb = new StringBuilder();
        foreach (int id in tokenIds)
        {
            if (id == _bosId || id == _eosId) continue;
            if (id >= 0 && id < _vocab.Length && _vocab[id] != null)
                sb.Append(_vocab[id]);
        }
        // Undo SentencePiece space encoding
        return sb.ToString().Replace("\u2581", " ").TrimStart();
    }

    /// <summary>Decode a single token ID to its string representation.</summary>
    public string Decode(int tokenId)
    {
        if (tokenId >= 0 && tokenId < _vocab.Length && _vocab[tokenId] != null)
            return _vocab[tokenId].Replace("\u2581", " ");
        return "";
    }

    #region SentencePiece Encoding

    private List<int> EncodeSentencePiece(string text)
    {
        // Start: each character (or UTF-8 byte) is a separate token
        var symbols = new List<string>();
        foreach (char c in text)
            symbols.Add(c.ToString());

        // Iteratively merge the pair with the highest score
        while (symbols.Count >= 2)
        {
            float bestScore = float.MinValue;
            int bestIdx = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                string merged = symbols[i] + symbols[i + 1];
                if (_tokenToId.TryGetValue(merged, out int mergedId))
                {
                    float score = _scores[mergedId];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx < 0) break; // No more valid merges

            symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }

        // Convert symbols to token IDs
        var ids = new List<int>(symbols.Count);
        foreach (var sym in symbols)
        {
            if (_tokenToId.TryGetValue(sym, out int id))
                ids.Add(id);
            else
            {
                // Fallback: encode unknown chars as byte tokens
                foreach (byte b in Encoding.UTF8.GetBytes(sym))
                {
                    string byteToken = $"<0x{b:X2}>";
                    if (_tokenToId.TryGetValue(byteToken, out int byteId))
                        ids.Add(byteId);
                }
            }
        }

        return ids;
    }

    #endregion

    #region BPE Merge-Based Encoding

    private List<int> EncodeBpeMerges(string text)
    {
        // Start with individual characters
        var symbols = text.Select(c => c.ToString()).ToList();

        // Iteratively apply the highest-priority merge
        while (symbols.Count >= 2)
        {
            int bestRank = int.MaxValue;
            int bestIdx = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                if (_mergeRanks.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0) break;

            symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }

        var ids = new List<int>(symbols.Count);
        foreach (var sym in symbols)
        {
            if (_tokenToId.TryGetValue(sym, out int id))
                ids.Add(id);
        }

        return ids;
    }

    #endregion
}
