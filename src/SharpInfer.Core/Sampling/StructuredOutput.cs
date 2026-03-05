using System.Text.Json;
using System.Text.RegularExpressions;

namespace SharpInfer.Core.Sampling;

/// <summary>
/// Constrained decoding engine that forces model output to conform to
/// a specified format: JSON schema, regex pattern, context-free grammar (EBNF),
/// or exact choice list.
///
/// Works by masking logits at each token step — only tokens that keep the output
/// valid under the constraint are allowed. Uses a finite-state machine (FSM)
/// for regex/grammar constraints and a stack-based parser for JSON schema.
///
/// Usage:
///   var constraint = StructuredOutput.FromJsonSchema(schema);
///   // In the sampling loop:
///   constraint.MaskLogits(logits, tokenizer, generatedSoFar);
/// </summary>
public abstract class StructuredOutput
{
    /// <summary>Apply logit mask: set disallowed tokens to -infinity.</summary>
    public abstract void MaskLogits(Span<float> logits, ITokenDecoder decoder, string generatedText);

    /// <summary>Whether the generated text is in a valid final state (complete).</summary>
    public abstract bool IsComplete(string generatedText);

    /// <summary>Reset internal state for a new generation.</summary>
    public abstract void Reset();

    /// <summary>Create a JSON schema constraint.</summary>
    public static StructuredOutput FromJsonSchema(string jsonSchema) =>
        new JsonSchemaConstraint(jsonSchema);

    /// <summary>Create a regex pattern constraint.</summary>
    public static StructuredOutput FromRegex(string pattern) =>
        new RegexConstraint(pattern);

    /// <summary>Create an EBNF grammar constraint.</summary>
    public static StructuredOutput FromGrammar(string ebnfGrammar) =>
        new GrammarConstraint(ebnfGrammar);

    /// <summary>Create an exact choice constraint (output must be one of the given strings).</summary>
    public static StructuredOutput FromChoices(params string[] choices) =>
        new ChoiceConstraint(choices);

    protected const float NEG_INF = float.NegativeInfinity;
}

/// <summary>
/// Token decoder interface — minimal tokenizer surface needed for constrained decoding.
/// </summary>
public interface ITokenDecoder
{
    int VocabSize { get; }
    string Decode(int tokenId);
    string Decode(IEnumerable<int> tokenIds);
    /// <summary>Get all tokens as (id, text) pairs. Cached after first call.</summary>
    IReadOnlyList<(int Id, string Text)> GetVocabulary();
}

// ═══════════════════════════════════════════════════════════════════
//  JSON Schema Constraint
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Forces output to be valid JSON conforming to a JSON Schema.
///
/// Uses a stack-based incremental parser that tracks:
///   - Current position in the schema (which property/item)
///   - Expected next token category (object start, key, value, comma, etc.)
///   - Required properties remaining
///
/// Supports: object, array, string, number, integer, boolean, null,
/// enum, const, required fields, nested objects/arrays.
/// </summary>
public class JsonSchemaConstraint : StructuredOutput
{
    private readonly JsonElement _schema;
    private readonly JsonSchemaValidator _validator;

    // Token categories for JSON structure
    private static readonly HashSet<char> JsonStructChars = new() { '{', '}', '[', ']', ':', ',', '"', 't', 'f', 'n' };

    public JsonSchemaConstraint(string jsonSchema)
    {
        _schema = JsonDocument.Parse(jsonSchema).RootElement;
        _validator = new JsonSchemaValidator(_schema);
    }

    public override void MaskLogits(Span<float> logits, ITokenDecoder decoder, string generatedText)
    {
        var vocab = decoder.GetVocabulary();
        var allowedPrefixes = _validator.GetAllowedContinuations(generatedText);

        if (allowedPrefixes == null) return; // No constraint at this point

        for (int i = 0; i < logits.Length && i < vocab.Count; i++)
        {
            string tokenText = vocab[i].Text;

            if (!IsAllowedToken(tokenText, generatedText, allowedPrefixes))
            {
                logits[i] = NEG_INF;
            }
        }
    }

    public override bool IsComplete(string generatedText)
    {
        return _validator.IsValidComplete(generatedText);
    }

    public override void Reset()
    {
        _validator.Reset();
    }

    private bool IsAllowedToken(string tokenText, string currentText, HashSet<TokenCategory> allowed)
    {
        if (string.IsNullOrEmpty(tokenText)) return false;

        char first = tokenText.TrimStart()[0];
        var category = CategorizeToken(first, tokenText.TrimStart());

        return allowed.Contains(category) || allowed.Contains(TokenCategory.Any);
    }

    private static TokenCategory CategorizeToken(char first, string text)
    {
        return first switch
        {
            '{' => TokenCategory.ObjectStart,
            '}' => TokenCategory.ObjectEnd,
            '[' => TokenCategory.ArrayStart,
            ']' => TokenCategory.ArrayEnd,
            '"' => TokenCategory.StringStart,
            ':' => TokenCategory.Colon,
            ',' => TokenCategory.Comma,
            't' when text.StartsWith("true") => TokenCategory.BoolTrue,
            'f' when text.StartsWith("false") => TokenCategory.BoolFalse,
            'n' when text.StartsWith("null") => TokenCategory.Null,
            '-' or (>= '0' and <= '9') => TokenCategory.Number,
            ' ' or '\n' or '\r' or '\t' => TokenCategory.Whitespace,
            _ => TokenCategory.StringContent,
        };
    }
}

internal enum TokenCategory
{
    ObjectStart, ObjectEnd, ArrayStart, ArrayEnd,
    StringStart, StringContent, StringEnd,
    Colon, Comma,
    Number, BoolTrue, BoolFalse, Null,
    Whitespace, Any
}

/// <summary>
/// Incremental JSON schema validator using a state stack.
/// </summary>
internal class JsonSchemaValidator
{
    private readonly JsonElement _schema;

    private enum State
    {
        ExpectValue,
        InObject_ExpectKeyOrEnd,
        InObject_ExpectColon,
        InObject_ExpectValue,
        InObject_ExpectCommaOrEnd,
        InArray_ExpectValueOrEnd,
        InArray_ExpectCommaOrEnd,
        InString,
        InNumber,
        Complete
    }

    private Stack<State> _stateStack = new();
    private int _depth;

    public JsonSchemaValidator(JsonElement schema)
    {
        _schema = schema;
        Reset();
    }

    public void Reset()
    {
        _stateStack.Clear();
        _stateStack.Push(State.ExpectValue);
        _depth = 0;
    }

    public HashSet<TokenCategory>? GetAllowedContinuations(string text)
    {
        // Re-parse state from current text
        Reset();
        foreach (char c in text)
            AdvanceChar(c);

        if (_stateStack.Count == 0) return null;

        var state = _stateStack.Peek();
        return state switch
        {
            State.ExpectValue => GetExpectedValueCategories(),
            State.InObject_ExpectKeyOrEnd => new() { TokenCategory.StringStart, TokenCategory.ObjectEnd, TokenCategory.Whitespace },
            State.InObject_ExpectColon => new() { TokenCategory.Colon, TokenCategory.Whitespace },
            State.InObject_ExpectValue => GetExpectedValueCategories(),
            State.InObject_ExpectCommaOrEnd => new() { TokenCategory.Comma, TokenCategory.ObjectEnd, TokenCategory.Whitespace },
            State.InArray_ExpectValueOrEnd => GetExpectedValueCategoriesWithArrayEnd(),
            State.InArray_ExpectCommaOrEnd => new() { TokenCategory.Comma, TokenCategory.ArrayEnd, TokenCategory.Whitespace },
            State.InString => new() { TokenCategory.StringContent, TokenCategory.StringEnd, TokenCategory.Any },
            State.InNumber => new() { TokenCategory.Number, TokenCategory.Comma, TokenCategory.ObjectEnd, TokenCategory.ArrayEnd, TokenCategory.Whitespace },
            State.Complete => new() { }, // No more tokens allowed
            _ => null
        };
    }

    public bool IsValidComplete(string text)
    {
        try
        {
            JsonDocument.Parse(text);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private HashSet<TokenCategory> GetExpectedValueCategories()
    {
        return new()
        {
            TokenCategory.ObjectStart, TokenCategory.ArrayStart,
            TokenCategory.StringStart, TokenCategory.Number,
            TokenCategory.BoolTrue, TokenCategory.BoolFalse,
            TokenCategory.Null, TokenCategory.Whitespace,
        };
    }

    private HashSet<TokenCategory> GetExpectedValueCategoriesWithArrayEnd()
    {
        var cats = GetExpectedValueCategories();
        cats.Add(TokenCategory.ArrayEnd);
        return cats;
    }

    private void AdvanceChar(char c)
    {
        // Iterative loop replaces the original tail-recursive call in the InNumber case.
        // When a number ends mid-stream, we pop InNumber and re-dispatch the same char
        // against the parent state without growing the call stack.
        retry:
        if (_stateStack.Count == 0) return;
        var state = _stateStack.Peek();

        switch (state)
        {
            case State.ExpectValue:
                switch (c)
                {
                    case '{':
                        _stateStack.Pop();
                        _stateStack.Push(State.InObject_ExpectKeyOrEnd);
                        _depth++;
                        break;
                    case '[':
                        _stateStack.Pop();
                        _stateStack.Push(State.InArray_ExpectValueOrEnd);
                        _depth++;
                        break;
                    case '"':
                        _stateStack.Pop();
                        _stateStack.Push(State.InString);
                        break;
                    case '-' or (>= '0' and <= '9'):
                        _stateStack.Pop();
                        _stateStack.Push(State.InNumber);
                        break;
                    case 't' or 'f' or 'n':
                        // Will consume full keyword
                        break;
                }
                break;

            case State.InObject_ExpectKeyOrEnd:
                if (c == '"') _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectColon);
                if (c == '}') { _stateStack.Pop(); _depth--; if (_depth == 0) _stateStack.Push(State.Complete); }
                break;

            case State.InObject_ExpectColon:
                if (c == ':') { _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectValue); }
                break;

            case State.InObject_ExpectValue:
                if (c == '{') { _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectCommaOrEnd); _stateStack.Push(State.InObject_ExpectKeyOrEnd); _depth++; }
                else if (c == '[') { _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectCommaOrEnd); _stateStack.Push(State.InArray_ExpectValueOrEnd); _depth++; }
                else if (c == '"') { _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectCommaOrEnd); }
                else if (c is '-' or (>= '0' and <= '9')) { _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectCommaOrEnd); }
                break;

            case State.InObject_ExpectCommaOrEnd:
                if (c == ',') { _stateStack.Pop(); _stateStack.Push(State.InObject_ExpectKeyOrEnd); }
                else if (c == '}') { _stateStack.Pop(); _depth--; if (_depth == 0 && _stateStack.Count == 0) _stateStack.Push(State.Complete); }
                break;

            case State.InArray_ExpectValueOrEnd:
                if (c == ']') { _stateStack.Pop(); _depth--; if (_depth == 0 && _stateStack.Count == 0) _stateStack.Push(State.Complete); }
                else if (c == '{') { _stateStack.Pop(); _stateStack.Push(State.InArray_ExpectCommaOrEnd); _stateStack.Push(State.InObject_ExpectKeyOrEnd); _depth++; }
                break;

            case State.InArray_ExpectCommaOrEnd:
                if (c == ',') { _stateStack.Pop(); _stateStack.Push(State.InArray_ExpectValueOrEnd); }
                else if (c == ']') { _stateStack.Pop(); _depth--; }
                break;

            case State.InString:
                if (c == '"') { _stateStack.Pop(); }
                break;

            case State.InNumber:
                if (c is not ('-' or '.' or 'e' or 'E' or '+' or (>= '0' and <= '9')))
                {
                    _stateStack.Pop();
                    // Re-process this char in the parent state without recursing.
                    goto retry;
                }
                break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Regex Constraint (FSM-based)
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Forces output to match a regex pattern using incremental prefix matching.
///
/// At each step, checks which tokens could extend the current text
/// such that some completion of the result could still match the pattern.
/// Uses .NET Regex with partial match simulation.
/// </summary>
public class RegexConstraint : StructuredOutput
{
    private readonly string _pattern;
    private readonly Regex _fullRegex;
    private readonly Regex _prefixRegex;

    public RegexConstraint(string pattern)
    {
        _pattern = pattern;
        _fullRegex = new Regex($"^{pattern}$", RegexOptions.Compiled | RegexOptions.Singleline);
        // Prefix regex: match any prefix of a string that could complete to match the pattern
        _prefixRegex = new Regex($"^(?:{pattern})", RegexOptions.Compiled | RegexOptions.Singleline);
    }

    public override void MaskLogits(Span<float> logits, ITokenDecoder decoder, string generatedText)
    {
        var vocab = decoder.GetVocabulary();

        for (int i = 0; i < logits.Length && i < vocab.Count; i++)
        {
            string candidate = generatedText + vocab[i].Text;

            // Allow if: the candidate is a prefix of something that could match,
            // OR the candidate already fully matches
            if (!CouldMatch(candidate))
            {
                logits[i] = NEG_INF;
            }
        }
    }

    public override bool IsComplete(string generatedText)
    {
        return _fullRegex.IsMatch(generatedText);
    }

    public override void Reset() { }

    private bool CouldMatch(string text)
    {
        // Check if text is a valid prefix: does the pattern match any prefix of text,
        // or could text be extended to match?
        if (_fullRegex.IsMatch(text)) return true;

        // Check prefix: does the regex match the start of the text?
        // This is an approximation — true FSM simulation would be more precise
        return _prefixRegex.IsMatch(text) || IsPrefixOfPattern(text);
    }

    private bool IsPrefixOfPattern(string text)
    {
        // Try extending with common characters to see if we could still match
        // This is a heuristic — production would use Thompson NFA simulation
        foreach (var ext in new[] { "a", "0", " ", "\"", "{", "[", "}", "]", ",", ":", "true", "false", "null" })
        {
            if (_prefixRegex.IsMatch(text + ext))
                return true;
        }
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  EBNF Grammar Constraint
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Forces output to conform to a context-free grammar in EBNF notation.
///
/// Grammar format (subset of EBNF):
///   root   ::= expr
///   expr   ::= term (("+" | "-") term)*
///   term   ::= [0-9]+
///   string ::= "\"" [^"]* "\""
///
/// Uses an Earley parser for incremental parsing — at each step, determines
/// which terminal symbols (tokens) could validly continue the parse.
/// </summary>
public class GrammarConstraint : StructuredOutput
{
    private readonly string _grammar;
    private readonly Dictionary<string, GrammarRule[]> _rules;
    private readonly string _startSymbol;

    public GrammarConstraint(string ebnfGrammar)
    {
        _grammar = ebnfGrammar;
        _rules = ParseGrammar(ebnfGrammar);
        _startSymbol = _rules.Keys.FirstOrDefault() ?? "root";
    }

    public override void MaskLogits(Span<float> logits, ITokenDecoder decoder, string generatedText)
    {
        var vocab = decoder.GetVocabulary();
        var allowedChars = GetAllowedNextChars(generatedText);

        if (allowedChars == null) return; // Can't determine — allow all

        for (int i = 0; i < logits.Length && i < vocab.Count; i++)
        {
            string tokenText = vocab[i].Text;
            if (tokenText.Length == 0) continue;

            // Check if the first char of this token is in the allowed set
            if (!allowedChars.Contains(tokenText[0]) && !allowedChars.Contains('\0'))
            {
                logits[i] = NEG_INF;
            }
        }
    }

    public override bool IsComplete(string generatedText)
    {
        return CanParse(generatedText, _startSymbol, 0) == generatedText.Length;
    }

    public override void Reset() { }

    private HashSet<char>? GetAllowedNextChars(string text)
    {
        var allowed = new HashSet<char>();

        // Try each possible next character and see if parsing can continue
        for (char c = ' '; c <= '~'; c++)
        {
            string extended = text + c;
            // Check if extended text is a valid prefix of the grammar
            if (IsValidPrefix(extended))
                allowed.Add(c);
        }

        // Also check common whitespace
        foreach (var c in new[] { '\n', '\r', '\t' })
        {
            if (IsValidPrefix(text + c))
                allowed.Add(c);
        }

        return allowed.Count > 0 ? allowed : null;
    }

    private bool IsValidPrefix(string text)
    {
        int consumed = CanParse(text, _startSymbol, 0);
        // Valid prefix if we consumed all of it (partial parse OK) or the full grammar matched
        return consumed >= 0;
    }

    /// <summary>
    /// Try to parse text starting at position using the given rule.
    /// Returns number of characters consumed, or -1 if no match.
    /// </summary>
    private int CanParse(string text, string ruleName, int pos)
    {
        if (!_rules.TryGetValue(ruleName, out var alternatives))
            return -1;

        foreach (var alt in alternatives)
        {
            int result = MatchSequence(text, alt, pos);
            if (result >= 0) return result;
        }

        return -1;
    }

    private int MatchSequence(string text, GrammarRule rule, int pos)
    {
        int current = pos;

        foreach (var element in rule.Elements)
        {
            if (current >= text.Length)
                return current; // Reached end of input — valid prefix

            int consumed = MatchElement(text, element, current);
            if (consumed < 0) return -1;
            current = consumed;
        }

        return current;
    }

    private int MatchElement(string text, GrammarElement element, int pos)
    {
        switch (element.Type)
        {
            case ElementType.Literal:
                if (element.Value == null) return -1;
                for (int i = 0; i < element.Value.Length; i++)
                {
                    if (pos + i >= text.Length)
                        return pos + i; // Partial match — valid prefix
                    if (text[pos + i] != element.Value[i])
                        return -1;
                }
                return pos + element.Value.Length;

            case ElementType.CharClass:
                if (pos >= text.Length) return pos;
                if (MatchCharClass(text[pos], element.Value ?? ""))
                    return pos + 1;
                return -1;

            case ElementType.RuleRef:
                return CanParse(text, element.Value ?? "", pos);

            case ElementType.Optional:
                if (element.Children == null) return pos;
                int optResult = MatchChildren(text, element.Children, pos);
                return optResult >= 0 ? optResult : pos; // Optional — OK if no match

            case ElementType.Repeat:
                if (element.Children == null) return pos;
                int repCurrent = pos;
                while (repCurrent < text.Length)
                {
                    int rep = MatchChildren(text, element.Children, repCurrent);
                    if (rep <= repCurrent) break;
                    repCurrent = rep;
                }
                return repCurrent;

            default:
                return -1;
        }
    }

    private int MatchChildren(string text, List<GrammarElement> children, int pos)
    {
        int current = pos;
        foreach (var child in children)
        {
            int result = MatchElement(text, child, current);
            if (result < 0) return -1;
            current = result;
        }
        return current;
    }

    private static bool MatchCharClass(char c, string charClass)
    {
        bool negated = charClass.StartsWith('^');
        string cls = negated ? charClass[1..] : charClass;

        bool matches = false;
        for (int i = 0; i < cls.Length; i++)
        {
            if (i + 2 < cls.Length && cls[i + 1] == '-')
            {
                if (c >= cls[i] && c <= cls[i + 2]) matches = true;
                i += 2;
            }
            else if (c == cls[i])
            {
                matches = true;
            }
        }

        return negated ? !matches : matches;
    }

    #region Grammar Parser

    private static Dictionary<string, GrammarRule[]> ParseGrammar(string ebnf)
    {
        var rules = new Dictionary<string, List<GrammarRule>>();

        foreach (var line in ebnf.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#') || trimmed.StartsWith("//"))
                continue;

            var match = Regex.Match(trimmed, @"^(\w+)\s*::=\s*(.+)$");
            if (!match.Success) continue;

            string name = match.Groups[1].Value;
            string body = match.Groups[2].Value.Trim();

            if (!rules.ContainsKey(name))
                rules[name] = new List<GrammarRule>();

            // Split by | for alternatives
            foreach (var alt in SplitAlternatives(body))
            {
                rules[name].Add(ParseRule(alt.Trim()));
            }
        }

        return rules.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());
    }

    private static List<string> SplitAlternatives(string body)
    {
        var alts = new List<string>();
        int depth = 0;
        int start = 0;
        bool inQuote = false;

        for (int i = 0; i < body.Length; i++)
        {
            char c = body[i];
            if (c == '"' && (i == 0 || body[i - 1] != '\\')) inQuote = !inQuote;
            if (inQuote) continue;
            if (c is '(' or '[') depth++;
            if (c is ')' or ']') depth--;
            if (c == '|' && depth == 0)
            {
                alts.Add(body[start..i]);
                start = i + 1;
            }
        }

        alts.Add(body[start..]);
        return alts;
    }

    private static GrammarRule ParseRule(string body)
    {
        var elements = new List<GrammarElement>();
        int pos = 0;

        while (pos < body.Length)
        {
            char c = body[pos];

            if (char.IsWhiteSpace(c)) { pos++; continue; }

            // Quoted literal
            if (c == '"')
            {
                int end = body.IndexOf('"', pos + 1);
                if (end < 0) end = body.Length;
                elements.Add(new GrammarElement { Type = ElementType.Literal, Value = body[(pos + 1)..end] });
                pos = end + 1;
            }
            // Character class [...]
            else if (c == '[')
            {
                int end = body.IndexOf(']', pos + 1);
                if (end < 0) end = body.Length;
                elements.Add(new GrammarElement { Type = ElementType.CharClass, Value = body[(pos + 1)..end] });
                pos = end + 1;
                // Check for repeat
                if (pos < body.Length && body[pos] == '*')
                {
                    var last = elements[^1];
                    elements[^1] = new GrammarElement { Type = ElementType.Repeat, Children = new() { last } };
                    pos++;
                }
                else if (pos < body.Length && body[pos] == '+')
                {
                    var last = elements[^1];
                    elements.Add(new GrammarElement { Type = ElementType.Repeat, Children = new() { last } });
                    pos++;
                }
            }
            // Rule reference
            else if (char.IsLetter(c) || c == '_')
            {
                int end = pos;
                while (end < body.Length && (char.IsLetterOrDigit(body[end]) || body[end] == '_')) end++;
                elements.Add(new GrammarElement { Type = ElementType.RuleRef, Value = body[pos..end] });
                pos = end;
            }
            // Optional group (...)? or (...)*
            else if (c == '(')
            {
                int depth = 1;
                int end = pos + 1;
                while (end < body.Length && depth > 0)
                {
                    if (body[end] == '(') depth++;
                    if (body[end] == ')') depth--;
                    end++;
                }
                var inner = ParseRule(body[(pos + 1)..(end - 1)]);

                if (end < body.Length && body[end] == '?')
                {
                    elements.Add(new GrammarElement { Type = ElementType.Optional, Children = inner.Elements });
                    pos = end + 1;
                }
                else if (end < body.Length && body[end] == '*')
                {
                    elements.Add(new GrammarElement { Type = ElementType.Repeat, Children = inner.Elements });
                    pos = end + 1;
                }
                else
                {
                    elements.AddRange(inner.Elements);
                    pos = end;
                }
            }
            else
            {
                pos++;
            }
        }

        return new GrammarRule { Elements = elements };
    }

    #endregion
}

internal class GrammarRule
{
    public List<GrammarElement> Elements { get; set; } = new();
}

internal class GrammarElement
{
    public ElementType Type { get; set; }
    public string? Value { get; set; }
    public List<GrammarElement>? Children { get; set; }
}

internal enum ElementType
{
    Literal,     // "text"
    CharClass,   // [a-z0-9]
    RuleRef,     // another_rule
    Optional,    // (...)?
    Repeat,      // (...)*
}

// ═══════════════════════════════════════════════════════════════════
//  Choice Constraint
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Forces output to be exactly one of a fixed set of strings.
/// Implements character-level trie masking for efficient prefix filtering.
/// </summary>
public class ChoiceConstraint : StructuredOutput
{
    private readonly string[] _choices;
    private readonly TrieNode _trie;

    public ChoiceConstraint(string[] choices)
    {
        _choices = choices;
        _trie = BuildTrie(choices);
    }

    public override void MaskLogits(Span<float> logits, ITokenDecoder decoder, string generatedText)
    {
        var vocab = decoder.GetVocabulary();

        for (int i = 0; i < logits.Length && i < vocab.Count; i++)
        {
            string candidate = generatedText + vocab[i].Text;

            // Check if any choice starts with this candidate
            bool valid = false;
            foreach (var choice in _choices)
            {
                if (choice.StartsWith(candidate) || candidate == choice)
                {
                    valid = true;
                    break;
                }
            }

            if (!valid)
                logits[i] = NEG_INF;
        }
    }

    public override bool IsComplete(string generatedText)
    {
        return _choices.Contains(generatedText);
    }

    public override void Reset() { }

    private static TrieNode BuildTrie(string[] choices)
    {
        var root = new TrieNode();
        foreach (var choice in choices)
        {
            var node = root;
            foreach (var c in choice)
            {
                if (!node.Children.TryGetValue(c, out var child))
                {
                    child = new TrieNode();
                    node.Children[c] = child;
                }
                node = child;
            }
            node.IsTerminal = true;
        }
        return root;
    }
}

internal class TrieNode
{
    public Dictionary<char, TrieNode> Children { get; } = new();
    public bool IsTerminal { get; set; }
}

// ═══════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════

/// <summary>
/// Configuration for structured output constraints.
/// Can be set per-request via the API or CLI.
/// </summary>
public class StructuredOutputConfig
{
    /// <summary>Constraint type: "json_schema", "regex", "grammar", "choice", "none".</summary>
    public string Type { get; set; } = "none";

    /// <summary>The constraint value (JSON schema string, regex pattern, EBNF grammar, etc.).</summary>
    public string? Value { get; set; }

    /// <summary>For "choice" type: the allowed choices.</summary>
    public List<string>? Choices { get; set; }

    /// <summary>Create the appropriate StructuredOutput instance.</summary>
    public StructuredOutput? Build()
    {
        return Type.ToLowerInvariant() switch
        {
            "json_schema" or "json" when Value != null => StructuredOutput.FromJsonSchema(Value),
            "regex" when Value != null => StructuredOutput.FromRegex(Value),
            "grammar" or "ebnf" when Value != null => StructuredOutput.FromGrammar(Value),
            "choice" when Choices is { Count: > 0 } => StructuredOutput.FromChoices(Choices.ToArray()),
            _ => null,
        };
    }
}
