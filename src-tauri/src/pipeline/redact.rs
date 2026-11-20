//! Unparliamentary-language detection over a transcript.
//!
//! Meeting minutes get circulated. A verbatim record of a heated discussion
//! often should not be — the way an official record expunges unparliamentary
//! words rather than reprinting them.
//!
//! **The design decision that matters most: this never destroys the text.**
//! Detection produces *annotations*, not edits. Each transcript turn gains a
//! `flags` array and `transcript.json` keeps exactly what was said; masking is
//! applied at render and export time. That keeps the project's raw-artifact
//! rule intact, makes the policy adjustable without re-running STT, keeps
//! evidence timestamps valid, and makes the whole thing reversible when it
//! gets something wrong — which it will. `stt.json` is never touched.
//!
//! Two tiers are implemented here, in the order the roadmap sets out:
//!
//! * **Tier 1, a lexicon matcher.** Aho-Corasick over normalised turns. It is
//!   deterministic, auditable, needs no model and is fast enough to be free.
//!   Terms are data, not code: the lists live in `assets/lexicon/` and an
//!   organisation can override them from its own app-data directory without a
//!   rebuild, the same pattern the versioned prompts already use.
//! * **Tier 2, tolerance for STT error.** Whisper mishears profanity often and
//!   some builds sanitise it outright, so exact matching under-detects.
//!   Bounded edit distance and a phonetic key close some of that gap. Both are
//!   **gated to `severe` terms only** — applied broadly, the false-positive
//!   rate climbs fast, and a false positive on an innocent word in a
//!   circulated record is the failure that actually costs something.
//!
//! Tier 3, a learned classifier, is deliberately not here. See the note at the
//! bottom of this file.
//!
//! What a lexicon fundamentally cannot do is understand context. Quotation,
//! sarcasm and reclaimed usage will be misclassified sometimes. That is the
//! accepted cost of a deterministic Tier 1, and it is the reason `flag` mode
//! exists: a human looks at what was found before anything is hidden.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

const BUILTIN_EN: &str = include_str!("../../assets/lexicon/en.txt");
const BUILTIN_TA: &str = include_str!("../../assets/lexicon/ta.txt");
const BUILTIN_TA_LATN: &str = include_str!("../../assets/lexicon/ta-latn.txt");

/// The languages a lexicon file can be supplied for. The stem is also the
/// override file name, so `ta-latn` is `ta-latn.txt`.
pub const LEXICON_LANGUAGES: [&str; 3] = ["en", "ta", "ta-latn"];

/// Shortest term Tier 2 will consider. Below this, one edit is most of the
/// word, and every four-letter term in the language is one edit from another.
const MIN_FUZZY_LEN: usize = 5;

/// Fewest single-character tokens that count as a deliberately spaced-out
/// word. Two is an ordinary sentence ("a b"), three is someone spelling
/// something out.
const MIN_SPACED_RUN: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Profanity,
    Slur,
    Sexual,
    Violent,
    Insult,
}

impl Category {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "profanity" => Some(Category::Profanity),
            "slur" => Some(Category::Slur),
            "sexual" => Some(Category::Sexual),
            "violent" => Some(Category::Violent),
            "insult" => Some(Category::Insult),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Category::Profanity => "profanity",
            Category::Slur => "slur",
            Category::Sexual => "sexual",
            Category::Violent => "violent",
            Category::Insult => "insult",
        }
    }
}

/// A slur and a mild expletive should not get identical handling, which is
/// the entire reason this exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Mild,
    Strong,
    Severe,
}

impl Severity {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mild" => Some(Severity::Mild),
            "strong" => Some(Severity::Strong),
            "severe" => Some(Severity::Severe),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Mild => "mild",
            Severity::Strong => "strong",
            Severity::Severe => "severe",
        }
    }
}

/// How a flag was found. Worth recording: an exact lexicon hit and a
/// phonetic guess deserve different amounts of trust from whoever reviews it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMethod {
    /// Tier 1: the normalised text contains the normalised term.
    Exact,
    /// Tier 1: the term was spelled out with separators between the letters.
    Spaced,
    /// Tier 2: within one edit of a severe term.
    Fuzzy,
    /// Tier 2: the same phonetic key as a severe term.
    Phonetic,
}

impl MatchMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchMethod::Exact => "exact",
            MatchMethod::Spaced => "spaced",
            MatchMethod::Fuzzy => "fuzzy",
            MatchMethod::Phonetic => "phonetic",
        }
    }
}

/// One flagged span of one transcript turn.
///
/// Offsets are in **Unicode scalar values (`char`s)**, counted from the start
/// of that turn's text — not bytes, which would be meaningless to the
/// TypeScript side, and not UTF-16 units. The frontend slices with
/// `Array.from(text)` to match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    pub start_char: usize,
    pub end_char: usize,
    /// The lexicon entry that matched, spelled as the policy author wrote it.
    /// This is what was *listed*, not necessarily what was said — a fuzzy or
    /// phonetic hit reports the term it resembled.
    pub term_id: String,
    pub category: Category,
    pub severity: Severity,
    pub method: MatchMethod,
}

impl Flag {
    pub fn len(&self) -> usize {
        self.end_char.saturating_sub(self.start_char)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What the user asked for. Every surface that shows transcript text has to
/// respect this — the transcript view, search, MoM evidence quotes and, most
/// of all, export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionMode {
    /// Do not even scan.
    Off,
    /// Detect and highlight; the original stays visible. The review mode, and
    /// the only honest default for something that misclassifies quotation.
    Flag,
    /// `f***` — keeps the first character and the length.
    Mask,
    /// `[expunged]` — hides the length too.
    Redact,
}

impl RedactionMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(RedactionMode::Off),
            "flag" => Some(RedactionMode::Flag),
            "mask" => Some(RedactionMode::Mask),
            "redact" => Some(RedactionMode::Redact),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RedactionMode::Off => "off",
            RedactionMode::Flag => "flag",
            RedactionMode::Mask => "mask",
            RedactionMode::Redact => "redact",
        }
    }

    /// True when this mode changes the text a reader sees. `flag` does not —
    /// it only decorates it.
    pub fn hides_text(self) -> bool {
        matches!(self, RedactionMode::Mask | RedactionMode::Redact)
    }
}

// ------------------------------------------------------------- normalising

/// Normalised text, plus the map back to the original.
///
/// `source[i]` is the char index in the original string that normalised char
/// `i` came from. Without it a match could be reported but never located, and
/// masking would have nothing to mask.
#[derive(Debug, Clone, Default)]
pub struct Normalized {
    pub text: Vec<char>,
    pub source: Vec<usize>,
}

/// Leetspeak substitutions, applied only next to a letter — so `Q1` and
/// `2025` survive, while `f0ck` and `sh!t` do not.
fn leet(c: char) -> Option<char> {
    match c {
        '@' => Some('a'),
        '0' => Some('o'),
        '1' | '!' | '|' => Some('i'),
        '3' => Some('e'),
        '4' => Some('a'),
        '5' | '$' => Some('s'),
        '7' => Some('t'),
        _ => None,
    }
}

/// Strips the diacritic from the Latin letters a meeting transcript actually
/// produces. Not a general Unicode decomposition: doing that properly needs a
/// normalisation table this crate has no reason to carry, and the failure
/// mode of missing one is a missed flag, not a wrong one.
fn strip_diacritic(c: char) -> char {
    match c {
        'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' | 'ā' => 'a',
        'é' | 'è' | 'ê' | 'ë' | 'ē' => 'e',
        'í' | 'ì' | 'î' | 'ï' | 'ī' => 'i',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ō' => 'o',
        'ú' | 'ù' | 'û' | 'ü' | 'ū' => 'u',
        'ç' => 'c',
        'ñ' => 'n',
        'ý' | 'ÿ' => 'y',
        other => other,
    }
}

/// Lowercases, folds leetspeak and diacritics, collapses repeated characters
/// and reduces every run of punctuation or whitespace to a single space.
///
/// Collapsing repeats is what catches `fuuuck`, and it is applied to the
/// lexicon terms too, so the two always agree. It does mean the normalised
/// form of an ordinary word is not a word (`assessment` becomes `asesment`),
/// which is fine: nothing but the matcher ever reads it.
pub fn normalize(text: &str) -> Normalized {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Normalized::default();

    for (index, &raw) in chars.iter().enumerate() {
        let neighbouring_letter = |offset: isize| -> bool {
            let position = index as isize + offset;
            if position < 0 || position as usize >= chars.len() {
                return false;
            }
            chars[position as usize].is_alphabetic()
        };

        let folded = match leet(raw) {
            Some(letter) if neighbouring_letter(-1) || neighbouring_letter(1) => letter,
            _ => strip_diacritic(raw),
        };

        for lowered in folded.to_lowercase() {
            if lowered.is_alphanumeric() {
                // Collapse a run of the same character to one.
                if out.text.last() == Some(&lowered) {
                    continue;
                }
                out.text.push(lowered);
                out.source.push(index);
            } else {
                // Any punctuation or whitespace becomes one separator, so
                // word boundaries survive but their spelling does not matter.
                if out.text.last() == Some(&' ') {
                    continue;
                }
                out.text.push(' ');
                out.source.push(index);
            }
        }
    }
    out
}

/// The normalised form of a lexicon term: the same pipeline, as a string.
fn normalize_term(term: &str) -> String {
    normalize(term).text.into_iter().collect()
}

fn is_boundary(text: &[char], index: usize) -> bool {
    match text.get(index) {
        None => true,
        Some(c) => !c.is_alphanumeric(),
    }
}

// -------------------------------------------------------------- the lexicon

#[derive(Debug, Clone)]
pub struct Term {
    /// The term as the lexicon author wrote it, lowercased. This is what
    /// appears in `redaction.json` and in the UI, because the normalised form
    /// is unreadable — repeat-collapsing turns "hoopy frood" into "hopy frod",
    /// which looks like a typo to anyone reviewing a policy.
    pub id: String,
    /// The normalised form actually matched against. Also the identity used
    /// when merging an override over a built-in entry, so two spellings that
    /// normalise the same way are one term.
    pub pattern: String,
    pub category: Category,
    pub severity: Severity,
}

/// Aho-Corasick over `char`s. Hand-written for the same reason the FFT and the
/// biquad are: it is a hundred lines against a dependency, and the alphabet
/// here is Unicode rather than bytes, which is the case a byte-oriented
/// implementation would need care around anyway.
#[derive(Debug, Default, Clone)]
struct Automaton {
    next: Vec<HashMap<char, usize>>,
    fail: Vec<usize>,
    /// Term indices ending at this node.
    outputs: Vec<Vec<usize>>,
}

impl Automaton {
    fn build(patterns: &[String]) -> Self {
        let mut automaton = Automaton {
            next: vec![HashMap::new()],
            fail: vec![0],
            outputs: vec![Vec::new()],
        };
        for (index, pattern) in patterns.iter().enumerate() {
            if pattern.is_empty() {
                continue;
            }
            let mut node = 0usize;
            for c in pattern.chars() {
                node = match automaton.next[node].get(&c) {
                    Some(&existing) => existing,
                    None => {
                        automaton.next.push(HashMap::new());
                        automaton.fail.push(0);
                        automaton.outputs.push(Vec::new());
                        let created = automaton.next.len() - 1;
                        automaton.next[node].insert(c, created);
                        created
                    }
                };
            }
            automaton.outputs[node].push(index);
        }

        // Failure links, breadth first.
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let roots: Vec<(char, usize)> = automaton.next[0]
            .iter()
            .map(|(&c, &node)| (c, node))
            .collect();
        for (_, node) in roots {
            automaton.fail[node] = 0;
            queue.push_back(node);
        }
        while let Some(node) = queue.pop_front() {
            let edges: Vec<(char, usize)> = automaton.next[node]
                .iter()
                .map(|(&c, &child)| (c, child))
                .collect();
            for (c, child) in edges {
                let mut candidate = automaton.fail[node];
                loop {
                    if let Some(&target) = automaton.next[candidate].get(&c) {
                        automaton.fail[child] = target;
                        break;
                    }
                    if candidate == 0 {
                        automaton.fail[child] = 0;
                        break;
                    }
                    candidate = automaton.fail[candidate];
                }
                // Flattening the suffix outputs here means the search does
                // not have to walk fail links to collect them.
                let inherited = automaton.outputs[automaton.fail[child]].clone();
                automaton.outputs[child].extend(inherited);
                queue.push_back(child);
            }
        }
        automaton
    }

    /// Every match, as `(start, end, term index)` over `text`.
    fn find(&self, text: &[char], lengths: &[usize]) -> Vec<(usize, usize, usize)> {
        let mut hits = Vec::new();
        let mut node = 0usize;
        for (position, &c) in text.iter().enumerate() {
            loop {
                if let Some(&target) = self.next[node].get(&c) {
                    node = target;
                    break;
                }
                if node == 0 {
                    break;
                }
                node = self.fail[node];
            }
            for &term in &self.outputs[node] {
                let end = position + 1;
                let start = end.saturating_sub(lengths[term]);
                hits.push((start, end, term));
            }
        }
        hits
    }
}

#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    pub terms: Vec<Term>,
    /// Normalised words that suppress a match they contain — the Scunthorpe
    /// guard.
    allow: HashSet<String>,
    automaton: Automaton,
    lengths: Vec<usize>,
    /// Indices of the `severe` terms long enough for Tier 2, with their
    /// phonetic keys precomputed.
    fuzzy: Vec<(usize, String)>,
}

/// One parsed line, before it becomes a term.
enum Entry {
    Term(Term),
    Allow(String),
}

fn parse_line(line: &str) -> Option<Entry> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    if let Some(word) = line.strip_prefix('!') {
        let normalized = normalize_term(word);
        return (!normalized.is_empty()).then_some(Entry::Allow(normalized));
    }
    let mut parts = line.split('|');
    let written = parts.next()?.trim().to_lowercase();
    let pattern = normalize_term(&written);
    if pattern.is_empty() {
        return None;
    }
    // A malformed line falls back to the most conservative reading rather
    // than being dropped silently: a term nobody can categorise is still a
    // term somebody meant to list.
    let category = parts
        .next()
        .and_then(Category::parse)
        .unwrap_or(Category::Profanity);
    let severity = parts
        .next()
        .and_then(Severity::parse)
        .unwrap_or(Severity::Strong);
    Some(Entry::Term(Term {
        id: written,
        pattern,
        category,
        severity,
    }))
}

impl Lexicon {
    /// Builds from raw lexicon file contents. Later sources win, which is
    /// what makes an override file an override.
    pub fn from_sources(sources: &[&str]) -> Self {
        let mut terms: BTreeMap<String, Term> = BTreeMap::new();
        let mut allow: HashSet<String> = HashSet::new();
        for source in sources {
            for line in source.lines() {
                match parse_line(line) {
                    Some(Entry::Term(term)) => {
                        terms.insert(term.pattern.clone(), term);
                    }
                    Some(Entry::Allow(word)) => {
                        allow.insert(word);
                    }
                    None => {}
                }
            }
        }

        let terms: Vec<Term> = terms.into_values().collect();
        let patterns: Vec<String> = terms.iter().map(|t| t.pattern.clone()).collect();
        let lengths: Vec<usize> = patterns.iter().map(|p| p.chars().count()).collect();
        let fuzzy = terms
            .iter()
            .enumerate()
            .filter(|(index, term)| {
                term.severity == Severity::Severe && lengths[*index] >= MIN_FUZZY_LEN
            })
            .map(|(index, term)| (index, phonetic_key(&term.pattern)))
            .collect();

        Lexicon {
            automaton: Automaton::build(&patterns),
            lengths,
            terms,
            allow,
            fuzzy,
        }
    }

    /// The lists compiled into the binary.
    pub fn builtin() -> Self {
        Self::from_sources(&[BUILTIN_EN, BUILTIN_TA, BUILTIN_TA_LATN])
    }

    /// The built-in lists, with any override files in `<data_dir>/lexicon/`
    /// merged over them.
    ///
    /// An unreadable override is skipped with a log line rather than being an
    /// error: a typo in an optional policy file must not stop a meeting from
    /// being processed.
    pub fn load(data_dir: &Path) -> Self {
        let mut owned: Vec<String> = Vec::new();
        for language in LEXICON_LANGUAGES {
            let path = data_dir.join("lexicon").join(format!("{language}.txt"));
            match std::fs::read_to_string(&path) {
                Ok(contents) => owned.push(contents),
                Err(error) if path.exists() => {
                    eprintln!("[redact] ignoring {}: {error}", path.display());
                }
                Err(_) => {}
            }
        }
        let mut sources: Vec<&str> = vec![BUILTIN_EN, BUILTIN_TA, BUILTIN_TA_LATN];
        sources.extend(owned.iter().map(|s| s.as_str()));
        Self::from_sources(&sources)
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

// ------------------------------------------------------------------ Tier 2

/// A deliberately crude phonetic key: drop vowels after the first character,
/// fold the consonants that English spelling uses interchangeably, and
/// collapse repeats.
///
/// This is Double Metaphone's idea without Double Metaphone's table. It is
/// only ever applied to `severe` terms, where the cost of a miss is high and
/// the number of candidate words is tiny, so the extra precision of a full
/// implementation would buy very little.
pub fn phonetic_key(word: &str) -> String {
    let lowered: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    let mut key = String::new();
    let mut index = 0usize;
    while index < lowered.len() {
        let c = lowered[index];
        let next = lowered.get(index + 1).copied();
        let folded = match (c, next) {
            ('p', Some('h')) => {
                index += 1;
                'f'
            }
            ('c', Some('k')) => {
                index += 1;
                'k'
            }
            ('g', Some('h')) => {
                index += 1;
                'g'
            }
            ('c', _) | ('q', _) | ('k', _) => 'k',
            ('z', _) | ('s', _) => 's',
            ('x', _) => 'k',
            ('v', _) | ('w', _) => 'f',
            ('j', _) | ('g', _) => 'j',
            ('y', _) | ('i', _) => 'i',
            (other, _) => other,
        };
        let is_vowel = matches!(folded, 'a' | 'e' | 'i' | 'o' | 'u');
        if key.is_empty() || !is_vowel {
            if key.chars().last() != Some(folded) {
                key.push(folded);
            }
        }
        index += 1;
    }
    key
}

/// True when `a` and `b` are at most one insertion, deletion or substitution
/// apart. Bounded at one, so it is a single linear scan rather than a matrix.
pub fn within_one_edit(a: &str, b: &str) -> bool {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > 1 {
        return false;
    }
    let (short, long) = if a.len() <= b.len() { (&a, &b) } else { (&b, &a) };
    let mut i = 0usize;
    let mut j = 0usize;
    let mut edited = false;
    while i < short.len() && j < long.len() {
        if short[i] == long[j] {
            i += 1;
            j += 1;
            continue;
        }
        if edited {
            return false;
        }
        edited = true;
        if short.len() == long.len() {
            i += 1;
        }
        j += 1;
    }
    true
}

// ------------------------------------------------------------------ scanning

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanOptions {
    /// Tier 2. Off by default: it is the half that guesses.
    pub fuzzy: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self { fuzzy: false }
    }
}

/// A token of the normalised text, with where it came from.
struct Token {
    start: usize,
    end: usize,
    text: String,
}

fn tokenize(normalized: &Normalized) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;
    for (index, &c) in normalized.text.iter().enumerate() {
        if c.is_alphanumeric() {
            start.get_or_insert(index);
        } else if let Some(from) = start.take() {
            tokens.push(Token {
                start: from,
                end: index,
                text: normalized.text[from..index].iter().collect(),
            });
        }
    }
    if let Some(from) = start {
        tokens.push(Token {
            start: from,
            end: normalized.text.len(),
            text: normalized.text[from..].iter().collect(),
        });
    }
    tokens
}

/// Maps a span of the normalised text back onto char offsets in the original.
fn to_source_span(normalized: &Normalized, start: usize, end: usize) -> Option<(usize, usize)> {
    let first = *normalized.source.get(start)?;
    let last = *normalized.source.get(end.saturating_sub(1))?;
    Some((first, last + 1))
}

/// Finds every flagged span in one piece of text.
pub fn scan(text: &str, lexicon: &Lexicon, options: &ScanOptions) -> Vec<Flag> {
    if text.is_empty() || lexicon.is_empty() {
        return Vec::new();
    }
    let normalized = normalize(text);
    let tokens = tokenize(&normalized);
    let mut flags: Vec<Flag> = Vec::new();

    let allowed = |start: usize, end: usize| -> bool {
        // A match is suppressed when the whole word containing it is on the
        // allowlist. "assess" contains "ass"; the word is not the term.
        tokens.iter().any(|token| {
            token.start <= start && token.end >= end && lexicon.allow.contains(&token.text)
        })
    };

    // Tier 1: the automaton over the normalised text.
    for (start, end, index) in lexicon.automaton.find(&normalized.text, &lexicon.lengths) {
        // Whole words only. Without this, every term is a substring of
        // something innocent somewhere.
        if !is_boundary(&normalized.text, start.wrapping_sub(1)) && start > 0 {
            continue;
        }
        if !is_boundary(&normalized.text, end) {
            continue;
        }
        if allowed(start, end) {
            continue;
        }
        if let Some((from, to)) = to_source_span(&normalized, start, end) {
            let term = &lexicon.terms[index];
            flags.push(Flag {
                start_char: from,
                end_char: to,
                term_id: term.id.clone(),
                category: term.category,
                severity: term.severity,
                method: MatchMethod::Exact,
            });
        }
    }

    // Tier 1b: a term spelled out one character at a time — "f u c k",
    // "f.u.c.k". Runs of single-character tokens are joined and matched on
    // their own, rather than stripping every separator in the text, which
    // would let a term be assembled across two ordinary words.
    for run in spaced_runs(&tokens) {
        let joined: String = run.iter().map(|token| token.text.as_str()).collect();
        for (index, term) in lexicon.terms.iter().enumerate() {
            let _ = index;
            if term.pattern.len() < 3 || !joined.contains(&term.pattern) {
                continue;
            }
            let start = run.first().map(|t| t.start).unwrap_or(0);
            let end = run.last().map(|t| t.end).unwrap_or(0);
            if let Some((from, to)) = to_source_span(&normalized, start, end) {
                flags.push(Flag {
                    start_char: from,
                    end_char: to,
                    term_id: term.id.clone(),
                    category: term.category,
                    severity: term.severity,
                    method: MatchMethod::Spaced,
                });
            }
        }
    }

    // Tier 2: severe terms only.
    if options.fuzzy {
        for token in &tokens {
            if token.text.chars().count() < MIN_FUZZY_LEN || allowed(token.start, token.end) {
                continue;
            }
            let key = phonetic_key(&token.text);
            for (index, term_key) in &lexicon.fuzzy {
                let term = &lexicon.terms[*index];
                if term.pattern == token.text {
                    continue; // Tier 1 already has this one.
                }
                let method = if within_one_edit(&token.text, &term.pattern) {
                    MatchMethod::Fuzzy
                } else if !key.is_empty()
                    && &key == term_key
                    && token.text.chars().count().abs_diff(term.pattern.chars().count()) <= 2
                {
                    MatchMethod::Phonetic
                } else {
                    continue;
                };
                if let Some((from, to)) = to_source_span(&normalized, token.start, token.end) {
                    flags.push(Flag {
                        start_char: from,
                        end_char: to,
                        term_id: term.id.clone(),
                        category: term.category,
                        severity: term.severity,
                        method,
                    });
                }
                break;
            }
        }
    }

    dedupe(flags)
}

/// Runs of consecutive single-character tokens, long enough to be deliberate.
fn spaced_runs(tokens: &[Token]) -> Vec<Vec<&Token>> {
    let mut runs = Vec::new();
    let mut current: Vec<&Token> = Vec::new();
    for token in tokens {
        if token.text.chars().count() == 1 {
            current.push(token);
        } else {
            if current.len() >= MIN_SPACED_RUN {
                runs.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.len() >= MIN_SPACED_RUN {
        runs.push(current);
    }
    runs
}

/// Drops a flag entirely contained in another, keeping the longest and, at
/// equal length, the most severe. "motherfucker" should produce one flag, not
/// three.
fn dedupe(mut flags: Vec<Flag>) -> Vec<Flag> {
    flags.sort_by(|a, b| {
        a.start_char
            .cmp(&b.start_char)
            .then(b.end_char.cmp(&a.end_char))
            .then(b.severity.cmp(&a.severity))
    });
    let mut kept: Vec<Flag> = Vec::new();
    for flag in flags {
        let contained = kept.iter().any(|existing| {
            existing.start_char <= flag.start_char && existing.end_char >= flag.end_char
        });
        if !contained {
            kept.push(flag);
        }
    }
    kept
}

// ------------------------------------------------------------------ applying

/// Renders `text` under `mode`.
///
/// `flag` returns the text unchanged — highlighting is the caller's job, and
/// the whole point of that mode is that the original is still readable.
pub fn apply(text: &str, flags: &[Flag], mode: RedactionMode) -> String {
    if !mode.hides_text() || flags.is_empty() {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut ordered: Vec<&Flag> = flags.iter().collect();
    ordered.sort_by_key(|f| f.start_char);

    let mut out = String::new();
    let mut cursor = 0usize;
    for flag in ordered {
        let start = flag.start_char.min(chars.len());
        let end = flag.end_char.min(chars.len());
        if start < cursor || start >= end {
            continue; // overlapping or empty; the first one already covered it
        }
        out.extend(chars[cursor..start].iter());
        match mode {
            RedactionMode::Mask => {
                // First character and length kept: enough for a reader to
                // follow the sentence, not enough to reprint the word.
                out.push(chars[start]);
                for _ in start + 1..end {
                    out.push('*');
                }
            }
            RedactionMode::Redact => out.push_str("[expunged]"),
            _ => out.extend(chars[start..end].iter()),
        }
        cursor = end;
    }
    out.extend(chars[cursor.min(chars.len())..].iter());
    out
}

/// Counts, for the UI badge and for `redaction.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RedactionSummary {
    pub total: usize,
    pub turns_affected: usize,
    /// Counts keyed by the serialised category / severity name, so the
    /// frontend does not need to know the enum shape.
    pub by_category: BTreeMap<String, usize>,
    pub by_severity: BTreeMap<String, usize>,
    /// Which lexicon entries fired, most frequent first. The term text, not
    /// the speaker — see the note about per-speaker counts below.
    pub terms: Vec<(String, usize)>,
    /// True when Tier 2 was enabled for this scan.
    pub fuzzy: bool,
    pub lexicon_terms: usize,
}

/// Scans every turn, returning one flag list per turn plus the summary.
///
/// Per-speaker counts are deliberately **not** produced. The data here would
/// make them trivial, and they would turn a transcription tool into a
/// surveillance one — which cuts against the project's rule about never
/// identifying real people. Build that only on an explicit, considered
/// request.
pub fn scan_turns(
    turns: &[String],
    lexicon: &Lexicon,
    options: &ScanOptions,
) -> (Vec<Vec<Flag>>, RedactionSummary) {
    let mut per_turn = Vec::with_capacity(turns.len());
    let mut summary = RedactionSummary {
        fuzzy: options.fuzzy,
        lexicon_terms: lexicon.len(),
        ..RedactionSummary::default()
    };
    let mut term_counts: BTreeMap<String, usize> = BTreeMap::new();

    for turn in turns {
        let flags = scan(turn, lexicon, options);
        if !flags.is_empty() {
            summary.turns_affected += 1;
            summary.total += flags.len();
            for flag in &flags {
                *summary
                    .by_category
                    .entry(flag.category.as_str().to_string())
                    .or_default() += 1;
                *summary
                    .by_severity
                    .entry(flag.severity.as_str().to_string())
                    .or_default() += 1;
                *term_counts.entry(flag.term_id.clone()).or_default() += 1;
            }
        }
        per_turn.push(flags);
    }

    let mut terms: Vec<(String, usize)> = term_counts.into_iter().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    summary.terms = terms;
    (per_turn, summary)
}

// A note on Tier 3, which is not implemented here.
//
// A small multilingual toxicity classifier exported to ONNX would run beside
// the diarizer at almost no architectural cost, and it would buy the one thing
// a wordlist can never have: context — an insult assembled from individually
// innocuous words, or a benign use of a flagged term. It is the right next
// step *if* Tiers 1 and 2 prove insufficient on real audio.
//
// It is not built yet because "insufficient" is not measurable today: no Tamil
// or code-switched audio has been through this pipeline, so there is no way to
// say what Tier 1 misses. Adding a model, a catalog entry, a Model Manager row
// and a memory-tier decision to fix an unmeasured gap would be guessing with a
// download attached.
//
// When it is built, prefer ONNX over reusing the loaded llama.cpp model. An
// LLM classification prompt needs no extra download, but it is slower per run
// and non-deterministic — both wrong properties for something that has to be
// auditable and reproducible.

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic lexicon. The real lists are not used for assertions here:
    /// tests that depend on shipped profanity break whenever the policy file
    /// changes, and a repository should not need real slurs in its test
    /// fixtures to prove a matcher works.
    const FIXTURE: &str = "
# a synthetic lexicon
zarquon | profanity | mild
blarg | insult | strong
frobnicate | profanity | severe
grunk | slur | severe
hoopy frood | insult | strong
!zarquonium
!blargicorp
";

    fn lexicon() -> Lexicon {
        Lexicon::from_sources(&[FIXTURE])
    }

    fn found(text: &str) -> Vec<String> {
        scan(text, &lexicon(), &ScanOptions::default())
            .into_iter()
            .map(|f| f.term_id)
            .collect()
    }

    fn found_fuzzy(text: &str) -> Vec<String> {
        scan(text, &lexicon(), &ScanOptions { fuzzy: true })
            .into_iter()
            .map(|f| f.term_id)
            .collect()
    }

    #[test]
    fn normalisation_folds_case_leetspeak_repeats_and_accents() {
        let normalized: String = normalize("ZAAARQU0N!!").text.into_iter().collect();
        assert_eq!(normalized, "zarquoni ");
        assert_eq!(normalize_term("Blarg"), "blarg");
        assert_eq!(normalize_term("café"), "cafe");
    }

    #[test]
    fn leetspeak_is_only_folded_next_to_letters() {
        // A number in ordinary use stays a number, so "Q1 2025 revenue" is
        // not quietly rewritten into letters.
        let normalized: String = normalize("Q1 2025").text.into_iter().collect();
        assert_eq!(normalized, "qi 2025");
    }

    #[test]
    fn an_exact_term_is_found_with_its_category_and_severity() {
        let flags = scan("That is total blarg, honestly", &lexicon(), &ScanOptions::default());
        assert_eq!(flags.len(), 1);
        assert_eq!(flags[0].term_id, "blarg");
        assert_eq!(flags[0].category, Category::Insult);
        assert_eq!(flags[0].severity, Severity::Strong);
        assert_eq!(flags[0].method, MatchMethod::Exact);
    }

    #[test]
    fn offsets_point_at_the_right_characters_of_the_original() {
        let text = "That is total blarg, honestly";
        let flags = scan(text, &lexicon(), &ScanOptions::default());
        let chars: Vec<char> = text.chars().collect();
        let slice: String = chars[flags[0].start_char..flags[0].end_char].iter().collect();
        assert_eq!(slice, "blarg");
    }

    #[test]
    fn offsets_survive_multibyte_text_before_the_match() {
        // Tamil ahead of the term: a byte offset would land mid-character.
        let text = "வணக்கம் blarg";
        let flags = scan(text, &lexicon(), &ScanOptions::default());
        assert_eq!(flags.len(), 1);
        let chars: Vec<char> = text.chars().collect();
        let slice: String = chars[flags[0].start_char..flags[0].end_char].iter().collect();
        assert_eq!(slice, "blarg");
    }

    #[test]
    fn a_stretched_or_leetspoken_spelling_still_matches() {
        assert_eq!(found("that is blaaaarg"), vec!["blarg"]);
        assert_eq!(found("that is bl@rg"), vec!["blarg"]);
    }

    #[test]
    fn a_term_inside_a_longer_word_is_not_flagged() {
        // The Scunthorpe problem: the substring is there, the word is not.
        assert!(found("blargonaut technologies").is_empty());
        assert!(found("preblarg").is_empty());
    }

    #[test]
    fn the_allowlist_suppresses_an_innocent_word() {
        assert!(found("we bought zarquonium shares").is_empty());
        assert!(found("blargicorp is the vendor").is_empty());
        // And the bare term is still caught.
        assert_eq!(found("zarquon"), vec!["zarquon"]);
    }

    #[test]
    fn a_multi_word_term_is_matched_across_the_space() {
        assert_eq!(found("what a hoopy frood he is"), vec!["hoopy frood"]);
    }

    #[test]
    fn a_word_spelled_out_letter_by_letter_is_caught() {
        let flags = scan("he said b l a r g out loud", &lexicon(), &ScanOptions::default());
        assert_eq!(flags.len(), 1);
        assert_eq!(flags[0].method, MatchMethod::Spaced);
        assert_eq!(flags[0].term_id, "blarg");

        let dotted = scan("he said b.l.a.r.g", &lexicon(), &ScanOptions::default());
        assert_eq!(dotted.len(), 1);
        assert_eq!(dotted[0].method, MatchMethod::Spaced);
    }

    #[test]
    fn ordinary_initials_are_not_assembled_into_a_term() {
        // Two single letters are a sentence, not a spelled-out word, and a
        // term must not be assembled across two ordinary words either.
        assert!(found("a b test of the ui").is_empty());
        assert!(found("the tabl a rg values").is_empty());
    }

    #[test]
    fn the_longest_overlapping_match_wins() {
        let source = "megablarg | insult | severe\nblarg | insult | strong";
        let lexicon = Lexicon::from_sources(&[source]);
        let flags = scan("that is megablarg", &lexicon, &ScanOptions::default());
        assert_eq!(flags.len(), 1);
        assert_eq!(flags[0].term_id, "megablarg");
    }

    #[test]
    fn tier_two_is_off_unless_asked_for() {
        // One letter out from a severe term.
        assert!(found("frobnicste").is_empty());
        assert_eq!(found_fuzzy("frobnicste"), vec!["frobnicate"]);
    }

    #[test]
    fn tier_two_only_applies_to_severe_terms() {
        // "blarg" is strong, not severe, and is under the length gate too:
        // a near miss of it stays unflagged even with Tier 2 on.
        assert!(found_fuzzy("blarj").is_empty());
        assert!(found_fuzzy("zarquan").is_empty());
    }

    #[test]
    fn a_phonetic_respelling_of_a_severe_term_is_caught() {
        assert_eq!(phonetic_key("frobnicate"), phonetic_key("phrobnikate"));
        assert_eq!(found_fuzzy("phrobnikate"), vec!["frobnicate"]);
    }

    #[test]
    fn edit_distance_is_bounded_at_one() {
        assert!(within_one_edit("blarg", "blarg"));
        assert!(within_one_edit("blarg", "blrg")); // deletion
        assert!(within_one_edit("blarg", "blaarg")); // insertion
        assert!(within_one_edit("blarg", "blorg")); // substitution
        assert!(!within_one_edit("blarg", "blrgg"));
        assert!(!within_one_edit("blarg", "bl"));
    }

    #[test]
    fn masking_keeps_the_first_character_and_the_length() {
        let text = "That is total blarg, honestly";
        let flags = scan(text, &lexicon(), &ScanOptions::default());
        assert_eq!(
            apply(text, &flags, RedactionMode::Mask),
            "That is total b****, honestly"
        );
    }

    #[test]
    fn redacting_hides_the_length_too() {
        let text = "That is total blarg, honestly";
        let flags = scan(text, &lexicon(), &ScanOptions::default());
        assert_eq!(
            apply(text, &flags, RedactionMode::Redact),
            "That is total [expunged], honestly"
        );
    }

    #[test]
    fn flag_mode_and_off_mode_leave_the_text_alone() {
        let text = "That is total blarg, honestly";
        let flags = scan(text, &lexicon(), &ScanOptions::default());
        assert_eq!(apply(text, &flags, RedactionMode::Flag), text);
        assert_eq!(apply(text, &flags, RedactionMode::Off), text);
    }

    #[test]
    fn masking_handles_several_flags_and_multibyte_text() {
        let text = "blarg வணக்கம் zarquon";
        let flags = scan(text, &lexicon(), &ScanOptions::default());
        assert_eq!(flags.len(), 2);
        assert_eq!(
            apply(text, &flags, RedactionMode::Mask),
            "b**** வணக்கம் z******"
        );
    }

    #[test]
    fn masking_an_empty_flag_list_is_the_original_text() {
        assert_eq!(apply("nothing here", &[], RedactionMode::Mask), "nothing here");
    }

    #[test]
    fn modes_round_trip_through_their_names() {
        for mode in [
            RedactionMode::Off,
            RedactionMode::Flag,
            RedactionMode::Mask,
            RedactionMode::Redact,
        ] {
            assert_eq!(RedactionMode::parse(mode.as_str()), Some(mode));
        }
        assert_eq!(RedactionMode::parse("nonsense"), None);
        assert!(!RedactionMode::Flag.hides_text());
        assert!(RedactionMode::Mask.hides_text());
        assert!(RedactionMode::Redact.hides_text());
    }

    #[test]
    fn a_summary_counts_by_category_and_severity_without_naming_speakers() {
        let turns = vec![
            "total blarg".to_string(),
            "nothing wrong here".to_string(),
            "blarg and zarquon".to_string(),
        ];
        let (per_turn, summary) = scan_turns(&turns, &lexicon(), &ScanOptions::default());
        assert_eq!(per_turn.len(), 3);
        assert!(per_turn[1].is_empty());
        assert_eq!(summary.total, 3);
        assert_eq!(summary.turns_affected, 2);
        assert_eq!(summary.by_category.get("insult"), Some(&2));
        assert_eq!(summary.by_category.get("profanity"), Some(&1));
        assert_eq!(summary.by_severity.get("strong"), Some(&2));
        assert_eq!(summary.terms[0], ("blarg".to_string(), 2));
    }

    #[test]
    fn a_malformed_lexicon_line_still_produces_a_usable_term() {
        let lexicon = Lexicon::from_sources(&["wibble", "wobble | nonsense | alsononsense"]);
        assert_eq!(lexicon.len(), 2);
        let flags = scan("wibble wobble", &lexicon, &ScanOptions::default());
        assert_eq!(flags.len(), 2);
        // The conservative fallback, not a dropped line.
        assert_eq!(flags[0].category, Category::Profanity);
        assert_eq!(flags[0].severity, Severity::Strong);
    }

    #[test]
    fn an_override_replaces_a_builtin_entry_rather_than_duplicating_it() {
        let base = "blarg | insult | mild";
        let over = "blarg | slur | severe";
        let lexicon = Lexicon::from_sources(&[base, over]);
        assert_eq!(lexicon.len(), 1);
        let flags = scan("blarg", &lexicon, &ScanOptions::default());
        assert_eq!(flags[0].category, Category::Slur);
        assert_eq!(flags[0].severity, Severity::Severe);
    }

    #[test]
    fn an_empty_lexicon_and_empty_text_are_both_safe() {
        let empty = Lexicon::from_sources(&["# nothing but a comment"]);
        assert!(empty.is_empty());
        assert!(scan("anything at all", &empty, &ScanOptions::default()).is_empty());
        assert!(scan("", &lexicon(), &ScanOptions::default()).is_empty());
    }

    #[test]
    fn the_shipped_lexicon_loads_and_carries_no_slurs() {
        let builtin = Lexicon::builtin();
        assert!(builtin.len() > 20, "only {} terms loaded", builtin.len());
        // The repository deliberately ships no slur list; the category exists
        // for organisations to populate from their own override file.
        assert!(
            !builtin.terms.iter().any(|t| t.category == Category::Slur),
            "a slur was committed to the repository lexicon"
        );
        // And the Tamil lists are present but empty, which is a stated gap
        // rather than an accident — see ta-latn.txt.
        let english_only = Lexicon::from_sources(&[BUILTIN_EN]);
        assert_eq!(english_only.len(), builtin.len());
    }

    #[test]
    fn an_override_directory_is_merged_over_the_builtin_lists() {
        let dir = std::env::temp_dir().join("cadence-lexicon-override");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("lexicon")).unwrap();
        std::fs::write(
            dir.join("lexicon").join("en.txt"),
            "quuxlorem | slur | severe\n",
        )
        .unwrap();

        let lexicon = Lexicon::load(&dir);
        assert!(lexicon.len() > Lexicon::builtin().len());
        let flags = scan("that is quuxlorem", &lexicon, &ScanOptions::default());
        assert_eq!(flags.len(), 1);
        assert_eq!(flags[0].category, Category::Slur);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loading_without_an_override_directory_is_the_builtin_lexicon() {
        let dir = std::env::temp_dir().join("cadence-lexicon-absent");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(Lexicon::load(&dir).len(), Lexicon::builtin().len());
    }
}
