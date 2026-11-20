//! Hand-written text metrics for `bench_cli compare`: word error rate, a
//! word-level diff, and two cheap hallucination signals.
//!
//! No crate for any of it, for the same reason the DSP is hand-written: each
//! piece is a few dozen lines, and the normalisation rules in particular have
//! to be ones this project chose — a generic "strip non-alphanumerics" would
//! delete the Tamil virama (U+0BCD, a combining mark that is not alphabetic)
//! and silently turn every Tamil word into a different one.

/// Characters treated as punctuation, beyond ASCII's. Deliberately an explicit
/// list rather than a Unicode category test: combining marks — which Tamil
/// needs — must never be caught by it.
const UNICODE_PUNCTUATION: &[char] = &[
    '\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}', '\u{201C}', '\u{201D}', '\u{201E}',
    '\u{2026}', '\u{2013}', '\u{2014}', '\u{2012}', '\u{2015}', '\u{00AB}', '\u{00BB}',
    '\u{00BF}', '\u{00A1}', '\u{0964}', '\u{0965}', '\u{3001}', '\u{3002}', '\u{FF0C}',
    '\u{FF01}', '\u{FF1F}', '\u{00B7}', '\u{2022}',
];

/// Apostrophes join rather than split ("don't" → "dont"), so a hypothesis
/// that drops the apostrophe is not charged for it.
const APOSTROPHES: &[char] = &['\'', '\u{2019}', '\u{2018}', '`'];

/// Lowercases, strips punctuation and splits on whitespace. Unicode-aware:
/// Tamil script (including its vowel signs and virama) passes through intact.
pub fn normalize_words(text: &str) -> Vec<String> {
    let mut cleaned = String::with_capacity(text.len());
    for c in text.chars() {
        if APOSTROPHES.contains(&c) {
            continue;
        }
        if c.is_ascii_punctuation() || UNICODE_PUNCTUATION.contains(&c) {
            cleaned.push(' ');
        } else {
            cleaned.extend(c.to_lowercase());
        }
    }
    cleaned.split_whitespace().map(str::to_string).collect()
}

/// Whitespace tokens exactly as written — for measuring what a cleanup pass
/// changed, where casing and punctuation *are* the change.
pub fn raw_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EditCounts {
    pub reference_words: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
}

impl EditCounts {
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    /// Word error rate, 0.0.. (can exceed 1.0 when the hypothesis is mostly
    /// insertions). `None` for an empty reference, where it is undefined.
    pub fn rate(&self) -> Option<f64> {
        if self.reference_words == 0 {
            return None;
        }
        Some(self.errors() as f64 / self.reference_words as f64)
    }

    pub fn add(&mut self, other: &EditCounts) {
        self.reference_words += other.reference_words;
        self.substitutions += other.substitutions;
        self.deletions += other.deletions;
        self.insertions += other.insertions;
    }
}

/// Word-level Levenshtein alignment of `hypothesis` against `reference`,
/// counting substitutions, deletions and insertions. Two rolling rows, so a
/// meeting-length transcript costs O(n·m) time but only O(m) memory.
pub fn edit_counts(reference: &[String], hypothesis: &[String]) -> EditCounts {
    #[derive(Clone, Copy, Default)]
    struct Cell {
        cost: usize,
        subs: usize,
        dels: usize,
        ins: usize,
    }
    let m = hypothesis.len();
    let mut previous: Vec<Cell> = (0..=m)
        .map(|j| Cell {
            cost: j,
            ins: j,
            ..Cell::default()
        })
        .collect();
    let mut current = vec![Cell::default(); m + 1];

    for (i, ref_word) in reference.iter().enumerate() {
        current[0] = Cell {
            cost: i + 1,
            dels: i + 1,
            ..Cell::default()
        };
        for j in 1..=m {
            let diagonal = previous[j - 1];
            let substitution = if &hypothesis[j - 1] == ref_word {
                diagonal
            } else {
                Cell {
                    cost: diagonal.cost + 1,
                    subs: diagonal.subs + 1,
                    ..diagonal
                }
            };
            let deletion = Cell {
                cost: previous[j].cost + 1,
                dels: previous[j].dels + 1,
                ..previous[j]
            };
            let insertion = Cell {
                cost: current[j - 1].cost + 1,
                ins: current[j - 1].ins + 1,
                ..current[j - 1]
            };
            // Ties prefer the diagonal, so an equal-cost alignment reports a
            // substitution rather than a delete+insert pair.
            let mut best = substitution;
            if deletion.cost < best.cost {
                best = deletion;
            }
            if insertion.cost < best.cost {
                best = insertion;
            }
            current[j] = best;
        }
        std::mem::swap(&mut previous, &mut current);
    }

    let last = previous[m];
    EditCounts {
        reference_words: reference.len(),
        substitutions: last.subs,
        deletions: last.dels,
        insertions: last.ins,
    }
}

/// WER of `hypothesis` against `reference`, both normalised first.
pub fn wer(reference: &str, hypothesis: &str) -> EditCounts {
    edit_counts(&normalize_words(reference), &normalize_words(hypothesis))
}

/// One step of a word diff.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffOp {
    Same(String),
    Removed(String),
    Added(String),
}

/// Word diff of `after` against `before`, from a full Levenshtein matrix with
/// backtrace. Callers diff one transcript turn at a time, which keeps the
/// matrix small.
pub fn diff_words(before: &[String], after: &[String]) -> Vec<DiffOp> {
    let n = before.len();
    let m = after.len();
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in cost.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        cost[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let diagonal = cost[i - 1][j - 1] + usize::from(before[i - 1] != after[j - 1]);
            cost[i][j] = diagonal.min(cost[i - 1][j] + 1).min(cost[i][j - 1] + 1);
        }
    }

    let mut ops = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && before[i - 1] == after[j - 1] && cost[i][j] == cost[i - 1][j - 1] {
            ops.push(DiffOp::Same(before[i - 1].clone()));
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && cost[i][j] == cost[i - 1][j - 1] + 1 {
            // A substitution reads best as "removed, then added".
            ops.push(DiffOp::Added(after[j - 1].clone()));
            ops.push(DiffOp::Removed(before[i - 1].clone()));
            i -= 1;
            j -= 1;
        } else if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            ops.push(DiffOp::Removed(before[i - 1].clone()));
            i -= 1;
        } else {
            ops.push(DiffOp::Added(after[j - 1].clone()));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

/// `word [-old-]{+new+} word` — the familiar `git diff --word-diff` notation.
pub fn render_diff(ops: &[DiffOp]) -> String {
    ops.iter()
        .map(|op| match op {
            DiffOp::Same(w) => w.clone(),
            DiffOp::Removed(w) => format!("[-{w}-]"),
            DiffOp::Added(w) => format!("{{+{w}+}}"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The longest run of one n-gram (n = 1..=4) repeated back to back, e.g.
/// "thank you thank you thank you" → (3, "thank you"). Whisper's classic
/// failure in a long pause is exactly this loop. Ties go to the longer
/// n-gram, which covers more of the transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct RepeatRun {
    pub times: usize,
    pub ngram: String,
}

pub fn longest_repeat_run(words: &[String]) -> Option<RepeatRun> {
    let mut best: Option<RepeatRun> = None;
    let mut best_key = (1usize, 0usize);
    for n in 1..=4usize {
        if words.len() < 2 * n {
            break;
        }
        let mut start = 0;
        while start + 2 * n <= words.len() {
            let gram = &words[start..start + n];
            let mut times = 1;
            while start + (times + 1) * n <= words.len()
                && &words[start + times * n..start + (times + 1) * n] == gram
            {
                times += 1;
            }
            if times >= 2 && (times, n) > best_key {
                best_key = (times, n);
                best = Some(RepeatRun {
                    times,
                    ngram: gram.join(" "),
                });
            }
            start += if times >= 2 { (times - 1) * n } else { 1 };
        }
    }
    best
}

/// Segments whose (normalised, non-empty) text repeats one of the previous
/// three segments' verbatim — whisper re-emitting a line it already wrote.
pub fn repeated_segments(texts: &[String]) -> usize {
    let normalized: Vec<String> = texts.iter().map(|t| normalize_words(t).join(" ")).collect();
    let mut count = 0;
    for (i, text) in normalized.iter().enumerate() {
        if text.is_empty() {
            continue;
        }
        let from = i.saturating_sub(3);
        if normalized[from..i].iter().any(|earlier| earlier == text) {
            count += 1;
        }
    }
    count
}

/// `mm:ss`, or `h:mm:ss` past the hour.
pub fn clock(ms: i64) -> String {
    let total = (ms / 1000).max(0);
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Parses `mm:ss` or `h:mm:ss` into milliseconds.
pub fn parse_clock(text: &str) -> Result<i64, String> {
    let parts: Vec<&str> = text.trim().split(':').collect();
    let numbers: Result<Vec<i64>, _> = parts.iter().map(|p| p.trim().parse::<i64>()).collect();
    let numbers = numbers.map_err(|_| format!("`{text}` is not a time (use mm:ss)"))?;
    let seconds = match numbers.as_slice() {
        [m, s] if *s < 60 => m * 60 + s,
        [h, m, s] if *m < 60 && *s < 60 => h * 3600 + m * 60 + s,
        _ => return Err(format!("`{text}` is not a time (use mm:ss or h:mm:ss)")),
    };
    Ok(seconds * 1000)
}

/// Parses `mm:ss-mm:ss`.
pub fn parse_range(text: &str) -> Result<(i64, i64), String> {
    let (start, end) = text
        .split_once('-')
        .ok_or_else(|| format!("`{text}` is not a range (use mm:ss-mm:ss)"))?;
    let (start, end) = (parse_clock(start)?, parse_clock(end)?);
    if end <= start {
        return Err(format!("range `{text}` ends before it starts"));
    }
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        normalize_words(text)
    }

    #[test]
    fn a_perfect_transcript_has_zero_wer() {
        let counts = wer("Hello, world!", "hello world");
        assert_eq!(counts.errors(), 0);
        assert_eq!(counts.rate(), Some(0.0));
    }

    #[test]
    fn counts_each_kind_of_error() {
        let sub = wer("we ship on friday", "we ship on monday");
        assert_eq!((sub.substitutions, sub.deletions, sub.insertions), (1, 0, 0));
        let del = wer("we ship on friday", "we ship friday");
        assert_eq!((del.substitutions, del.deletions, del.insertions), (0, 1, 0));
        let ins = wer("we ship on friday", "we ship it on friday");
        assert_eq!((ins.substitutions, ins.deletions, ins.insertions), (0, 0, 1));
        // Six reference words, three errors of whatever mix: WER 0.5.
        let counts = wer("the cat sat on the mat", "the bat sat the mat mat");
        assert_eq!(counts.reference_words, 6);
        assert_eq!(counts.errors(), 3);
        assert!((counts.rate().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_cases_are_well_defined() {
        assert_eq!(wer("", "anything").rate(), None);
        let all_deleted = wer("one two three", "");
        assert_eq!(all_deleted.deletions, 3);
        assert_eq!(all_deleted.rate(), Some(1.0));
    }

    #[test]
    fn tamil_script_survives_normalisation_intact() {
        // "நாளைக்கு meeting இருக்கு" — the virama (்) and vowel signs are
        // combining marks; a naive alphanumeric filter would delete them.
        let normalized = words("நாளைக்கு Meeting, இருக்கு!");
        assert_eq!(normalized, vec!["நாளைக்கு", "meeting", "இருக்கு"]);
        assert!(normalized[0].contains('\u{0BCD}'), "virama was stripped");
        assert_eq!(wer("நாளைக்கு meeting இருக்கு", "நாளைக்கு Meeting இருக்கு.").errors(), 0);
        // A different Tamil word is still an error, not a normalisation match.
        let counts = wer("நாளைக்கு meeting இருக்கு", "நாளை meeting இருக்கு");
        assert_eq!(counts.substitutions, 1);
    }

    #[test]
    fn apostrophes_join_and_curly_quotes_are_punctuation() {
        assert_eq!(words("Don’t “stop” — now"), vec!["dont", "stop", "now"]);
        assert_eq!(wer("don't stop", "dont stop").errors(), 0);
    }

    #[test]
    fn diff_marks_changes_in_word_diff_notation() {
        let before = raw_words("we will ship it friday");
        let after = raw_words("We will ship it on Friday.");
        let ops = diff_words(&before, &after);
        let rendered = render_diff(&ops);
        assert!(rendered.starts_with("[-we-] {+We+} will ship it"), "{rendered}");
        assert!(rendered.contains("{+on+}"), "{rendered}");
        // Replaying the diff reproduces both sides.
        let kept_before: Vec<String> = ops
            .iter()
            .filter_map(|op| match op {
                DiffOp::Same(w) | DiffOp::Removed(w) => Some(w.clone()),
                DiffOp::Added(_) => None,
            })
            .collect();
        let kept_after: Vec<String> = ops
            .iter()
            .filter_map(|op| match op {
                DiffOp::Same(w) | DiffOp::Added(w) => Some(w.clone()),
                DiffOp::Removed(_) => None,
            })
            .collect();
        assert_eq!(kept_before, before);
        assert_eq!(kept_after, after);
    }

    #[test]
    fn finds_a_whisper_style_repetition_loop() {
        let run = longest_repeat_run(&words("okay so thank you thank you thank you bye")).unwrap();
        assert_eq!(run.times, 3);
        assert_eq!(run.ngram, "thank you");
        assert!(longest_repeat_run(&words("no repeats in this sentence")).is_none());
    }

    #[test]
    fn counts_segments_that_repeat_a_recent_one() {
        let texts: Vec<String> = ["Hello.", "hello", "Something else", "Hello!", "", ""]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // "hello" repeats #0, "Hello!" repeats #1 (within three back); empties never count.
        assert_eq!(repeated_segments(&texts), 2);
    }

    #[test]
    fn parses_clock_times_and_ranges() {
        assert_eq!(parse_clock("02:30").unwrap(), 150_000);
        assert_eq!(parse_clock("1:02:03").unwrap(), 3_723_000);
        assert!(parse_clock("2:75").is_err());
        assert_eq!(parse_range("01:00-02:30").unwrap(), (60_000, 150_000));
        assert!(parse_range("02:00-01:00").is_err());
        assert_eq!(clock(65_000), "01:05");
        assert_eq!(clock(3_723_000), "1:02:03");
    }
}
