//! D8 stop-string matching at the scheduler layer.
//!
//! Stop strings are handled as token-id sequences (the text→token conversion
//! belongs to the serving/API layer; the scheduler matches on the raw token
//! stream, so there is no detokenization ambiguity).
//!
//! Matching is incremental: after every generated token we maintain, per
//! pattern, the longest prefix of the pattern that matches the suffix of the
//! output stream ("partial match state"). A stop therefore fires the moment
//! the full pattern appears as a suffix of the stream — stop latency ≤ 1
//! step, no ambiguity. The matched tokens are consumed (not emitted),
//! matching OpenAI semantics.

use std::collections::VecDeque;

/// Result of feeding one generated token to the matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOutcome {
    /// No pattern fully matched yet.
    Continue,
    /// A stop pattern fully matched (index into the pattern list).
    Match {
        /// Index of the matched pattern.
        pattern: usize,
    },
}

/// Incremental multi-pattern stop matcher (D8 partial-match state).
#[derive(Debug, Clone)]
pub struct StopMatcher {
    patterns: Vec<Vec<u32>>,
    /// `matched[pi]`: longest prefix of `patterns[pi]` that is a suffix of the
    /// stream (0 = no partial match).
    matched: Vec<usize>,
    /// Last `max_len - 1` tokens of the stream (suffix window used for the
    /// fallback recomputation).
    buf: VecDeque<u32>,
    max_len: usize,
}

impl StopMatcher {
    /// Build a matcher from stop token sequences; empty patterns are ignored
    /// (no-op, same as vLLM's empty stop string).
    pub fn new(patterns: Vec<Vec<u32>>) -> Self {
        let patterns: Vec<Vec<u32>> = patterns.into_iter().filter(|p| !p.is_empty()).collect();
        let max_len = patterns.iter().map(Vec::len).max().unwrap_or(0);
        let n = patterns.len();
        Self { patterns, matched: vec![0; n], buf: VecDeque::new(), max_len }
    }

    /// Feed one generated token; updates the partial-match state of every
    /// pattern and returns a full-match outcome if any stop fired.
    pub fn push(&mut self, token: u32) -> StopOutcome {
        // Suffix window: last (max_len - 1) buffered tokens + the new token.
        let mut s: Vec<u32> = Vec::with_capacity(self.buf.len() + 1);
        s.extend(self.buf.iter().copied());
        s.push(token);
        for (pi, p) in self.patterns.iter().enumerate() {
            // Longest k with p[..k] == suffix(s, k): the extension case
            // (previous partial match + token) is covered by this loop too.
            let mut k = p.len().min(s.len());
            while k > 0 && p[..k] != s[s.len() - k..] {
                k -= 1;
            }
            self.matched[pi] = k;
            if k == p.len() {
                return StopOutcome::Match { pattern: pi };
            }
        }
        if self.max_len > 1 {
            self.buf.push_back(token);
            while self.buf.len() >= self.max_len {
                self.buf.pop_front();
            }
        }
        StopOutcome::Continue
    }

    /// Reset all partial-match state. Used when a request restarts prefill
    /// after preemption (its output stream starts over, D7).
    pub fn reset(&mut self) {
        self.matched.iter_mut().for_each(|m| *m = 0);
        self.buf.clear();
    }

    /// Number of configured (non-empty) patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn single_token_stop() {
        let mut m = StopMatcher::new(vec![vec![7]]);
        assert_eq!(m.push(1), StopOutcome::Continue);
        assert_eq!(m.push(7), StopOutcome::Match { pattern: 0 });
    }

    #[test]
    fn multi_token_stop_spans_pushes() {
        // Partial match state must survive across pushes.
        let mut m = StopMatcher::new(vec![vec![3, 1, 4]]);
        assert_eq!(m.push(3), StopOutcome::Continue);
        assert_eq!(m.push(1), StopOutcome::Continue);
        assert_eq!(m.push(9), StopOutcome::Continue, "mismatch resets");
        assert_eq!(m.push(3), StopOutcome::Continue);
        assert_eq!(m.push(1), StopOutcome::Continue);
        assert_eq!(m.push(4), StopOutcome::Match { pattern: 0 });
    }

    #[test]
    fn partial_then_mismatch_uses_fallback() {
        // Pattern abab; a partial "aba" followed by a mismatch must not
        // stick — the fallback (longest-prefix match against the stream
        // suffix) resets cleanly and a fresh "abab" fires later.
        let mut m = StopMatcher::new(vec![vec![1, 2, 1, 2]]);
        for t in [1, 2, 1] {
            assert_eq!(m.push(t), StopOutcome::Continue);
        }
        assert_eq!(m.push(9), StopOutcome::Continue, "mismatch resets the partial match");
        for t in [1, 2, 1] {
            assert_eq!(m.push(t), StopOutcome::Continue);
        }
        assert_eq!(m.push(2), StopOutcome::Match { pattern: 0 });
    }

    #[test]
    fn suffix_match_without_replaying_window() {
        // Pattern aaa: stream aaaa matches at the 3rd token (suffix "aaa").
        let mut m = StopMatcher::new(vec![vec![5, 5, 5]]);
        assert_eq!(m.push(5), StopOutcome::Continue);
        assert_eq!(m.push(5), StopOutcome::Continue);
        assert_eq!(m.push(5), StopOutcome::Match { pattern: 0 });
    }

    #[test]
    fn overlapping_patterns_and_suffix_patterns() {
        // "bc" is a suffix of "abc"; both must work independently.
        let mut m = StopMatcher::new(vec![vec![1, 2, 3], vec![2, 3]]);
        assert_eq!(m.push(1), StopOutcome::Continue);
        assert_eq!(m.push(2), StopOutcome::Continue);
        assert_eq!(m.push(3), StopOutcome::Match { pattern: 0 });
        let mut m = StopMatcher::new(vec![vec![1, 2, 3], vec![2, 3]]);
        assert_eq!(m.push(2), StopOutcome::Continue);
        assert_eq!(m.push(3), StopOutcome::Match { pattern: 1 }, "shorter pattern fires");
    }

    #[test]
    fn empty_patterns_are_ignored() {
        let mut m = StopMatcher::new(vec![vec![], vec![4]]);
        assert_eq!(m.pattern_count(), 1);
        assert_eq!(m.push(4), StopOutcome::Match { pattern: 0 });
    }

    #[test]
    fn reset_clears_partial_match() {
        let mut m = StopMatcher::new(vec![vec![2, 2]]);
        assert_eq!(m.push(2), StopOutcome::Continue);
        m.reset();
        assert_eq!(m.push(1), StopOutcome::Continue);
        assert_eq!(m.push(2), StopOutcome::Continue, "no stale state after reset");
    }

    #[test]
    fn long_stream_keeps_matching_at_edges() {
        // Pattern with length 1 must keep firing at the right token.
        let mut m = StopMatcher::new(vec![vec![9]]);
        for t in 0..100u32 {
            let expect =
                if t == 9 { StopOutcome::Match { pattern: 0 } } else { StopOutcome::Continue };
            assert_eq!(m.push(t), expect, "token {t}");
        }
    }
}
