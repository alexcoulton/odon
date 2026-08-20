use super::{
    GeoJsonObjectFeature, ObjectPropertyColumn, ObjectPropertyStore, column_value_to_display_text,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ObjectFilterQueryExpr {
    And(Box<ObjectFilterQueryExpr>, Box<ObjectFilterQueryExpr>),
    Or(Box<ObjectFilterQueryExpr>, Box<ObjectFilterQueryExpr>),
    Not(Box<ObjectFilterQueryExpr>),
    Predicate(ObjectFilterPredicate),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ObjectFilterPredicate {
    pub property_key: String,
    pub op: ObjectFilterQueryOp,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ObjectFilterQueryOp {
    Truthy,
    Eq(ObjectFilterQueryValue),
    Ne(ObjectFilterQueryValue),
    Gt(ObjectFilterQueryValue),
    Ge(ObjectFilterQueryValue),
    Lt(ObjectFilterQueryValue),
    Le(ObjectFilterQueryValue),
    Contains(ObjectFilterQueryValue),
    StartsWith(ObjectFilterQueryValue),
    EndsWith(ObjectFilterQueryValue),
    In(Vec<ObjectFilterQueryValue>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ObjectFilterQueryValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ObjectFilterQueryError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for ObjectFilterQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at character {}", self.message, self.position + 1)
    }
}

impl ObjectFilterQueryExpr {
    pub(super) fn parse(input: &str) -> Result<Self, ObjectFilterQueryError> {
        let tokens = lex(input)?;
        let mut parser = Parser { tokens, cursor: 0 };
        let expr = parser.parse_or()?;
        if let Some(token) = parser.peek() {
            return Err(ObjectFilterQueryError {
                message: format!("unexpected token '{}'", token.display_text()),
                position: token.span.start,
            });
        }
        Ok(expr)
    }

    pub(super) fn referenced_properties(&self) -> Vec<String> {
        let mut out = BTreeSet::new();
        self.collect_referenced_properties(&mut out);
        out.into_iter().collect()
    }

    fn collect_referenced_properties(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::And(left, right) | Self::Or(left, right) => {
                left.collect_referenced_properties(out);
                right.collect_referenced_properties(out);
            }
            Self::Not(inner) => inner.collect_referenced_properties(out),
            Self::Predicate(predicate) => {
                if predicate.property_key != "id" {
                    out.insert(predicate.property_key.clone());
                }
            }
        }
    }

    pub(super) fn matches(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
        properties: &ObjectPropertyStore,
    ) -> bool {
        match self {
            Self::And(left, right) => {
                left.matches(object_index, obj, properties)
                    && right.matches(object_index, obj, properties)
            }
            Self::Or(left, right) => {
                left.matches(object_index, obj, properties)
                    || right.matches(object_index, obj, properties)
            }
            Self::Not(inner) => !inner.matches(object_index, obj, properties),
            Self::Predicate(predicate) => predicate.matches(object_index, obj, properties),
        }
    }
}

impl ObjectFilterPredicate {
    fn matches(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
        properties: &ObjectPropertyStore,
    ) -> bool {
        let actual = object_query_value(object_index, obj, properties, &self.property_key);
        match &self.op {
            ObjectFilterQueryOp::Truthy => actual.is_truthy(),
            ObjectFilterQueryOp::Eq(expected) => actual.equals_query_value(expected),
            ObjectFilterQueryOp::Ne(expected) => !actual.equals_query_value(expected),
            ObjectFilterQueryOp::Gt(expected) => actual
                .compare_query_value(expected)
                .is_some_and(|ord| ord > 0),
            ObjectFilterQueryOp::Ge(expected) => actual
                .compare_query_value(expected)
                .is_some_and(|ord| ord >= 0),
            ObjectFilterQueryOp::Lt(expected) => actual
                .compare_query_value(expected)
                .is_some_and(|ord| ord < 0),
            ObjectFilterQueryOp::Le(expected) => actual
                .compare_query_value(expected)
                .is_some_and(|ord| ord <= 0),
            ObjectFilterQueryOp::Contains(expected) => {
                actual.string_matches(expected, |actual, expected| actual.contains(expected))
            }
            ObjectFilterQueryOp::StartsWith(expected) => {
                actual.string_matches(expected, |actual, expected| actual.starts_with(expected))
            }
            ObjectFilterQueryOp::EndsWith(expected) => {
                actual.string_matches(expected, |actual, expected| actual.ends_with(expected))
            }
            ObjectFilterQueryOp::In(expected_values) => expected_values
                .iter()
                .any(|expected| actual.equals_query_value(expected)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ObjectFilterActualValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
    Missing,
}

impl ObjectFilterActualValue {
    fn is_truthy(&self) -> bool {
        match self {
            Self::String(value) => !value.trim().is_empty(),
            Self::Number(value) => value.is_finite() && *value != 0.0,
            Self::Bool(value) => *value,
            Self::Null | Self::Missing => false,
        }
    }

    fn equals_query_value(&self, expected: &ObjectFilterQueryValue) -> bool {
        match (self, expected) {
            (Self::Missing, ObjectFilterQueryValue::Null)
            | (Self::Null, ObjectFilterQueryValue::Null) => true,
            (Self::Bool(actual), ObjectFilterQueryValue::Bool(expected)) => actual == expected,
            (Self::Number(actual), ObjectFilterQueryValue::Number(expected)) => {
                actual.is_finite() && expected.is_finite() && actual == expected
            }
            (Self::String(actual), ObjectFilterQueryValue::String(expected)) => {
                actual.trim().eq_ignore_ascii_case(expected.trim())
            }
            (Self::Bool(actual), ObjectFilterQueryValue::String(expected)) => {
                actual.to_string().eq_ignore_ascii_case(expected.trim())
            }
            (Self::Number(actual), ObjectFilterQueryValue::String(expected)) => expected
                .trim()
                .parse::<f64>()
                .ok()
                .is_some_and(|expected| actual.is_finite() && actual == &expected),
            (Self::String(actual), ObjectFilterQueryValue::Bool(expected)) => {
                actual.trim().eq_ignore_ascii_case(&expected.to_string())
            }
            (Self::String(actual), ObjectFilterQueryValue::Number(expected)) => actual
                .trim()
                .parse::<f64>()
                .ok()
                .is_some_and(|actual| expected.is_finite() && actual == *expected),
            _ => false,
        }
    }

    fn compare_query_value(&self, expected: &ObjectFilterQueryValue) -> Option<i8> {
        let actual = self.as_number()?;
        let expected = match expected {
            ObjectFilterQueryValue::Number(value) => *value,
            ObjectFilterQueryValue::String(value) => value.trim().parse::<f64>().ok()?,
            _ => return None,
        };
        if !actual.is_finite() || !expected.is_finite() {
            return None;
        }
        Some(if actual < expected {
            -1
        } else if actual > expected {
            1
        } else {
            0
        })
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::String(value) => value.trim().parse::<f64>().ok(),
            Self::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
            Self::Null | Self::Missing => None,
        }
    }

    fn string_matches(
        &self,
        expected: &ObjectFilterQueryValue,
        predicate: impl FnOnce(&str, &str) -> bool,
    ) -> bool {
        let Some(actual) = self.display_text() else {
            return false;
        };
        let expected = match expected {
            ObjectFilterQueryValue::String(value) => value.clone(),
            ObjectFilterQueryValue::Number(value) => value.to_string(),
            ObjectFilterQueryValue::Bool(value) => value.to_string(),
            ObjectFilterQueryValue::Null => return false,
        };
        predicate(
            actual.to_ascii_lowercase().as_str(),
            expected.trim().to_ascii_lowercase().as_str(),
        )
    }

    fn display_text(&self) -> Option<String> {
        match self {
            Self::String(value) => Some(value.clone()),
            Self::Number(value) => Some(value.to_string()),
            Self::Bool(value) => Some(value.to_string()),
            Self::Null | Self::Missing => None,
        }
    }
}

fn object_query_value(
    object_index: usize,
    obj: &GeoJsonObjectFeature,
    properties: &ObjectPropertyStore,
    property_key: &str,
) -> ObjectFilterActualValue {
    if property_key == "id" {
        return ObjectFilterActualValue::String(obj.id.clone());
    }
    if let Some(column) = properties.loaded_columns.get(property_key) {
        return column_query_value(column, object_index);
    }
    match obj.inline_properties.get(property_key) {
        Some(value) => json_query_value(value),
        None => ObjectFilterActualValue::Missing,
    }
}

fn column_query_value(
    column: &ObjectPropertyColumn,
    object_index: usize,
) -> ObjectFilterActualValue {
    match column {
        ObjectPropertyColumn::Bool(values) => values
            .get(object_index)
            .and_then(|value| *value)
            .map(ObjectFilterActualValue::Bool)
            .unwrap_or(ObjectFilterActualValue::Null),
        ObjectPropertyColumn::I64(values) => values
            .get(object_index)
            .and_then(|value| *value)
            .map(|value| ObjectFilterActualValue::Number(value as f64))
            .unwrap_or(ObjectFilterActualValue::Null),
        ObjectPropertyColumn::F64(values) => values
            .get(object_index)
            .and_then(|value| *value)
            .map(ObjectFilterActualValue::Number)
            .unwrap_or(ObjectFilterActualValue::Null),
        ObjectPropertyColumn::Dictionary { dictionary, values } => values
            .get(object_index)
            .and_then(|code| *code)
            .and_then(|code| dictionary.get(code as usize).cloned())
            .map(ObjectFilterActualValue::String)
            .unwrap_or(ObjectFilterActualValue::Null),
        ObjectPropertyColumn::Json(values) => values
            .get(object_index)
            .and_then(|value| value.as_ref())
            .map(json_query_value)
            .unwrap_or(ObjectFilterActualValue::Null),
    }
}

fn json_query_value(value: &serde_json::Value) -> ObjectFilterActualValue {
    match value {
        serde_json::Value::Null => ObjectFilterActualValue::Null,
        serde_json::Value::Bool(value) => ObjectFilterActualValue::Bool(*value),
        serde_json::Value::Number(value) => value
            .as_f64()
            .map(ObjectFilterActualValue::Number)
            .unwrap_or_else(|| ObjectFilterActualValue::String(value.to_string())),
        serde_json::Value::String(value) => ObjectFilterActualValue::String(value.clone()),
        other => ObjectFilterActualValue::String(column_value_to_display_text(other)),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokenKind,
    span: std::ops::Range<usize>,
}

impl Token {
    fn display_text(&self) -> String {
        self.kind.display_text()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Ident(String),
    String(String),
    Number(f64),
    Bool(bool),
    Null,
    And,
    Or,
    Not,
    In,
    Contains,
    StartsWith,
    EndsWith,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
}

impl TokenKind {
    fn display_text(&self) -> String {
        match self {
            Self::Ident(value) => value.clone(),
            Self::String(value) => format!("\"{value}\""),
            Self::Number(value) => value.to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Null => "null".to_string(),
            Self::And => "and".to_string(),
            Self::Or => "or".to_string(),
            Self::Not => "not".to_string(),
            Self::In => "in".to_string(),
            Self::Contains => "contains".to_string(),
            Self::StartsWith => "starts_with".to_string(),
            Self::EndsWith => "ends_with".to_string(),
            Self::Eq => "==".to_string(),
            Self::Ne => "!=".to_string(),
            Self::Gt => ">".to_string(),
            Self::Ge => ">=".to_string(),
            Self::Lt => "<".to_string(),
            Self::Le => "<=".to_string(),
            Self::LParen => "(".to_string(),
            Self::RParen => ")".to_string(),
            Self::LBracket => "[".to_string(),
            Self::RBracket => "]".to_string(),
            Self::Comma => ",".to_string(),
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    fn parse_or(&mut self) -> Result<ObjectFilterQueryExpr, ObjectFilterQueryError> {
        let mut expr = self.parse_and()?;
        while self.eat(|kind| matches!(kind, TokenKind::Or)).is_some() {
            let right = self.parse_and()?;
            expr = ObjectFilterQueryExpr::Or(Box::new(expr), Box::new(right));
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<ObjectFilterQueryExpr, ObjectFilterQueryError> {
        let mut expr = self.parse_unary()?;
        while self.eat(|kind| matches!(kind, TokenKind::And)).is_some() {
            let right = self.parse_unary()?;
            expr = ObjectFilterQueryExpr::And(Box::new(expr), Box::new(right));
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<ObjectFilterQueryExpr, ObjectFilterQueryError> {
        if self.eat(|kind| matches!(kind, TokenKind::Not)).is_some() {
            return Ok(ObjectFilterQueryExpr::Not(Box::new(self.parse_unary()?)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<ObjectFilterQueryExpr, ObjectFilterQueryError> {
        if self.eat(|kind| matches!(kind, TokenKind::LParen)).is_some() {
            let expr = self.parse_or()?;
            self.expect(|kind| matches!(kind, TokenKind::RParen), "')'")?;
            return Ok(expr);
        }
        self.parse_predicate()
    }

    fn parse_predicate(&mut self) -> Result<ObjectFilterQueryExpr, ObjectFilterQueryError> {
        let property = match self.next() {
            Some(Token {
                kind: TokenKind::Ident(property),
                ..
            }) => property,
            Some(token) => {
                return Err(ObjectFilterQueryError {
                    message: format!("expected property name, found '{}'", token.display_text()),
                    position: token.span.start,
                });
            }
            None => return self.unexpected_end("property name"),
        };

        let Some(op_token) = self.peek().cloned() else {
            return Ok(ObjectFilterQueryExpr::Predicate(ObjectFilterPredicate {
                property_key: property,
                op: ObjectFilterQueryOp::Truthy,
            }));
        };

        let op = match op_token.kind {
            TokenKind::Eq => {
                self.next();
                ObjectFilterQueryOp::Eq(self.parse_value()?)
            }
            TokenKind::Ne => {
                self.next();
                ObjectFilterQueryOp::Ne(self.parse_value()?)
            }
            TokenKind::Gt => {
                self.next();
                ObjectFilterQueryOp::Gt(self.parse_value()?)
            }
            TokenKind::Ge => {
                self.next();
                ObjectFilterQueryOp::Ge(self.parse_value()?)
            }
            TokenKind::Lt => {
                self.next();
                ObjectFilterQueryOp::Lt(self.parse_value()?)
            }
            TokenKind::Le => {
                self.next();
                ObjectFilterQueryOp::Le(self.parse_value()?)
            }
            TokenKind::Contains => {
                self.next();
                ObjectFilterQueryOp::Contains(self.parse_value()?)
            }
            TokenKind::StartsWith => {
                self.next();
                ObjectFilterQueryOp::StartsWith(self.parse_value()?)
            }
            TokenKind::EndsWith => {
                self.next();
                ObjectFilterQueryOp::EndsWith(self.parse_value()?)
            }
            TokenKind::In => {
                self.next();
                ObjectFilterQueryOp::In(self.parse_value_list()?)
            }
            _ => ObjectFilterQueryOp::Truthy,
        };

        Ok(ObjectFilterQueryExpr::Predicate(ObjectFilterPredicate {
            property_key: property,
            op,
        }))
    }

    fn parse_value_list(&mut self) -> Result<Vec<ObjectFilterQueryValue>, ObjectFilterQueryError> {
        let (open, close) = match self.peek() {
            Some(Token {
                kind: TokenKind::LBracket,
                ..
            }) => {
                self.next();
                ("[", "]")
            }
            Some(Token {
                kind: TokenKind::LParen,
                ..
            }) => {
                self.next();
                ("(", ")")
            }
            Some(token) => {
                return Err(ObjectFilterQueryError {
                    message: format!("expected '[' after 'in', found '{}'", token.display_text()),
                    position: token.span.start,
                });
            }
            None => return self.unexpected_end("'['"),
        };

        let mut values = Vec::new();
        if self
            .eat(|kind| {
                matches!(
                    (open, kind),
                    ("[", TokenKind::RBracket) | ("(", TokenKind::RParen)
                )
            })
            .is_some()
        {
            return Ok(values);
        }

        loop {
            values.push(self.parse_value()?);
            if self.eat(|kind| matches!(kind, TokenKind::Comma)).is_none() {
                break;
            }
        }
        self.expect(
            |kind| {
                matches!(
                    (close, kind),
                    ("]", TokenKind::RBracket) | (")", TokenKind::RParen)
                )
            },
            close,
        )?;
        Ok(values)
    }

    fn parse_value(&mut self) -> Result<ObjectFilterQueryValue, ObjectFilterQueryError> {
        match self.next() {
            Some(Token {
                kind: TokenKind::String(value),
                ..
            }) => Ok(ObjectFilterQueryValue::String(value)),
            Some(Token {
                kind: TokenKind::Ident(value),
                ..
            }) => Ok(ObjectFilterQueryValue::String(value)),
            Some(Token {
                kind: TokenKind::Number(value),
                ..
            }) => Ok(ObjectFilterQueryValue::Number(value)),
            Some(Token {
                kind: TokenKind::Bool(value),
                ..
            }) => Ok(ObjectFilterQueryValue::Bool(value)),
            Some(Token {
                kind: TokenKind::Null,
                ..
            }) => Ok(ObjectFilterQueryValue::Null),
            Some(token) => Err(ObjectFilterQueryError {
                message: format!("expected value, found '{}'", token.display_text()),
                position: token.span.start,
            }),
            None => self.unexpected_end("value"),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.cursor).cloned()?;
        self.cursor += 1;
        Some(token)
    }

    fn eat(&mut self, pred: impl FnOnce(&TokenKind) -> bool) -> Option<Token> {
        if self.peek().is_some_and(|token| pred(&token.kind)) {
            self.next()
        } else {
            None
        }
    }

    fn expect(
        &mut self,
        pred: impl FnOnce(&TokenKind) -> bool,
        expected: &str,
    ) -> Result<Token, ObjectFilterQueryError> {
        match self.next() {
            Some(token) if pred(&token.kind) => Ok(token),
            Some(token) => Err(ObjectFilterQueryError {
                message: format!("expected {expected}, found '{}'", token.display_text()),
                position: token.span.start,
            }),
            None => self.unexpected_end(expected),
        }
    }

    fn unexpected_end<T>(&self, expected: &str) -> Result<T, ObjectFilterQueryError> {
        let position = self.tokens.last().map_or(0, |token| token.span.end);
        Err(ObjectFilterQueryError {
            message: format!("expected {expected}, found end of query"),
            position,
        })
    }
}

fn lex(input: &str) -> Result<Vec<Token>, ObjectFilterQueryError> {
    let chars = input.char_indices().collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < chars.len() {
        let (start, ch) = chars[idx];
        if ch.is_whitespace() {
            idx += 1;
            continue;
        }
        match ch {
            '(' => {
                out.push(token(TokenKind::LParen, start, start + ch.len_utf8()));
                idx += 1;
            }
            ')' => {
                out.push(token(TokenKind::RParen, start, start + ch.len_utf8()));
                idx += 1;
            }
            '[' => {
                out.push(token(TokenKind::LBracket, start, start + ch.len_utf8()));
                idx += 1;
            }
            ']' => {
                out.push(token(TokenKind::RBracket, start, start + ch.len_utf8()));
                idx += 1;
            }
            ',' => {
                out.push(token(TokenKind::Comma, start, start + ch.len_utf8()));
                idx += 1;
            }
            '=' => {
                let end = if chars.get(idx + 1).is_some_and(|(_, next)| *next == '=') {
                    idx += 2;
                    chars
                        .get(idx)
                        .map_or(input.len(), |(next_start, _)| *next_start)
                } else {
                    idx += 1;
                    start + ch.len_utf8()
                };
                out.push(token(TokenKind::Eq, start, end));
            }
            '!' => {
                if chars.get(idx + 1).is_some_and(|(_, next)| *next == '=') {
                    idx += 2;
                    let end = chars
                        .get(idx)
                        .map_or(input.len(), |(next_start, _)| *next_start);
                    out.push(token(TokenKind::Ne, start, end));
                } else {
                    return Err(ObjectFilterQueryError {
                        message: "expected '!='".to_string(),
                        position: start,
                    });
                }
            }
            '>' => {
                if chars.get(idx + 1).is_some_and(|(_, next)| *next == '=') {
                    idx += 2;
                    let end = chars
                        .get(idx)
                        .map_or(input.len(), |(next_start, _)| *next_start);
                    out.push(token(TokenKind::Ge, start, end));
                } else {
                    idx += 1;
                    out.push(token(TokenKind::Gt, start, start + ch.len_utf8()));
                }
            }
            '<' => {
                if chars.get(idx + 1).is_some_and(|(_, next)| *next == '=') {
                    idx += 2;
                    let end = chars
                        .get(idx)
                        .map_or(input.len(), |(next_start, _)| *next_start);
                    out.push(token(TokenKind::Le, start, end));
                } else {
                    idx += 1;
                    out.push(token(TokenKind::Lt, start, start + ch.len_utf8()));
                }
            }
            '"' | '\'' => {
                let (value, next_idx) = lex_quoted(input, &chars, idx, ch)?;
                let end = chars
                    .get(next_idx)
                    .map_or(input.len(), |(next_start, _)| *next_start);
                out.push(token(TokenKind::String(value), start, end));
                idx = next_idx;
            }
            '`' => {
                let (value, next_idx) = lex_quoted(input, &chars, idx, ch)?;
                let end = chars
                    .get(next_idx)
                    .map_or(input.len(), |(next_start, _)| *next_start);
                out.push(token(TokenKind::Ident(value), start, end));
                idx = next_idx;
            }
            '-' | '0'..='9' => {
                if let Some((number, next_idx)) = lex_number(input, &chars, idx) {
                    let end = chars
                        .get(next_idx)
                        .map_or(input.len(), |(next_start, _)| *next_start);
                    out.push(token(TokenKind::Number(number), start, end));
                    idx = next_idx;
                } else if ch == '-' {
                    return Err(ObjectFilterQueryError {
                        message: "expected number after '-'".to_string(),
                        position: start,
                    });
                } else {
                    return Err(ObjectFilterQueryError {
                        message: format!("unexpected character '{ch}'"),
                        position: start,
                    });
                }
            }
            _ if is_ident_start(ch) => {
                let mut end_idx = idx + 1;
                while chars
                    .get(end_idx)
                    .is_some_and(|(_, next)| is_ident_continue(*next))
                {
                    end_idx += 1;
                }
                let end = chars
                    .get(end_idx)
                    .map_or(input.len(), |(next_start, _)| *next_start);
                let text = &input[start..end];
                out.push(token(keyword_or_ident(text), start, end));
                idx = end_idx;
            }
            _ => {
                return Err(ObjectFilterQueryError {
                    message: format!("unexpected character '{ch}'"),
                    position: start,
                });
            }
        }
    }
    Ok(out)
}

fn token(kind: TokenKind, start: usize, end: usize) -> Token {
    Token {
        kind,
        span: start..end,
    }
}

fn keyword_or_ident(text: &str) -> TokenKind {
    match text.to_ascii_lowercase().as_str() {
        "and" => TokenKind::And,
        "or" => TokenKind::Or,
        "not" => TokenKind::Not,
        "in" => TokenKind::In,
        "contains" => TokenKind::Contains,
        "starts_with" | "startswith" => TokenKind::StartsWith,
        "ends_with" | "endswith" => TokenKind::EndsWith,
        "true" => TokenKind::Bool(true),
        "false" => TokenKind::Bool(false),
        "null" | "none" => TokenKind::Null,
        _ => TokenKind::Ident(text.to_string()),
    }
}

fn lex_quoted(
    input: &str,
    chars: &[(usize, char)],
    start_idx: usize,
    quote: char,
) -> Result<(String, usize), ObjectFilterQueryError> {
    let (start, _) = chars[start_idx];
    let mut out = String::new();
    let mut idx = start_idx + 1;
    while let Some((_, ch)) = chars.get(idx).copied() {
        if ch == quote {
            return Ok((out, idx + 1));
        }
        if ch == '\\' {
            let Some((_, escaped)) = chars.get(idx + 1).copied() else {
                return Err(ObjectFilterQueryError {
                    message: "unterminated escape sequence".to_string(),
                    position: input.len().saturating_sub(1),
                });
            };
            out.push(match escaped {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                '`' => '`',
                other => other,
            });
            idx += 2;
            continue;
        }
        out.push(ch);
        idx += 1;
    }
    Err(ObjectFilterQueryError {
        message: format!("unterminated {quote}-quoted string"),
        position: start,
    })
}

fn lex_number(input: &str, chars: &[(usize, char)], start_idx: usize) -> Option<(f64, usize)> {
    let mut idx = start_idx;
    if chars.get(idx).is_some_and(|(_, ch)| *ch == '-') {
        idx += 1;
    }
    let digit_start = idx;
    while chars.get(idx).is_some_and(|(_, ch)| ch.is_ascii_digit()) {
        idx += 1;
    }
    if chars.get(idx).is_some_and(|(_, ch)| *ch == '.') {
        idx += 1;
        while chars.get(idx).is_some_and(|(_, ch)| ch.is_ascii_digit()) {
            idx += 1;
        }
    }
    if idx == digit_start {
        return None;
    }
    if chars
        .get(idx)
        .is_some_and(|(_, ch)| matches!(*ch, 'e' | 'E'))
    {
        let exp_idx = idx;
        idx += 1;
        if chars
            .get(idx)
            .is_some_and(|(_, ch)| matches!(*ch, '+' | '-'))
        {
            idx += 1;
        }
        let exp_digit_start = idx;
        while chars.get(idx).is_some_and(|(_, ch)| ch.is_ascii_digit()) {
            idx += 1;
        }
        if idx == exp_digit_start {
            idx = exp_idx;
        }
    }
    let start = chars[start_idx].0;
    let end = chars
        .get(idx)
        .map_or(input.len(), |(next_start, _)| *next_start);
    input[start..end]
        .parse::<f64>()
        .ok()
        .map(|value| (value, idx))
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '.' || ch.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui;
    use serde_json::json;
    use std::sync::Arc;

    fn object(id: &str) -> GeoJsonObjectFeature {
        GeoJsonObjectFeature {
            id: id.to_string(),
            polygons_world: Vec::new(),
            point_position_world: None,
            bbox_world: egui::Rect::NOTHING,
            area_px: 0.0,
            perimeter_px: 0.0,
            centroid_world: egui::Pos2::ZERO,
            inline_properties: serde_json::Map::new(),
            source_row_index: None,
        }
    }

    fn store() -> ObjectPropertyStore {
        let mut store = ObjectPropertyStore::default();
        store.insert_column(
            "broad_cell_type".to_string(),
            ObjectPropertyColumn::from_json_values(vec![
                Some(json!("immune_lymphoid")),
                Some(json!("immune_myeloid")),
                Some(json!("tumor_myogenic")),
            ]),
        );
        store.insert_column(
            "zz_mask_cd3".to_string(),
            ObjectPropertyColumn::Bool(Arc::new(vec![Some(true), Some(false), Some(false)])),
        );
        store.insert_column(
            "zz_mask_hla_dr".to_string(),
            ObjectPropertyColumn::Bool(Arc::new(vec![Some(false), Some(true), Some(false)])),
        );
        store.insert_column(
            "median_intensity_cd3".to_string(),
            ObjectPropertyColumn::F64(Arc::new(vec![Some(1500.0), Some(200.0), Some(50.0)])),
        );
        store
    }

    #[test]
    fn parses_and_evaluates_phenotype_union_query() {
        let expr = ObjectFilterQueryExpr::parse(
            "(broad_cell_type == \"immune_lymphoid\" and zz_mask_cd3) or \
             (broad_cell_type == \"immune_myeloid\" and zz_mask_hla_dr)",
        )
        .unwrap();
        let store = store();
        let obj = object("cell");

        assert!(expr.matches(0, &obj, &store));
        assert!(expr.matches(1, &obj, &store));
        assert!(!expr.matches(2, &obj, &store));
    }

    #[test]
    fn supports_numeric_comparisons_and_not() {
        let expr =
            ObjectFilterQueryExpr::parse("not zz_mask_cd3 and median_intensity_cd3 < 500").unwrap();
        let store = store();
        let obj = object("cell");

        assert!(!expr.matches(0, &obj, &store));
        assert!(expr.matches(1, &obj, &store));
    }

    #[test]
    fn supports_in_and_contains() {
        let expr = ObjectFilterQueryExpr::parse(
            "broad_cell_type in [\"immune_lymphoid\", \"immune_myeloid\"] and id contains cell",
        )
        .unwrap();
        let store = store();
        let obj = object("Cell 42");

        assert!(expr.matches(0, &obj, &store));
        assert!(expr.matches(1, &obj, &store));
        assert!(!expr.matches(2, &obj, &store));
    }

    #[test]
    fn supports_backtick_property_names() {
        let expr = ObjectFilterQueryExpr::parse("`marker with space` == true").unwrap();
        assert_eq!(
            expr.referenced_properties(),
            vec!["marker with space".to_string()]
        );
    }

    #[test]
    fn reports_syntax_errors_with_position() {
        let err = ObjectFilterQueryExpr::parse("(broad_cell_type == immune").unwrap_err();
        assert!(err.message.contains("expected ')'"));
        assert!(err.position > 0);
    }
}
