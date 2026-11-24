//! Exact finite-decimal arithmetic on digit strings. JSON Schema `multipleOf`
//! must not rely on a floating-point remainder (0.3 is a multiple of 0.1).

use crate::error::{Result, invalid};

/// `digits × 10^-scale`, normalized: no leading or trailing zeros, and zero
/// is `"0"` with scale 0 and no sign.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decimal {
    pub digits: Vec<u8>,
    pub scale: i64,
    pub negative: bool,
}

impl Default for Decimal {
    fn default() -> Self {
        Self::integer(b"0")
    }
}

impl Decimal {
    pub fn integer(digits: &[u8]) -> Self {
        Self {
            digits: digits.to_vec(),
            scale: 0,
            negative: false,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.digits == b"0"
    }

    /// Reads a JSON number, or the prefix of one the grammar produced.
    pub fn parse(text: &[u8]) -> Result<Self> {
        let mut value = Decimal {
            digits: Vec::with_capacity(text.len()),
            scale: 0,
            negative: false,
        };
        let mut text = text;
        if let [b'-', rest @ ..] = text {
            value.negative = true;
            text = rest;
        }
        let mut fraction = false;
        let mut i = 0;
        while i < text.len() && text[i] != b'e' && text[i] != b'E' {
            if text[i] == b'.' {
                fraction = true;
            } else {
                value.digits.push(text[i]);
                value.scale += i64::from(fraction);
            }
            i += 1;
        }
        if i < text.len() {
            i += 1;
            if text.get(i) == Some(&b'+') {
                i += 1;
            }
            let Some(exponent) = exponent(&text[i..]) else {
                return invalid("numeric exponent is outside the supported range");
            };
            value.scale -= i64::from(exponent);
        }
        value.normalize();
        Ok(value)
    }

    pub fn normalize(&mut self) {
        let Some(first) = self.digits.iter().position(|&d| d != b'0') else {
            *self = Self::default();
            return;
        };
        self.digits.drain(..first);
        while self.digits.len() > 1 && self.digits.last() == Some(&b'0') {
            self.digits.pop();
            self.scale -= 1;
        }
    }

    /// The decimal exponent of the leading digit plus one.
    fn exponent(&self) -> i64 {
        self.digits.len() as i64 - self.scale
    }
}

/// The leading `int` of `text`, as `std::from_chars` reads it: an optional
/// minus sign and at least one digit, within 32 bits.
fn exponent(text: &[u8]) -> Option<i32> {
    let (negative, digits) = match text {
        [b'-', rest @ ..] => (true, rest),
        _ => (false, text),
    };
    let count = digits.iter().take_while(|d| d.is_ascii_digit()).count();
    if count == 0 {
        return None;
    }
    let mut value: i64 = 0;
    for &digit in &digits[..count] {
        value = value * 10 + i64::from(digit - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    i32::try_from(if negative { -value } else { value }).ok()
}

fn zeros(count: i64) -> impl Iterator<Item = u8> {
    std::iter::repeat_n(b'0', usize::try_from(count).unwrap_or(0))
}

fn sign(ordering: std::cmp::Ordering) -> i32 {
    ordering as i32
}

pub fn magnitude(a: &Decimal, b: &Decimal) -> i32 {
    if a.is_zero() || b.is_zero() {
        return i32::from(!a.is_zero()) - i32::from(!b.is_zero());
    }
    let (ae, be) = (a.exponent(), b.exponent());
    if ae != be {
        return if ae < be { -1 } else { 1 };
    }
    for i in 0..a.digits.len().max(b.digits.len()) {
        let x = a.digits.get(i).copied().unwrap_or(b'0');
        let y = b.digits.get(i).copied().unwrap_or(b'0');
        if x != y {
            return if x < y { -1 } else { 1 };
        }
    }
    0
}

pub fn compare(a: &Decimal, b: &Decimal) -> i32 {
    if a.negative != b.negative {
        return if a.negative { -1 } else { 1 };
    }
    (if a.negative { -1 } else { 1 }) * magnitude(a, b)
}

fn natural_compare(a: &[u8], b: &[u8]) -> i32 {
    if a.len() != b.len() {
        return if a.len() < b.len() { -1 } else { 1 };
    }
    sign(a.cmp(b))
}

fn strip_leading_zeros(a: &mut Vec<u8>) {
    match a.iter().position(|&d| d != b'0') {
        Some(first) => {
            a.drain(..first);
        }
        None => *a = b"0".to_vec(),
    }
}

/// `a -= b` for natural numbers with `a >= b`.
fn subtract(a: &mut Vec<u8>, b: &[u8]) {
    let mut borrow = 0i32;
    let n = a.len();
    for i in 0..n {
        let mut digit = i32::from(a[n - 1 - i]) - i32::from(b'0') - borrow;
        if i < b.len() {
            digit -= i32::from(b[b.len() - 1 - i]) - i32::from(b'0');
        }
        borrow = i32::from(digit < 0);
        a[n - 1 - i] = (i32::from(b'0') + digit + 10 * borrow) as u8;
    }
    strip_leading_zeros(a);
}

/// The integer quotient and remainder of `|value| / |divisor|`, both scaled to
/// their common decimal grid.
pub fn divide(value: &Decimal, divisor: &Decimal) -> (Vec<u8>, Vec<u8>) {
    let scale = value.scale.max(divisor.scale);
    let numerator: Vec<u8> = value
        .digits
        .iter()
        .copied()
        .chain(zeros(scale - value.scale))
        .collect();
    let denominator: Vec<u8> = divisor
        .digits
        .iter()
        .copied()
        .chain(zeros(scale - divisor.scale))
        .collect();
    let mut remainder = b"0".to_vec();
    let mut quotient = Vec::new();
    for digit in numerator {
        if remainder == b"0" {
            remainder.clear();
        }
        remainder.push(digit);
        let mut count = 0u8;
        while natural_compare(&remainder, &denominator) >= 0 {
            subtract(&mut remainder, &denominator);
            count += 1;
        }
        if count != 0 || !quotient.is_empty() {
            quotient.push(b'0' + count);
        }
    }
    if quotient.is_empty() {
        quotient = b"0".to_vec();
    }
    (quotient, remainder)
}

pub fn multiple(value: &Decimal, divisor: &Decimal) -> bool {
    value.is_zero() || divide(value, divisor).1 == b"0"
}

pub fn increment(mut value: Vec<u8>) -> Vec<u8> {
    for i in (0..value.len()).rev() {
        if value[i] != b'9' {
            value[i] += 1;
            return value;
        }
        value[i] = b'0';
    }
    value.insert(0, b'1');
    value
}

pub fn product(value: &Decimal, factor: &[u8]) -> Decimal {
    let mut digits = vec![0u32; value.digits.len() + factor.len()];
    for (i, &x) in value.digits.iter().enumerate() {
        for (j, &y) in factor.iter().enumerate() {
            digits[i + j + 1] += u32::from(x - b'0') * u32::from(y - b'0');
        }
    }
    for i in (1..digits.len()).rev() {
        digits[i - 1] += digits[i] / 10;
        digits[i] %= 10;
    }
    let mut result = Decimal {
        digits: digits.iter().map(|&d| b'0' + d as u8).collect(),
        scale: value.scale,
        negative: false,
    };
    result.normalize();
    result
}

/// `|value|` plus one unit in the last of `scale` decimal places.
pub fn add_unit(value: &Decimal, scale: i64) -> Decimal {
    let mut next = value.clone();
    next.negative = false;
    if next.is_zero() {
        return Decimal {
            digits: b"1".to_vec(),
            scale,
            negative: false,
        };
    }
    next.digits.extend(zeros(scale - next.scale));
    next.scale = scale;
    let mut carry = true;
    let mut i = next.digits.len();
    while i > 0 && carry {
        carry = next.digits[i - 1] == b'9';
        next.digits[i - 1] = if carry { b'0' } else { next.digits[i - 1] + 1 };
        i -= 1;
    }
    if carry {
        next.digits.insert(0, b'1');
    }
    next.normalize();
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(text: &str) -> Decimal {
        Decimal::parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn parse_normalizes() {
        assert_eq!(d("-0.000"), Decimal::default());
        assert_eq!(
            d("120.50"),
            Decimal {
                digits: b"1205".to_vec(),
                scale: 1,
                negative: false
            }
        );
        assert_eq!(
            d("1e+22"),
            Decimal {
                digits: b"1".to_vec(),
                scale: -22,
                negative: false
            }
        );
        assert_eq!(d("5e-324").scale, 324);
        assert!(Decimal::parse(b"1e").is_err());
        assert!(Decimal::parse(b"1e99999999999").is_err());
        assert_eq!(d("1e-2147483648").scale, 2147483648);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(compare(&d("0.3"), &d("0.30")), 0);
        assert_eq!(compare(&d("-2"), &d("1")), -1);
        assert_eq!(compare(&d("-2"), &d("-10")), 1);
        assert!(multiple(&d("0.3"), &d("0.1")));
        assert!(!multiple(&d("0.35"), &d("0.1")));
        assert_eq!(divide(&d("7"), &d("2")), (b"3".to_vec(), b"1".to_vec()));
        assert_eq!(increment(b"199".to_vec()), b"200");
        assert_eq!(increment(b"99".to_vec()), b"100");
        assert_eq!(product(&d("1.5"), b"12"), d("18"));
        assert_eq!(add_unit(&d("1.9"), 1), d("2"));
        assert_eq!(add_unit(&d("0"), 3), d("0.001"));
        assert_eq!(add_unit(&d("-99"), 0), d("100"));
    }
}
