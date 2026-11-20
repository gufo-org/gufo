//! The number lexeme: `minimum`, `maximum`, their exclusive forms and
//! `multipleOf`, checked exactly on every prefix the grammar produces.

use crate::decimal::{Decimal, add_unit, compare, divide, increment, multiple, product};
use crate::error::{Result, empty, invalid};
use crate::value::Value;

/// The longest number the grammar produces, in bytes.
const MAX_BYTES: usize = 4096;
/// How far an integer prefix may be extended by appending digits.
const MAX_SHIFTS: u32 = 1024;

#[derive(Clone, Debug)]
struct Bound {
    value: Decimal,
    exclusive: bool,
}

#[derive(Debug)]
pub struct Number {
    integer: bool,
    lower: Option<Bound>,
    upper: Option<Bound>,
    multiple: Option<Decimal>,
    integral_digits: i64,
    fraction_digits: i64,
}

/// A prefix can continue to an accepted number; a complete number is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Match {
    pub prefix: bool,
    pub complete: bool,
}

const fn matched(prefix: bool, complete: bool) -> Match {
    Match { prefix, complete }
}

pub const KEYWORDS: [&str; 5] = [
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

impl Number {
    /// `keywords` holds the values of `KEYWORDS`, in order. Numbers of up to
    /// `digits` integral and fraction digits are produced, more where a
    /// schema number needs them.
    pub fn new(keywords: &[Value<'_>; 5], integer: bool, digits: i64) -> Result<Self> {
        let mut number = Number {
            integer,
            lower: None,
            upper: None,
            multiple: None,
            integral_digits: digits,
            fraction_digits: digits,
        };
        for (key, entry) in KEYWORDS.iter().zip(keywords) {
            if *entry == Value::Absent {
                continue;
            }
            let Some(text) = entry.finite_number() else {
                return invalid(format!("{key} must be a finite number"));
            };
            let value = Decimal::parse(text)?;
            if *key == "multipleOf" {
                if value.negative || value.is_zero() {
                    return invalid("multipleOf must be positive");
                }
                number.multiple = Some(value);
                continue;
            }
            let lower = *key == "minimum" || *key == "exclusiveMinimum";
            let exclusive = key.starts_with("exclusive");
            let bound = if lower {
                &mut number.lower
            } else {
                &mut number.upper
            };
            let comparison = bound.as_ref().map_or(0, |b| compare(&value, &b.value));
            if bound.is_none()
                || (if lower {
                    comparison > 0
                } else {
                    comparison < 0
                })
                || (comparison == 0 && exclusive)
            {
                *bound = Some(Bound { value, exclusive });
            }
        }
        if let (Some(lower), Some(upper)) = (&number.lower, &number.upper) {
            let comparison = compare(&lower.value, &upper.value);
            if comparison > 0
                || (comparison == 0
                    && (lower.exclusive || upper.exclusive || !number.valid(&lower.value)))
            {
                return empty("numeric constraints describe an empty interval");
            }
        }
        if integer {
            let multiple = number
                .multiple
                .get_or_insert_with(|| Decimal::integer(b"1"));
            if multiple.scale > 0 {
                let scale = multiple.scale;
                multiple.scale = 0;
                for factor in [b"2", b"5"] {
                    for _ in 0..scale {
                        let (quotient, remainder) = divide(multiple, &Decimal::integer(factor));
                        if remainder != b"0" {
                            break;
                        }
                        multiple.digits = quotient;
                    }
                }
                multiple.normalize();
            }
        }
        if let (Some(lower), Some(upper), Some(_)) =
            (&number.lower, &number.upper, &number.multiple)
        {
            if !number.grid_point(lower.clone(), upper.clone()) {
                return empty("numeric constraints contain no multipleOf value");
            }
        }
        // Widen the digit caps only for schema numbers that need longer values.
        let values = [
            number.lower.as_ref().map(|b| b.value.clone()),
            number.upper.as_ref().map(|b| b.value.clone()),
            number.multiple.clone(),
        ];
        for value in values.iter().flatten() {
            let integral = value.digits.len() as i64 - value.scale;
            if integral >= number.integral_digits {
                number.integral_digits = integral + 1;
            }
            if value.scale >= number.fraction_digits {
                number.fraction_digits = value.scale + 1;
            }
        }
        Ok(number)
    }

    /// Whether the JSON number spelled `text` is accepted.
    pub fn accept(&self, text: &[u8]) -> Result<bool> {
        Ok(self.valid(&Decimal::parse(text)?))
    }

    pub fn check(&self, bytes: &[u8]) -> Match {
        if bytes.is_empty() {
            return matched(true, false);
        }
        if bytes.len() > MAX_BYTES {
            return Match::default();
        }
        let negative = bytes[0] == b'-';
        let text = if negative { &bytes[1..] } else { bytes };
        let Some(&last) = text.last() else {
            let lower = self.lower.as_ref();
            return matched(
                lower.is_none_or(|l| l.value.negative || (l.value.is_zero() && !l.exclusive)),
                false,
            );
        };
        let dot = text.iter().position(|&c| c == b'.');
        if dot == Some(0) || (self.integer && dot.is_some()) {
            return Match::default();
        }
        if text[0] == b'0' && text.len() > 1 && text[1] != b'.' {
            return Match::default();
        }
        for (i, &c) in text.iter().enumerate() {
            if !c.is_ascii_digit() && (c != b'.' || Some(i) != dot) {
                return Match::default();
            }
        }
        let integral = dot.unwrap_or(text.len()) as i64;
        let places = dot.map_or(0, |d| text.len() - d - 1) as i64;
        if integral > self.integral_digits || places > self.fraction_digits {
            return Match::default();
        }
        // The grammar produces no exponent, so this parse cannot fail.
        let Ok(value) = Decimal::parse(bytes) else {
            return Match::default();
        };
        let complete = last != b'.' && self.valid(&value);
        // At the scalar budget or the decimal cap there is no room for another
        // digit. Do not admit an unfinished prefix which would fail next step.
        if bytes.len() == MAX_BYTES || places == self.fraction_digits {
            return matched(complete, complete);
        }
        // Decimal prefixes cover an interval. Before the decimal point, appending
        // integer digits also shifts that interval by powers of ten.
        let mut low = value;
        let mut high = add_unit(&low, places);
        // The interval [first, last] in signed terms, without copying digits:
        // a negative prefix spans [-high, -low].
        low.negative = negative && !low.is_zero();
        high.negative = negative && !high.is_zero();
        let extend_integer = dot.is_none() && text[0] != b'0';
        for shift in 0..=MAX_SHIFTS {
            let (first, last) = if negative {
                (&high, &low)
            } else {
                (&low, &high)
            };
            let intersects_lower = self.lower.as_ref().is_none_or(|lower| {
                let c = compare(last, &lower.value);
                c > 0 || (negative && !lower.exclusive && c == 0)
            });
            let intersects_upper = self.upper.as_ref().is_none_or(|upper| {
                let c = compare(first, &upper.value);
                c < 0 || (!negative && !upper.exclusive && c == 0)
            });
            if intersects_lower && intersects_upper {
                // Without a step every value in the interval is accepted.
                if self.multiple.is_none() {
                    return matched(true, complete);
                }
                let mut lower = Bound {
                    value: first.clone(),
                    exclusive: negative,
                };
                let mut upper = Bound {
                    value: last.clone(),
                    exclusive: !negative,
                };
                if let Some(bound) = &self.lower {
                    let c = compare(&bound.value, &lower.value);
                    if c > 0 || (c == 0 && bound.exclusive) {
                        lower = bound.clone();
                    }
                }
                if let Some(bound) = &self.upper {
                    let c = compare(&bound.value, &upper.value);
                    if c < 0 || (c == 0 && bound.exclusive) {
                        upper = bound.clone();
                    }
                }
                if self.grid_point(lower, upper) {
                    return matched(true, complete);
                }
            }
            if !extend_integer
                || integral + i64::from(shift) + 1 > self.integral_digits
                || (negative && !intersects_lower)
                || (!negative && !intersects_upper)
            {
                break;
            }
            low.scale -= 1;
            high.scale -= 1;
        }
        matched(complete, complete)
    }

    /// Whether a multiple lies between the bounds.
    fn grid_point(&self, mut low: Bound, mut high: Bound) -> bool {
        let Some(step) = &self.multiple else {
            return true;
        };
        if low.value.negative && !high.value.negative && !high.value.is_zero() {
            return true; // Zero is inside.
        }
        if low.value.negative {
            std::mem::swap(&mut low, &mut high);
            low.value.negative = false;
            high.value.negative = false;
        }
        let (mut quotient, remainder) = divide(&low.value, step);
        if remainder != b"0" || low.exclusive {
            quotient = increment(quotient);
        }
        let candidate = product(step, &quotient);
        let comparison = compare(&candidate, &high.value);
        comparison < 0 || (comparison == 0 && !high.exclusive)
    }

    fn valid(&self, value: &Decimal) -> bool {
        if self.integer && value.scale > 0 {
            return false;
        }
        if let Some(lower) = &self.lower {
            if compare(value, &lower.value) < i32::from(lower.exclusive) {
                return false;
            }
        }
        if let Some(upper) = &self.upper {
            if compare(value, &upper.value) > -i32::from(upper.exclusive) {
                return false;
            }
        }
        self.multiple
            .as_ref()
            .is_none_or(|step| multiple(value, step))
    }
}

/// The least common multiple of two `multipleOf` values, spelled as
/// `digits` `e` `exponent`. The caller checks that a double holds it exactly.
pub fn intersect_multiple_of(a: &Value<'_>, b: &Value<'_>) -> Result<Vec<u8>> {
    let mut texts = [&b""[..]; 2];
    for (text, value) in texts.iter_mut().zip([a, b]) {
        match value.positive_number() {
            Some(found) => *text = found,
            None => return invalid("multipleOf must be a positive finite number"),
        }
    }
    let mut left = Decimal::parse(texts[0])?;
    let mut right = Decimal::parse(texts[1])?;
    // Put both steps on the same integer grid, then compute the exact LCM.
    let scale = left.scale.max(right.scale);
    for value in [&mut left, &mut right] {
        value
            .digits
            .extend(std::iter::repeat_n(b'0', (scale - value.scale) as usize));
        value.scale = 0;
    }
    let (mut x, mut y) = (left.clone(), right.clone());
    while !y.is_zero() {
        let remainder = divide(&x, &y).1;
        x = y;
        y = Decimal::integer(&remainder);
    }
    let mut common = product(&Decimal::integer(&divide(&left, &x).0), &right.digits);
    common.scale += scale;
    common.normalize();
    let mut text = common.digits;
    text.extend_from_slice(format!("e{}", -common.scale).as_bytes());
    Ok(text)
}

/// Whether two JSON numbers have the same exact decimal value.
pub fn equal(a: &[u8], b: &[u8]) -> Result<bool> {
    Ok(compare(&Decimal::parse(a)?, &Decimal::parse(b)?) == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(text: &str) -> Value<'_> {
        Value::Number {
            value: text.parse().unwrap(),
            text: text.as_bytes(),
        }
    }

    fn number(keywords: [Value<'_>; 5], integer: bool) -> Result<Number> {
        Number::new(&keywords, integer, 16)
    }

    const A: Value<'static> = Value::Absent;

    #[test]
    fn bounds_and_steps() {
        let number = number([n("1"), n("10"), A, A, n("0.5")], false).unwrap();
        assert_eq!(number.check(b""), matched(true, false));
        assert_eq!(number.check(b"1.5"), matched(true, true));
        assert_eq!(number.check(b"1.0"), matched(true, true));
        assert_eq!(number.check(b"1.2"), Match::default());
        assert_eq!(number.check(b"-"), Match::default());
        assert_eq!(number.check(b"0"), Match::default());
        assert_eq!(number.check(b"01"), Match::default());
        assert_eq!(number.check(b"10"), matched(true, true));
        assert_eq!(number.check(b"11"), Match::default());
        assert!(number.accept(b"9.5").unwrap() && !number.accept(b"10.5").unwrap());
    }

    #[test]
    fn integers_and_failures() {
        let number = number([A, A, n("-3"), A, n("0.5")], true).unwrap();
        assert!(number.check(b"-2").complete);
        assert_eq!(number.check(b"-3"), Match::default());
        assert_eq!(number.check(b"1."), Match::default());
        use crate::error::Failure;
        assert_eq!(
            number_error([n("2"), n("1"), A, A, A]),
            Some(Failure::Empty(
                "numeric constraints describe an empty interval".into()
            ))
        );
        assert_eq!(
            number_error([n("1.1"), n("1.9"), A, A, A]),
            Some(Failure::Empty(
                "numeric constraints contain no multipleOf value".into()
            ))
        );
        assert!(Number::new(&[A, A, A, A, n("0.3")], true, 16).is_ok());
        let not_number = [Value::Other, A, A, A, A];
        assert_eq!(
            Number::new(&not_number, false, 16).err(),
            Some(Failure::Rejected("minimum must be a finite number".into()))
        );
    }

    fn number_error(keywords: [Value<'_>; 5]) -> Option<crate::error::Failure> {
        Number::new(&keywords, true, 16).err()
    }

    #[test]
    fn least_common_multiple() {
        assert_eq!(
            intersect_multiple_of(&n("0.4"), &n("0.6")).unwrap(),
            b"12e-1"
        );
        assert_eq!(intersect_multiple_of(&n("3"), &n("5")).unwrap(), b"15e0");
        assert!(intersect_multiple_of(&n("-1"), &n("5")).is_err());
        assert!(equal(b"1.20", b"12e-1").unwrap());
    }
}
