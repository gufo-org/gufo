fn layout_like_printf_g(negative: bool, digits: &str, exponent: i32, precision: i32) -> String {
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let mut text = String::with_capacity(digits.len() + 8);
    if negative {
        text.push('-');
    }
    if exponent < -4 || exponent >= precision {
        text.push_str(&digits[..1]);
        if digits.len() > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        text.push('e');
        text.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            text.push('0');
        }
        text.push_str(&magnitude.to_string());
    } else if exponent < 0 {
        text.push_str("0.");
        for _ in 0..(-exponent - 1) {
            text.push('0');
        }
        text.push_str(digits);
    } else {
        let point = exponent as usize + 1;
        if digits.len() <= point {
            text.push_str(digits);
            for _ in digits.len()..point {
                text.push('0');
            }
        } else {
            text.push_str(&digits[..point]);
            text.push('.');
            text.push_str(&digits[point..]);
        }
    }
    text
}

fn split_scientific(text: &str) -> (String, i32) {
    let (mantissa, exponent) = text.split_once('e').unwrap_or((text, "0"));
    let digits = mantissa.replace('.', "");
    (digits, exponent.parse().unwrap_or(0))
}

fn non_finite(value: f64) -> Option<&'static str> {
    if value.is_nan() {
        Some("NaN")
    } else if value == f64::INFINITY {
        Some("+Inf")
    } else if value == f64::NEG_INFINITY {
        Some("-Inf")
    } else {
        None
    }
}

pub fn printf_17g(value: f64) -> String {
    if let Some(text) = non_finite(value) {
        return text.to_string();
    }
    let (digits, exponent) = split_scientific(&format!("{:.16e}", value.abs()));
    layout_like_printf_g(value.is_sign_negative(), &digits, exponent, 17)
}

pub fn shortest_round_trip(value: f64) -> String {
    if let Some(text) = non_finite(value) {
        return text.to_string();
    }
    let (digits, exponent) = split_scientific(&format!("{:e}", value.abs()));
    layout_like_printf_g(value.is_sign_negative(), &digits, exponent, 17)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printf_17g_matches_c() {
        let cases: [(f64, &str); 18] = [
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (3.0, "3"),
            (0.1, "0.10000000000000001"),
            (1e-9, "1.0000000000000001e-09"),
            (1234.5678, "1234.5678"),
            (13.0 / 131072.0, "9.918212890625e-05"),
            (0.0001, "0.0001"),
            (0.00001, "1.0000000000000001e-05"),
            (1e16, "10000000000000000"),
            (1e17, "1e+17"),
            (1e20, "1e+20"),
            (123456789012.0 * 1000.0, "123456789012000"),
            (1.0 / 3.0, "0.33333333333333331"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "4.9406564584124654e-324"),
            (-2.5, "-2.5"),
        ];
        for (value, expected) in cases {
            assert_eq!(printf_17g(value), expected, "{value:e}");
        }
    }

    #[test]
    fn non_finite_values_use_exposition_names() {
        assert_eq!(printf_17g(f64::NAN), "NaN");
        assert_eq!(printf_17g(f64::INFINITY), "+Inf");
        assert_eq!(printf_17g(f64::NEG_INFINITY), "-Inf");
        assert_eq!(shortest_round_trip(f64::INFINITY), "+Inf");
    }

    #[test]
    fn shortest_round_trip_stays_short() {
        assert_eq!(shortest_round_trip(0.1), "0.1");
        assert_eq!(shortest_round_trip(0.005), "0.005");
        assert_eq!(shortest_round_trip(1.0), "1");
        assert_eq!(shortest_round_trip(10.0), "10");
        assert_eq!(shortest_round_trip(2.5e-7), "2.5e-07");
        assert_eq!(shortest_round_trip(1e21), "1e+21");
        for value in [0.1, 1234.5678, 1e-9, 7.0e22, 123456.789] {
            assert_eq!(shortest_round_trip(value).parse::<f64>().unwrap(), value);
        }
    }
}
